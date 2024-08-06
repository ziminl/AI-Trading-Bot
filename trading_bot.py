import datetime
import os
import re
import time
import numpy as np
import pandas as pd
from pytz import timezone
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from alpaca.trading.requests import  MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide
from alpaca.data.timeframe import TimeFrame

def get_tickers():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')
    symbols_list=sp500['Symbol'].unique().tolist()

    return symbols_list

def map_tickers(tickers):
    return [ticker.replace('-', '.') for ticker in tickers]

def fetch_data(ticker, start_date, end_date, data_client):
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        start=start_date,
        end=end_date,
        timeframe=TimeFrame.Day
    )
    bar_data = data_client.get_stock_bars(request_params)

    return bar_data

def fetch_prices_for_ticker(data, ticker):
    data_date = [bar.timestamp.strftime('%Y-%m-%d') for bar in data[ticker]]
    close_prices = [bar.close for bar in data[ticker]]
    high_prices = [bar.high for bar in data[ticker]]
    low_prices = [bar.low for bar in data[ticker]]

    return data_date, close_prices, high_prices, low_prices

def download_data(tickers, start_date, end_date, data_client):
    all_data = []

    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date, data_client)
        data_date, close_prices, high_prices, low_prices = fetch_prices_for_ticker(data, ticker)

        ticker_df = pd.DataFrame({
            'date': data_date,
            'ticker': ticker,
            'close': close_prices,
            'high': high_prices,
            'low': low_prices
        })

        all_data.append(ticker_df)

    df = pd.concat(all_data)
    df.set_index(['date', 'ticker'], inplace=True)
    df.sort_index(inplace=True)

    close_prices_df = df['close'].unstack(level=-1)
    high_prices_df = df['high'].unstack(level=-1)
    low_prices_df = df['low'].unstack(level=-1)

    close_prices_df.ffill(inplace=True)
    high_prices_df.ffill(inplace=True)
    low_prices_df.ffill(inplace=True)

    data_close_price = close_prices_df.values
    data_close_price = data_close_price.T
    data_high_price = high_prices_df.values
    data_high_price = data_high_price.T
    data_low_price = low_prices_df.values
    data_low_price = data_low_price.T

    num_data_points = len(data_date)
    display_date_range = f"from {data_date[0]} to {data_date[-1]}"

    return data_date, data_close_price, data_high_price, data_low_price, num_data_points, display_date_range, tickers

def calculate_past_prices(data_close_price, data_high_price, data_low_price):
    past_14_close_prices = data_close_price[:, -14:]
    past_14_high_prices = data_high_price[:, -14:]
    past_14_low_prices = data_low_price[:, -14:]

    return past_14_close_prices, past_14_high_prices, past_14_low_prices

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x)
        self.sd = np.std(x)
        normalized_x = (x - self.mu)/self.sd

        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size, x.shape[1]), strides=(x.strides[0], x.strides[0], x.strides[1]))

    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    output = x[window_size:]
    
    return output

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=32, num_layers=2, output_size=None, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.relu(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x)

        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        x = self.dropout(x)
        predictions = self.linear_2(x)
        
        return predictions
    
def run_epoch(dataloader, model, optimizer, criterion, scheduler, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        x = x.to('cpu')
        y = y.to('cpu')

        out = model(x)
        loss = criterion(out, y)

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    lr = scheduler.get_last_lr()[0]

    return epoch_loss / len(dataloader), lr

def predict_next_day_prices(data_close_price):
    mean_value = np.nanmean(data_close_price)
    data_close_price[np.isnan(data_close_price)] = mean_value

    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)
    normalized_data_close_price = normalized_data_close_price.T

    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=20)
    data_y = prepare_data_y(normalized_data_close_price, window_size=20)

    split_index = int(data_y.shape[0] * 0.80)
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    input_size = data_x.shape[2]
    output_size = data_y.shape[1]

    model = LSTMModel(input_size=input_size, hidden_layer_size=128, num_layers=3, output_size=output_size, dropout=0.4)
    model = model.to('cpu')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    for epoch in range(100):
        loss_train, lr_train = run_epoch(train_dataloader, model, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader, model, optimizer, criterion, scheduler)

        scheduler.step()
        
    model.eval()
    x = torch.tensor(data_x_unseen).float().to('cpu')
    x = x.unsqueeze(0) 
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    predicted_next_day_prices = scaler.inverse_transform(prediction.squeeze())
    current_close_prices = data_close_price[:, -1]

    return predicted_next_day_prices, current_close_prices

def calculate_indicators(tickers, current_close_prices, predicted_next_day_prices, past_14_high_prices, past_14_low_prices, past_14_close_prices):
    df = pd.DataFrame(tickers, columns=['ticker'])
    df['current_close_price'] = current_close_prices
    df['predicted_next_day_price'] = predicted_next_day_prices

    def calculate_mar(df, past_14_close_prices):
        mar_values = []
        for i, ticker in enumerate(df['ticker']):
            start_price = past_14_close_prices[i][0]
            end_price = past_14_close_prices[i][-1]
            mar = (end_price / start_price) - 1
            mar_values.append(mar)
        
        return np.array(mar_values)

    df['mar'] = calculate_mar(df, past_14_close_prices)
    df['threshold_long'] = df['mar'] * 0.2
    df['threshold_short'] = df['mar'] * -0.2

    def calculate_atr(past_14_high_prices, past_14_low_prices, past_14_close_prices):
        num_tickers = past_14_high_prices.shape[0]
        num_days = past_14_high_prices.shape[1]

        true_ranges = np.zeros((num_tickers, num_days - 1))

        for i in range(num_tickers):
            for j in range(1, num_days):
                high = past_14_high_prices[i, j]
                low = past_14_low_prices[i, j]
                prev_close = past_14_close_prices[i, j - 1]

                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges[i, j - 1] = tr

        atr_values = np.mean(true_ranges, axis=1) 

        return atr_values

    df['atr'] = calculate_atr(past_14_high_prices, past_14_low_prices, past_14_close_prices)

    tickers = df['ticker'].tolist()
    threshold_long = df['threshold_long'].values
    threshold_short = df['threshold_short'].values
    atr_values = df['atr'].values

    return threshold_long, threshold_short, atr_values, tickers

def get_entry_prices(tickers, data_client):
    entry_prices = {}
    
    start = datetime.datetime.now() - datetime.timedelta(days=1)
    end = datetime.datetime.now()
    
    for ticker in tickers:
        request_params = StockTradesRequest(
            symbol_or_symbols=ticker,
            start=start,
            end=end,
            limit=1,
        )

        trade_data = data_client.get_stock_trades(request_params)
        trade_data_str = str(trade_data[ticker][0])
        match = re.search(r"price=(\d+\.\d+)", trade_data_str)
        
        price = float(match.group(1))
        entry_prices[ticker] = price


    return entry_prices

def calculate_entries(predicted_next_day_prices, current_close_prices, threshold_long, threshold_short, tickers):
    long_entries = []
    short_entries = []

    for i, ticker in enumerate(tickers):
        predicted_price = predicted_next_day_prices[i]
        current_close_price = current_close_prices[i]
        long_threshold = threshold_long[i]
        short_threshold = threshold_short[i]

        if (predicted_price - current_close_price) / current_close_price >= long_threshold:
            long_entries.append(ticker)

        elif (predicted_price - current_close_price) / current_close_price <= short_threshold:
            short_entries.append(ticker)

    return long_entries, short_entries

def get_positions(positions):
    long_positions = []
    short_positions = []

    for position in positions:
        symbol = position.symbol
        side = position.side

        if side == PositionSide.LONG:
            long_positions.append(symbol)
        elif side == PositionSide.SHORT:
            short_positions.append(symbol)

    return long_positions, short_positions

def calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss_price, max_allocation=0.001):
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        risk_per_share = entry_price * 0.01

    position_size = risk_per_trade / risk_per_share
    max_position_size = (portfolio_value * max_allocation) / entry_price

    if position_size > max_position_size:
        position_size = max_position_size

    position_size = int(position_size)
    
    return position_size

def buy_long_entries(trading_client, entry_prices, long_entries, long_positions, max_allocation=0.01):
    for ticker in long_entries:

        if ticker not in long_positions:
            portfolio_value = float(trading_client.get_account().portfolio_value)
            entry_price = entry_prices[ticker]
            risk_per_trade = portfolio_value * 0.0001
            stop_loss_price = entry_price * 0.9
            position_size = calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss_price)

            if position_size * entry_price > portfolio_value * max_allocation:
                position_size = int((portfolio_value * max_allocation) / entry_price)

            try:
                if position_size > 0:
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=position_size,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )

                    buy_order = trading_client.submit_order(market_order_data)

            except Exception:
                pass

def sell_short_entries(trading_client, entry_prices, short_entries, short_positions, max_allocation=0.001):
    for ticker in short_entries:

        if ticker not in short_positions:
            portfolio_value = float(trading_client.get_account().portfolio_value)
            entry_price = entry_prices[ticker]
            risk_per_trade = portfolio_value * 0.0001
            stop_loss_price = entry_price * 1.1
            position_size = calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss_price)

            if position_size * entry_price > portfolio_value * max_allocation:
                position_size = int((portfolio_value * max_allocation) / entry_price)

            try: 
                if position_size > 0:
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=position_size,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )

                    sell_order = trading_client.submit_order(market_order_data)

            except Exception:
                pass

def get_current_prices(tickers, data_client):
    current_prices = {}

    start = datetime.datetime.now() - datetime.timedelta(days=1)
    end = datetime.datetime.now()

    for ticker in tickers:
        
        request_params = StockTradesRequest(
            symbol_or_symbols=ticker,
            start=start,
            end=end,
            limit=1,
        )

        trade_data = data_client.get_stock_trades(request_params)
        
        trade_data_str = str(trade_data[ticker][0])
        match = re.search(r"price=(\d+\.\d+)", trade_data_str)
        
        price = float(match.group(1))
        current_prices[ticker] = price
    
    return current_prices

def calculate_exits(entry_prices, current_prices, atr_values, long_positions, short_positions, trailing_stop_percentage=0.1, atr_multiple_stop_loss=1):
    long_exits = []
    short_exits = []

    for i, ticker in enumerate(long_positions):
        current_price = current_prices[ticker]
        entry_price = entry_prices[ticker]
        atr = atr_values[i]

        highest_price = max(entry_price, current_price)
        trailing_stop_long = highest_price * (1 - trailing_stop_percentage)
        stop_loss_long = entry_price - (atr_multiple_stop_loss * atr)

        if current_price <= trailing_stop_long or current_price <= stop_loss_long:
            long_exits.append(ticker)

    for i, ticker in enumerate(short_positions):
        current_price = current_prices[ticker]
        entry_price = entry_prices[ticker]
        atr = atr_values[i]
        
        lowest_price = min(entry_price, current_price)
        trailing_stop_short = lowest_price * (1 + trailing_stop_percentage)
        stop_loss_short = entry_price + (atr_multiple_stop_loss * atr)

        if current_price >= trailing_stop_short or current_price >= stop_loss_short:
            short_exits.append(ticker)

    return long_exits, short_exits

def liquidate_positions(trading_client, long_exits, short_exits, long_positions, short_positions):
    for ticker in long_exits:
        if ticker in long_positions:
            trading_client.close_position(ticker)

    for ticker in short_exits:
        if ticker in short_positions:
            trading_client.close_position(ticker)

def update_data(data_client, data_close_price, data_high_price, data_low_price, tickers):
    close_prices = []
    high_prices = []
    low_prices = []

    start = datetime.datetime.now() - datetime.timedelta(days=1)
    end = datetime.datetime.now()

    for ticker in tickers:
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )

        data = data_client.get_stock_bars(request_params)
        
        close_price = [bar.close for bar in data[ticker]]
        high_price = [bar.high for bar in data[ticker]]
        low_price = [bar.low for bar in data[ticker]]

        close_prices.append(close_price[0])
        high_prices.append(high_price[0])
        low_prices.append(low_price[0])

    new_data_close_price = np.empty((data_close_price.shape[0], data_close_price.shape[1] + 1))
    new_data_close_price[:, :-1] = data_close_price
    new_data_close_price[:, -1] = np.nan

    new_data_high_price = np.empty((data_high_price.shape[0], data_high_price.shape[1] + 1))
    new_data_high_price[:, :-1] = data_high_price
    new_data_high_price[:, -1] = np.nan

    new_data_low_price = np.empty((data_low_price.shape[0], data_low_price.shape[1] + 1))
    new_data_low_price[:, :-1] = data_low_price
    new_data_low_price[:, -1] = np.nan

    for i in range(len(close_prices)):
        new_data_close_price[i, -1] = close_prices[i]
        new_data_high_price[i, -1] = high_prices[i]
        new_data_low_price[i, -1] = low_prices[i]
    
    return new_data_close_price, new_data_high_price, new_data_low_price

def main():
    FLAG_FILE_PATH = 'initial_data_download.txt'

    load_dotenv()

    ALPACA_DATA_KEY = os.getenv('ALPACA_DATA_KEY')
    ALPACA_DATA_SECRET = os.getenv('ALPACA_DATA_SECRET')
    ALPACA_TRADING_KEY = os.getenv('ALPACA_TRADING_KEY')
    ALPACA_TRADING_SECRET = os.getenv('ALPACA_TRADING_SECRET')

    data_client = StockHistoricalDataClient(ALPACA_DATA_KEY, ALPACA_DATA_SECRET)
    trading_client = TradingClient(ALPACA_TRADING_KEY, ALPACA_TRADING_SECRET, paper=True)

    data_close_price = None
    data_high_price = None
    data_low_price = None

    tickers = None

    if not os.path.exists(FLAG_FILE_PATH):
        start_date = datetime.datetime(2016, 1, 1)
        end_date = datetime.datetime.now()

        tickers = get_tickers()
        tickers = map_tickers(tickers)

        data_date, data_close_price, data_high_price, data_low_price, num_data_points, display_date_range, tickers = download_data(tickers, start_date, end_date, data_client)
        
        with open(FLAG_FILE_PATH, 'w') as f:
            f.write('Initial data download complete')

    predicted_next_day_prices = None
    current_close_prices = None
    past_14_close_prices = None
    past_14_high_prices = None
    past_14_low_prices = None
    threshold_long = None
    threshold_short = None
    atr_values = None
    entry_prices = None
    positions = None
    long_positions = None
    short_positions = None
    current_prices = None
    long_entries = None
    short_entries = None
    long_exits = None
    short_exits = None
    
    while True:
        now = datetime.datetime.now(timezone('America/New_York'))

        if now.weekday() >= 0 and now.weekday() <= 4:

            if now.hour == 9 and now.minute == 30:
                predicted_next_day_prices, current_close_prices = predict_next_day_prices(data_close_price)

                time.sleep(60)

            elif now.hour == 10 and now.minute == 30:
                past_14_close_prices, past_14_high_prices, past_14_low_prices = calculate_past_prices(data_close_price, data_high_price, data_low_price)

                threshold_long, threshold_short, atr_values, tickers = calculate_indicators(tickers, current_close_prices, predicted_next_day_prices, past_14_high_prices, past_14_low_prices, past_14_close_prices)

                entry_prices = get_entry_prices(tickers, data_client)

                long_entries, short_entries = calculate_entries(predicted_next_day_prices, current_close_prices, threshold_long, threshold_short, tickers)

                positions = trading_client.get_all_positions()
                long_positions, short_positions = get_positions(positions)

                buy_long_entries(trading_client, entry_prices, long_entries, long_positions)
                sell_short_entries(trading_client, entry_prices, short_entries, short_positions)

                time.sleep(60)

            elif now.hour == 15 and now.minute == 35:
                current_prices = get_current_prices(tickers, data_client)

                positions = trading_client.get_all_positions()
                long_positions, short_positions = get_positions(positions)

                long_exits, short_exits = calculate_exits(entry_prices, current_prices, atr_values, long_positions, short_positions, trailing_stop_percentage=0.1, atr_multiple_stop_loss=1)

                liquidate_positions(trading_client, long_exits, short_exits, long_positions, short_positions)

                time.sleep(60)

            elif now.hour == 16 and now.minute == 30:
                data_close_price, data_high_price, data_low_price = update_data(data_client, data_close_price, data_high_price, data_low_price, tickers)
                
                time.sleep(60)
            
        time.sleep(1)

if __name__ == '__main__':
    main()
