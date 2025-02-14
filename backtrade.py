import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
import os
from create_dataset import load_timeseries_data, split_train_test_data, drop_zero_columns, create_features
from catboost import CatBoostRegressor
import ipywidgets as widgets
from ipywidgets import interact

class BackTradeTester:
    def __init__(self, model, initial_capital=10000.0, trade_size=1, transaction_cost=0.0):
        """
        Initializes the backtesting framework.
        
        Parameters:
            model: Pretrained ML model or list of models with a .predict() method.
            initial_capital (float): Starting cash amount.
            trade_size (int): Number of shares per trade.
            transaction_cost (float): Fixed cost per executed trade (applied when closing a position).
        """
        self.model = model
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0          # 1 for long, -1 for short, 0 for neutral
        self.entry_price = None    # price at which current position was opened
        self.trade_size = trade_size
        self.transaction_cost = transaction_cost
        
        self.trade_history = []    # list to record all trade events
        self.equity_curve = []     # equity value at each tick
        self.timestamps = []       # corresponding timestamps for equity curve

    def execute_trade(self, new_position, price, timestamp):
        """
        Closes an existing position (if any) and opens a new position given a new signal.
        Records the trade events in trade_history.
        """
        if self.position == new_position:
            return  # no new trade is required
        
        # Close any existing position
        if self.position != 0:
            # Calculate profit/loss: profit = position * (exit_price - entry_price) * trade_size
            profit = self.position * (price - self.entry_price) * self.trade_size - self.transaction_cost
            self.cash += profit
            trade_exit = {
                'timestamp': timestamp,
                'action': 'CLOSE',
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': price,
                'profit': profit
            }
            self.trade_history.append(trade_exit)
            # Reset position
            self.position = 0
            self.entry_price = None

        # Open a new position if required
        if new_position != 0:
            self.entry_price = price
            self.position = new_position
            trade_open = {
                'timestamp': timestamp,
                'action': 'OPEN',
                'position': new_position,
                'price': price
            }
            self.trade_history.append(trade_open)

    def update_equity(self, price):
        """
        Computes the mark-to-market equity based on current cash and open positions.
        """
        if self.position != 0 and self.entry_price is not None:
            # Unrealized PnL = position * (current_price - entry_price) * trade_size
            equity = self.cash + self.position * (price - self.entry_price) * self.trade_size
        else:
            equity = self.cash
        return equity

    def get_signal(self, features):
        """
        Calls the pretrained ML model(s) to obtain a trading signal.
        Only allows orders to be executed if the DataTime is before 10:00.
        
        Parameters:
            features: A pd.Series containing tick data (typically without non-feature fields such as timestamp).
            
        Returns:
            A string signal: "LONG", "SHORT", or "HOLD".
        """
        # Check the DataTime field to determine the trading window.
        # If DataTime exists and is 10:00 or later, return "HOLD" immediately.
        dt_value = features.get("DataTime", None)
        if dt_value is not None:
            import datetime
            # Convert dt_value to a datetime object if it is not already one.
            if not isinstance(dt_value, (datetime.datetime, pd.Timestamp)):
                dt_parsed = pd.to_datetime(dt_value)
            else:
                dt_parsed = dt_value

            threshold_time = datetime.time(10, 0)
            if dt_parsed.time() >= threshold_time:
                return "HOLD"

        # Drop non-feature columns that should not be used for prediction.
        features = features.drop(labels=['Nano', 'DataTime', 'TradingDay', 'InstrumentID', 'TimeBinStart', 'TimeBinEnd'], errors='ignore')
        features_array = features.values

        # If self.model is a list of models, aggregate their predictions.
        if isinstance(self.model, list):
            predictions = [model.predict(features_array) for model in self.model]
            ensemble_pred = sum(predictions)  # You may also choose to average these predictions.
        else:
            ensemble_pred = self.model.predict(features_array)

        # Convert the numeric prediction into a trading signal based on a threshold.
        threshold = 0.01
        if ensemble_pred > threshold:
            return "LONG"
        elif ensemble_pred < -threshold:
            return "SHORT"
        else:
            return "HOLD"

    def run_backtest(self, tick_data):
        """
        Runs the backtest simulation over the provided tick-level data.
        
        Parameters:
            tick_data (pd.DataFrame): Must contain at least 'timestamp' and 'price' columns.
            
        Returns:
            Tuple of (equity_curve, trade_history).
        """
        for idx, row in tick_data.iterrows():
            timestamp = row['Nano']
            price = row['LastPrice']
            signal = self.get_signal(row)
            
            if signal == 'LONG':
                self.execute_trade(new_position=100, price=price, timestamp=timestamp)
            elif signal == 'SHORT':
                self.execute_trade(new_position=-100, price=price, timestamp=timestamp)
            elif signal == 'HOLD':
                # Do nothing; maintain current position
                pass
            else:
                # If an unrecognized signal is returned, you might want to log or ignore
                pass

            # Update and record the current equity
            current_equity = self.update_equity(price)
            self.equity_curve.append(current_equity)
            self.timestamps.append(pd.to_datetime(timestamp, unit='ns'))
        
        # At the end of the backtest, close any open position using the final tick price.
        if self.position != 0:
            final_tick = tick_data.iloc[-1]
            self.execute_trade(new_position=0, price=final_tick['LastPrice'], timestamp=final_tick['Nano'])
            # Update the last equity point
            self.equity_curve[-1] = self.update_equity(final_tick['LastPrice'])
            
        return self.equity_curve, self.trade_history

def load_model(model_path):
    """
    Loads a pretrained model or models.
    
    If model_path is a directory, loads all CatBoost models (.cbm files) found within.
    Otherwise, attempts to load a single model using pickle.
    
    Parameters:
        model_path (str): Path to a pickle file or a directory containing CatBoost .cbm files.
        
    Returns:
        A single model or a list of models.
    """
    # (Optional) Import locally to avoid any naming shadowing
    from catboost import CatBoostRegressor

    if os.path.isdir(model_path):
        models = []
        for file in os.listdir(model_path):
            if file.endswith('.cbm'):
                full_path = os.path.join(model_path, file)
                model_instance = CatBoostRegressor()
                model_instance.load_model(full_path)
                print(f"Loaded CatBoost model from {full_path}")
                models.append(model_instance)
        return models
    else:
        model_instance = CatBoostRegressor()
        model_instance.load_model(model_path)
        print(f"Loaded CatBoost model from {model_path}")
        return [model_instance]

def load_tick_data(file_path):
    """
    Loads tick-level data from a CSV file into a pandas DataFrame.
    Expects at least the following columns: 'timestamp' and 'price'.
    
    Parameters:
        file_path (str): Path to the CSV data file.
        
    Returns:
        pd.DataFrame: DataFrame containing the tick data.
    """
    data = pd.read_csv(file_path, parse_dates=['timestamp'])
    return data

def plot_equity_curve(timestamps, equity_curve, price_series, trade_history=None):
    """
    Plots the equity curve (total capital) together with the real time asset price.
    The equity curve is plotted against the left y-axis, and the real time price is
    mapped to the right y-axis. Additionally, if trade_history is provided, the function
    will mark the trade entries (OPEN events):
        - LONG trades (position == 100) are marked with an upward green triangle.
        - SHORT trades (position == -100) are marked with a downward red triangle.
    
    Parameters:
        timestamps (List[pd.Timestamp]): List of timestamps corresponding to the data points.
        equity_curve (List[float]): List of equity (capital) values.
        price_series (List[float]): List of real time asset prices.
        trade_history (List[dict], optional): List of trade events.
            Only trade events with action "OPEN" will be marked.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # If trade_history is provided, filter the data to only include days with trades.
    if trade_history is not None:
        # Set of days with at least one OPEN trade.
        trade_days = {
            pd.to_datetime(trade["timestamp"], unit="ns").date()
            for trade in trade_history if trade.get("action") == "OPEN"
        }
        filtered_timestamps = [ts for ts in timestamps if ts.date() in trade_days]
        filtered_equity = [eq for ts, eq in zip(timestamps, equity_curve) if ts.date() in trade_days]
        filtered_prices = [price for ts, price in zip(timestamps, price_series) if ts.date() in trade_days]
    else:
        filtered_timestamps = timestamps
        filtered_equity = equity_curve
        filtered_prices = price_series

    # Create the primary plot.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(filtered_timestamps, filtered_equity, label="Equity", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Total Capital", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a secondary y-axis for real time price.
    ax2 = ax1.twinx()
    ax2.plot(filtered_timestamps, filtered_prices, label="Real Time Price", color="orange")
    ax2.set_ylabel("Real Time Price", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Mark the trades on the equity curve.
    if trade_history is not None:
        long_times = []
        long_equity = []
        short_times = []
        short_equity = []
        for trade in trade_history:
            if trade.get("action") == "OPEN":
                trade_time = pd.to_datetime(trade["timestamp"], unit="ns")
                # Find the indices in filtered_timestamps corresponding to the same day.
                matching_indices = [
                    i for i, ts in enumerate(filtered_timestamps) if ts.date() == trade_time.date()
                ]
                if not matching_indices:
                    continue
                # Choose the index with minimum difference in seconds.
                best_index = min(
                    matching_indices,
                    key=lambda i: abs((filtered_timestamps[i] - trade_time).total_seconds())
                )
                eq_val = filtered_equity[best_index]
                if trade.get("position", 0) == 100:
                    long_times.append(filtered_timestamps[best_index])
                    long_equity.append(eq_val)
                elif trade.get("position", 0) == -100:
                    short_times.append(filtered_timestamps[best_index])
                    short_equity.append(eq_val)
        if long_times:
            ax1.scatter(long_times, long_equity, marker="^", color="green",
                        s=10, label="LONG")
        if short_times:
            ax1.scatter(short_times, short_equity, marker="v", color="red",
                        s=10, label="SHORT")
    
    ax1.set_title("Equity Curve & Real Time Price (Days with Trades Only)")
    ax1.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def plot_specific_day(day, timestamps, equity_curve, price_series, trade_history=None):
    """
    Plots the equity curve and real time price with trade markers for a specified day.
    
    Parameters:
        day (datetime.date): The day to visualize.
        timestamps (List[pd.Timestamp]): List of all tick timestamps.
        equity_curve (List[float]): Equity values corresponding to the timestamps.
        price_series (List[float]): Real time price values corresponding to the timestamps.
        trade_history (List[dict], optional): Trade events to mark on the plot.
            Only "OPEN" events are marked with LONG and SHORT indicators.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Filter the data to the chosen day.
    filtered_timestamps = [ts for ts in timestamps if ts.date() == day]
    filtered_equity = [eq for ts, eq in zip(timestamps, equity_curve) if ts.date() == day]
    filtered_prices = [price for ts, price in zip(timestamps, price_series) if ts.date() == day]
    
    if not filtered_timestamps:
        print(f"No data available for {day}")
        return

    # Create the primary plot for the equity curve.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(filtered_timestamps, filtered_equity, label="Equity", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    
    # Create the secondary y-axis for real time price.
    ax2 = ax1.twinx()
    ax2.plot(filtered_timestamps, filtered_prices, label="Real Time Price", color="orange")
    ax2.set_ylabel("Real Time Price", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    
    # Mark trades on the equity curve.
    if trade_history:
        long_times = []
        long_equity = []
        short_times = []
        short_equity = []
        for trade in trade_history:
            if trade.get("action") == "OPEN":
                trade_time = pd.to_datetime(trade["timestamp"], unit="ns")
                if trade_time.date() == day:
                    try:
                        idx = filtered_timestamps.index(trade_time)
                        eq_val = filtered_equity[idx]
                    except ValueError:
                        continue
                    if trade.get("position", 0) == 100:
                        long_times.append(trade_time)
                        long_equity.append(eq_val)
                    elif trade.get("position", 0) == -100:
                        short_times.append(trade_time)
                        short_equity.append(eq_val)
        if long_times:
            ax1.scatter(long_times, long_equity, marker="^", color="green", s=100, label="LONG")
        if short_times:
            ax1.scatter(short_times, short_equity, marker="v", color="red", s=100, label="SHORT")

    ax1.set_title(f"Equity Curve and Real Time Price for {day}")
    ax1.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def interactive_day_plot(timestamps, equity_curve, price_series, trade_history=None):
    """
    Displays an interactive widget that allows you to choose a day
    to visualize the corresponding equity curve, real time price, and trade markers.
    
    Parameters:
        timestamps (List[pd.Timestamp]): Full list of simulation timestamps.
        equity_curve (List[float]): The corresponding equity curve values.
        price_series (List[float]): The corresponding real time price values.
        trade_history (List[dict], optional): Trade events for marking on the plot.
    """
    from ipywidgets import interact, widgets

    # Extract unique days from the timestamps.
    unique_days = sorted({ts.date() for ts in timestamps})
    day_selector = widgets.Dropdown(
        options=unique_days,
        description="Day:",
    )
    interact(plot_specific_day,
             day=day_selector,
             timestamps=widgets.fixed(timestamps),
             equity_curve=widgets.fixed(equity_curve),
             price_series=widgets.fixed(price_series),
             trade_history=widgets.fixed(trade_history))

def main():
    parser = argparse.ArgumentParser(description='BackTrade Testing Framework')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pretrained model pickle file or directory containing CatBoost .cbm files')
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital for the backtest')
    parser.add_argument('--trade_size', type=int, default=1, help='Number of shares per trade')
    parser.add_argument('--transaction_cost', type=float, default=0.0, help='Transaction cost per trade (applied on close)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve at the end of the backtest')
    args = parser.parse_args()
    
    df = load_timeseries_data(stock_symbol=["002521"])
    _, test_df = split_train_test_data(df, split_date="2022-09-30")
    test_df = drop_zero_columns(test_df)
    test_df = create_features(test_df)
    
    print("After splitting by datetime:")
    print(f"Test set shape: {test_df.shape}")
    
    features = [col for col in df.columns if col not in 
                ['Return_1min', 'Return_5min', 'Return_10min', 'Return_1h', 
                 'Nano', 'DataTime', 'TradingDay', 'InstrumentID', 'TimeBinStart', 'TimeBinEnd']]
    print(df.columns)
    print("Length of features:", len(features))
    
    model = load_model(args.model)
    
    # Initialize and run the backtest
    tester = BackTradeTester(model,
                             initial_capital=args.initial_capital,
                             trade_size=args.trade_size,
                             transaction_cost=args.transaction_cost)
    equity_curve, trades = tester.run_backtest(test_df)
    
    # Get the real time price series from the tick data (for example, 'LastPrice')
    # Ensure the order matches the timestamps.
    price_series = test_df["LastPrice"].tolist()
    
    print("Final Equity: {:.2f}".format(equity_curve[-1]))
    print("Trade History:")
    for trade in trades:
        print(trade)
    
    if args.plot:
        # Pass the price_series as the third argument and trade_history as the fourth.
        plot_equity_curve(tester.timestamps, equity_curve, price_series, trades)

if __name__ == '__main__':
    main()