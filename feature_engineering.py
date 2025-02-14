import pandas as pd

def create_features(df, version='v2'):
    # df['Return_tick'] = pct_return(df, window='1s')  # 1 second window for tick return
    df['Return_1min'] = df.groupby(['InstrumentID', 'TradingDay'])[['LastPrice']]\
                        .transform(lambda x: pct_return(x, '1min'))
    df['Return_5min'] = df.groupby(['InstrumentID', 'TradingDay'])[['LastPrice']]\
                        .transform(lambda x: pct_return(x, '5min'))
    df['Return_10min'] = df.groupby(['InstrumentID', 'TradingDay'])[['LastPrice']]\
                        .transform(lambda x: pct_return(x, '10min'))
    df['Return_1h'] = df.groupby(['InstrumentID', 'TradingDay'])[['LastPrice']]\
                      .transform(lambda x: pct_return(x, '1h'))
    df['VolumeChange'] = df.groupby(['InstrumentID', 'TradingDay'])[['Volume']]\
                        .transform(lambda x: volume_change(x))
    if version != 'v1':
        df['LastPrice_return'] = df['LastPrice'].pct_change()

        # Calculate volatility using LastPrice returns instead of Return_1min (which is the label)
        df['Volatility_5min'] = df['LastPrice_return'].rolling(window=5, min_periods=1).std() * 100  # 5-minute rolling volatility
        df['RealizedVol_1min'] = (
            df['LastPrice_return'].abs()
            .resample('1min')
            .sum()
            .reindex(df.index, method='ffill')
        )  # 30-min realized volatility

        df['RealizedVol_5min'] = (
            df['LastPrice_return'].abs()
            .resample('5min')
            .sum()
            .reindex(df.index, method='ffill')
        )  # 30-min realized volatility

        df['RealizedVol_10min'] = (
            df['LastPrice_return'].abs()
            .resample('10min')
            .sum()
            .reindex(df.index, method='ffill')
        )  # 30-min realized volatility
        # Calculate number of trades (assuming each row represents a trade)
        df['Trades_1min'] = df.resample('1min', on='DataTime')['Volume'].transform('count')
        df['Trades_5min'] = df.resample('5min', on='DataTime')['Volume'].transform('count').ffill()

        # Calculate number of book orders (sum of all ask/bid volumes)
        df['TotalBookOrders'] = (df[[f'AskVolume{i}' for i in range(1,11)] + 
                                                    [f'BidVolume{i}' for i in range(1,11)]].sum(axis=1))
        df['BookOrders_1min'] = df.resample('1min', on='DataTime')['TotalBookOrders'].transform('sum').ffill()
    imbalance_series_1min = df.groupby(['InstrumentID', 'TradingDay']).apply(lambda group: imbalance_price(group, '1min'))
    df['ImbalancePrice_1min'] = imbalance_series_1min.reset_index(level=[0, 1], drop=True)
    
    imbalance_series_5min = df.groupby(['InstrumentID', 'TradingDay']).apply(lambda group: imbalance_price(group, '5min'))
    df['ImbalancePrice_5min'] = imbalance_series_5min.reset_index(level=[0, 1], drop=True)

    imbalance_series_10min = df.groupby(['InstrumentID', 'TradingDay']).apply(lambda group: imbalance_price(group, '10min'))
    df['ImbalancePrice_10min'] = imbalance_series_10min.reset_index(level=[0, 1], drop=True)

    wap_series = df.groupby(['InstrumentID', 'TradingDay']).apply(lambda group: weighted_imbalance_price(group))
    df['WeightedImbalancePrice'] = wap_series.reset_index(level=[0, 1], drop=True)

    df = time_bin(df)

    return df

def time_bin(df):
    # Ensure that the 'DataTime' column is in datetime format.
    df['DataTime'] = pd.to_datetime(df['DataTime'])

    # Create a new column that floors 'DataTime' to the nearest 30 minutes.
    df['TimeBinStart'] = df['DataTime'].dt.floor('5min')
    df['TimeBinNumber'] = df.groupby('TradingDay')['TimeBinStart'].transform(lambda x: pd.factorize(x)[0] + 1)
    return df

def imbalance_price(group, window):
    return group["AskPrice1"].rolling(window).mean() / group["BidPrice1"].rolling(window).mean() - 1

def weighted_imbalance_price(group):
    # Define column names for Ask and Bid levels 1 to 10
    ask_price_cols = [f'AskPrice{i}' for i in range(1, 11)]
    ask_volume_cols = [f'AskVolume{i}' for i in range(1, 11)]
    bid_price_cols = [f'BidPrice{i}' for i in range(1, 11)]
    bid_volume_cols = [f'BidVolume{i}' for i in range(1, 11)]
    
    # Compute the weighted average Ask Price by pairing corresponding columns
    weighted_ask = sum(group[price] * group[volume]
                       for price, volume in zip(ask_price_cols, ask_volume_cols))
    total_ask_volume = sum(group[volume] for volume in ask_volume_cols)
    group['WeightedAvgAskPrice'] = weighted_ask / total_ask_volume
    
    # Compute the weighted average Bid Price similarly:
    weighted_bid = sum(group[price] * group[volume]
                       for price, volume in zip(bid_price_cols, bid_volume_cols))
    total_bid_volume = sum(group[volume] for volume in bid_volume_cols)
    group['WeightedAvgBidPrice'] = weighted_bid / total_bid_volume
   
    return group['WeightedAvgAskPrice'] / group['WeightedAvgBidPrice'] - 1

def volume_change(df):
    return df.diff().fillna(0)

def pct_return(df, window):
    shifted = df.asof(df.index + pd.Timedelta(window))
    returns = (shifted.values-df.values) / df.values
    df[:] = returns
    return df.fillna(0)