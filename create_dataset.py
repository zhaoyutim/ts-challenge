import os
import glob
import pandas as pd
import numpy as np
import h5py
from datetime import datetime, time
from feature_engineering import create_features

def load_timeseries_data(stock_symbol=None, data_folder='data'):
    """
    Reads all h5 files in the given folder, extracts the stock symbol and date 
    from the filename (assumed to be '{stock_symbol}_{yyyymmdd}.h5'), reads the data 
    using h5py, and returns a consolidated pandas DataFrame with formatted DataTime
    and TradingDay columns.
    
    Args:
        stock_symbol (str or list, optional): A stock symbol or list of stock symbols to filter.
                                               If None, all files in the folder will be processed.
        data_folder (str): The folder where h5 files are located.
    
    Returns:
        pd.DataFrame: A DataFrame consolidating the data, with added columns for 'stock'.
                      The DataTime column is formatted as "year month day hour minute second millisecond",
                      and the TradingDay column is converted to a datetime object.
    """
    # If a single stock symbol is provided as a string, convert it to a list.
    if stock_symbol is not None and isinstance(stock_symbol, str):
        stock_symbol = [stock_symbol]
    
    all_data = []
    
    # List all .h5 files in the data folder.
    h5_files = glob.glob(os.path.join(data_folder, '*.h5'))
    
    for file in h5_files:
        base = os.path.basename(file)
        
        # Expect filenames of the form: {stock_symbol}_{yyyymmdd}.h5
        try:
            stock, date_part = base.split('_')
            date_str = date_part.replace('.h5', '')
            _ = datetime.strptime(date_str, '%Y%m%d')
        except Exception as e:
            print(f"Skipping file {base} due to naming format error: {e}")
            continue
        
        # If filtering by stock_symbol, skip files that do not match.
        if stock_symbol is not None and stock not in stock_symbol:
            continue
        
        # Read the h5 file using h5py.
        try:
            with h5py.File(file, 'r') as h5f:
                data_dict = {col_name: h5f[col_name][()] for col_name in h5f.keys()}
                df_temp = pd.DataFrame(data_dict).dropna()
        except Exception as e:
            print(f"Error reading {base}: {e}")
            continue

        all_data.append(df_temp)
    
    # Combine all the dataframes into one consolidated DataFrame.
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        
        # Convert TradingDay column to datetime if it exists,
        # assuming it's stored as an integer or string in "YYYYMMDD" format.
        if 'TradingDay' in df_all.columns:
            df_all['TradingDay'] = pd.to_datetime(df_all['TradingDay'].astype(str), format='%Y%m%d')
        
        # Convert and format the DataTime column if it exists.
        if 'DataTime' in df_all.columns:
            # Ensure DataTime is a string
            df_all['DataTime'] = df_all['DataTime'].astype(str)
            # Parse the first 14 characters as the base datetime (YYYYMMDDHHMMSS)
            base_dt = pd.to_datetime(df_all['DataTime'].str[:], format='%Y%m%d%H%M%S%f')

            df_all['DataTime'] = base_dt
            
        if 'Nano' in df_all.columns:
            # Convert to integer nanoseconds directly instead of using dt accessor
            df_all['Nano'] = df_all['Nano'].astype('int64')

        # Optionally sort the DataFrame by stock and TradingDay.
        if 'InstrumentID' in df_all.columns and 'Nano' in df_all.columns:
            df_all.sort_values(by=['InstrumentID', 'Nano'], inplace=True)
    else:
        df_all = pd.DataFrame()
    df_all.set_index(pd.DatetimeIndex(df_all['Nano']), inplace=True)
    df_all.sort_index(inplace=True)
    return df_all

def drop_zero_columns(df):
    zero_columns = []
    for col in df.columns:
        # Check if ALL values in column are zero using .all()
        if (df[col] == 0).all():
            zero_columns.append(col)
            print(f"Column '{col}' has only zeros")
            
    # After checking all columns
    print("\nColumns with only zeros:", zero_columns)

    # Filter the dataframe by dropping zero columns
    df_filtered = df.drop(columns=zero_columns)
    df_filtered = df_filtered[df_filtered['LastPrice'] != 0] 
    df_filtered = df_filtered[df_filtered['Volume'] != 0] 
    return df_filtered

def remove_lunch_break_data(df):
    """
    Removes rows that fall within the time period 11:30 to 13:00 for each day.
    
    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
    
    Returns:
        pd.DataFrame: Filtered DataFrame without records from 11:30 to 13:00.
    """
    lunch_start = time(11, 30)
    lunch_end = time(13, 0)
    # Create a condition that is True for rows NOT in the lunch period:
    condition = (df.index.time < lunch_start) | (df.index.time >= lunch_end)
    return df[condition]

def prepare_data(df, save_path="prepared_data.h5"):
    """
    Prepares the data from the input dataframe and saves it in HDF5 format.
    
    This function converts the dataframe into two NumPy arrays:
      - X_np with shape (n, t, f) where:
          n = number of stocks (grouped by InstrumentID)
          t = number of time stamps per stock (assumed uniform after sorting)
          f = number of features (all numeric columns except the excluded ones)
      - y_np with shape (n, t, 4) corresponding to:
          4 return metrics: Return_1min, Return_5min, Return_10min, Return_1h

    The arrays are saved in the HDF5 file specified by `save_path`
    under the datasets 'X_np' and 'y_np'.
    
    Parameters:
        df (pd.DataFrame): The input dataframe. Must include, at minimum, the
                           following columns:
                           - Feature columns (any not in the exclude list)
                           - 'Return_1min', 'Return_5min', 'Return_10min', 'Return_1h'
                           - 'DataTime', 'TradingDay', 'InstrumentID'
        save_path (str): Path to the HDF5 file where the data will be saved.
        
    Returns:
        X_np (np.ndarray): Array of features with shape (n, t, f).
        y_np (np.ndarray): Array of returns with shape (n, t, 4).
    """

    # Define the columns to exclude from the features.
    exclude_cols = [
        'Return_1min', 'Return_5min', 'Return_10min', 'Return_1h', 
        'Nano', 'DataTime', 'TradingDay', 'InstrumentID', 'TimeBinStart', 'TimeBinEnd'
    ]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X_list, y_list = [], []
    # Group the DataFrame by 'InstrumentID' and sort data in temporal order.
    for stock in df['InstrumentID'].unique():
        df_stock = df[df['InstrumentID'] == stock]
        X_stock = df_stock[feature_columns].to_numpy()  # Shape: (t, f)
        y_stock = df_stock[['Return_1min', 'Return_5min', 'Return_10min', 'Return_1h']].to_numpy()  # Shape: (t, 4)
        X_list.append(X_stock)
        y_list.append(y_stock)
    
    # Convert lists into NumPy arrays with shapes (n, t, f) and (n, t, 4)
    X_np = np.stack(X_list, axis=0)
    y_np = np.stack(y_list, axis=0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(f"{save_path}/features.npy", X_np)
    np.save(f"{save_path}/labels.npy", y_np)
    
    return X_np, y_np

def split_train_test_data(df, split_date="2022-09-30"):
    """
    Splits the DataFrame into training and test sets based on the given split_date.
    Rows with datetime (from the DataFrame's index) before the split_date are used for training,
    and rows on or after the split_date for testing.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        split_date (str): The cutoff date in "YYYY-MM-DD" format.

    Returns:
        tuple: (train_df, test_df)
    """
    split_timestamp = pd.Timestamp(split_date)
    train_df = df.loc[df.index < split_timestamp].copy()
    test_df = df.loc[df.index >= split_timestamp].copy()
    return train_df, test_df

if __name__ == '__main__':
    # Example usage: load only the specific stock symbols.
    df = load_timeseries_data(stock_symbol=["002521"])
    train_df, test_df = split_train_test_data(df, split_date="2022-09-30")
    train_df = drop_zero_columns(train_df)
    train_df = create_features(train_df)
    test_df = drop_zero_columns(test_df)
    test_df = create_features(test_df)
    
    # Split data into training and test sets using the datetime index.
    print("After splitting by datetime:")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    features = [col for col in train_df.columns if col not in 
                ['Return_1min', 'Return_5min', 'Return_10min', 'Return_1h', 
                 'Nano', 'DataTime', 'TradingDay', 'InstrumentID', 'TimeBinStart', 'TimeBinEnd']]
    print(train_df.columns)
    print("Length of features:", len(features))
    
    # Prepare and save data separately for training and test sets.
    prepare_data(train_df, save_path="dataset/train")
    prepare_data(test_df, save_path="dataset/test")
