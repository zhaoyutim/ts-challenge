import os
import glob
import pandas as pd
import h5py
from datetime import datetime

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
                df_temp = pd.DataFrame(data_dict)
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
            base_dt = pd.to_datetime(df_all['DataTime'].str[:14], format='%Y%m%d%H%M%S%f')

            df_all['DataTime'] = base_dt
        
        # Optionally sort the DataFrame by stock and TradingDay.
        if 'InstrumentID' in df_all.columns and 'TradingDay' in df_all.columns:
            df_all.sort_values(by=['InstrumentID', 'TradingDay'], inplace=True)
    else:
        df_all = pd.DataFrame()

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
    return df_filtered

if __name__ == '__main__':
    # Example usage: load only the two specific stock symbols.
    df = load_timeseries_data(stock_symbol=["002521", "300132"])
    df = drop_zero_columns(df)
    print(df.head())

