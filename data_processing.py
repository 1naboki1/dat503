import os
import json
import logging
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask_ml.preprocessing import DummyEncoder
from tqdm import tqdm

def load_and_preprocess_data(train_folder, train_filters, output_file_path, exclude_columns, delimiter=';'):
    """
    Load and preprocess data from the specified folder.

    Parameters:
    train_folder (str): The folder containing the CSV files to be processed.
    filters (dict): A dictionary of filters to apply to the data.
    output_file_path (str): The file path where the processed data should be saved.
    exclude_columns (list): A list of columns to exclude from the data.
    delimiter (str): The delimiter used in the CSV files.

    Returns:
    str: The path to the saved Parquet file, or None if an error occurred.
    """
    
    data_files = _get_csv_files(train_folder)
    logging.debug(f"Found {len(data_files)} CSV files to process: {data_files}")
    
    data = _load_data_into_dask_dataframe(data_files, delimiter, exclude_columns)
    if data is None:
        return None
    print(f"\033[92m✔ {len(data_files)} files loaded successfully.\033[0m")

    data = _apply_filters(data, train_filters)
    if data is None:
        return None
    
    data = _fill_missing(data)
    if data is None:
        return None
    
    data = _calculate_time_differences(data)
    if data is None:
        return None
    
    data = _transform_betriebstag(data)
    if data is None:
        return None
    
    #data = preprocess_data(data)
    #if data is None:
        #return None

    return _save_data(data, output_file_path)

"""def preprocess_data(data):

    Preprocess the data by applying filters, excluding columns, and encoding categorical columns.

    Parameters:
    data (dask.dataframe.DataFrame): The data to be processed.
    return [entry.path for entry in os.scandir(folder) if entry.is_file() and entry.name.endswith('.csv')]
    exclude_columns (list): A list of columns to exclude from the data.

    Returns:
    dask.dataframe.DataFrame: The preprocessed data, or None if an error occurred.

    
    data = _encode_categorical_columns(data)
    if data is None:
        return None
    
    return data"""

def _get_csv_files(folder):
    """
    Get a list of CSV files in the specified folder.

    Parameters:
    folder (str): The folder to search for CSV files.

    Returns:
    list: A list of file paths to the CSV files in the folder.
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def _load_data_into_dask_dataframe(data_files, delimiter, exclude_columns):
    """
    The function reads multiple CSV files into Dask DataFrames, drops specified columns,
    concatenates them into a single Dask DataFrame, and logs various details about the process.

    Load data from CSV files into a Dask DataFrame.
    Parameters:
    data_files (list): A list of file paths to the CSV files.
    delimiter (str): The delimiter used in the CSV files.
    exclude_columns (list): A list of column names to exclude from the DataFrame.

    Returns:
    dask.dataframe.DataFrame: The loaded data, or None if an error occurred.
    """
    try:
        logging.info("Loading data into Dask DataFrame.")
        dtype = {
            'BETRIEBSTAG': 'object',
            'FAHRT_BEZEICHNER': 'object',
            'BETREIBER_ID': 'object',
            'BETREIBER_ABK': 'object',
            'BETREIBER_NAME': 'object',
            'PRODUKT_ID': 'object',
            'LINIEN_ID': 'object',
            'LINIEN_TEXT': 'object',
            'UMLAUF_ID': 'object',
            'VERKEHRSMITTEL_TEXT': 'object',
            'ZUSATZFAHRT_TF': 'object',
            'FAELLT_AUS_TF': 'object',
            'BPUIC': 'object',
            'HALTESTELLEN_NAME': 'object',
            'ANKUNFTSZEIT': 'object',
            'AN_PROGNOSE': 'object',
            'AN_PROGNOSE_STATUS': 'object',
            'ABFAHRTSZEIT': 'object',
            'AB_PROGNOSE': 'object',
            'AB_PROGNOSE_STATUS': 'object',
            'DURCHFAHRT_TF': 'object',
        }
        
        data_list = []
        total_rows = 0
        
        with tqdm(total=len(data_files), desc="Loading CSV files") as pbar:
            for file in data_files:
                df = dd.read_csv(file, delimiter=delimiter, dtype=dtype, assume_missing=True, blocksize=None)
                df = df.drop(columns=exclude_columns, errors='ignore')
                data_list.append(df)
                total_rows = total_rows + len(df)
                pbar.update(1)
        
        logging.debug(f"Total number of rows across all files: {total_rows}")
        
        data = dd.concat(data_list, axis=0)

        logging.debug(f"Total partitions: {data.npartitions}")
        logging.debug(f"Columns: {data.columns}")
        logging.debug(f"Data types: {data.dtypes}")
        logging.debug(f"Number of rows: {len(data)}")
        logging.info("Data loaded into Dask DataFrame.")
        return data
    except Exception as e:
        logging.error(f"Error loading data into Dask DataFrame from files {data_files}: {e}")
    return None

def _apply_filters(data, train_filters):
    """
    Apply filters to the data.

    Parameters:
    data (dask.dataframe.DataFrame): The data to be filtered.
    filters (dict): A dictionary of filters to apply.

    Returns:
    dask.dataframe.DataFrame: The filtered data, or None if an error occurred.
    """
    try:
        if train_filters:
            for column, values in train_filters.items():
                if not isinstance(values, list):
                    values = [values]
                logging.debug(f"Applying filter: {column} in {values}")
                try:
                    if column in data.columns:
                        data = data[data[column].isin(values)]
                        logging.debug(f"Filter applied on column: {column}")
                    else:
                        logging.debug(f"Column {column} does not exist in the data.")
                        return None
                except Exception as e:
                    logging.debug(f"Error applying filter on column {column}: {e}")
                    return None

        logging.info("Applying custom filters for AN_PROGNOSE_STATUS and AB_PROGNOSE_STATUS.")
        data = data[(data["AN_PROGNOSE_STATUS"] == "REAL") & (data["AB_PROGNOSE_STATUS"] == "REAL")]
        logging.info("Custom filters applied.")
        
        # Drop the columns after applying the filters
        data = data.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"], errors='ignore')
        logging.info("Dropped columns AN_PROGNOSE_STATUS and AB_PROGNOSE_STATUS.")

        logging.debug(f"Total partitions: {data.npartitions}")
        logging.debug(f"Columns: {data.columns}")
        logging.debug(f"Data types: {data.dtypes}")
        logging.debug(f"Number of rows: {len(data)}")

        print("\033[92m✔ Filters applied successfully.\033[0m")
        
    except Exception as e:
        logging.error(f"Error applying custom filters: {e}")
        return None
    return data

def _fill_missing(data):
    """
    Fill missing values with NaN.

    Parameters:
    data (dask.dataframe.DataFrame): The data to be processed.

    Returns:
    dask.dataframe.DataFrame: The processed data, or None if an error occurred.
    """
    try:
        data = data.fillna(np.nan)
        print("\033[92m✔ Filles missing values with NaN successfully.\033[0m")
        return data
    except Exception as e:
        logging.error(f"Error during data processing: {e}")
        return None

def _calculate_time_differences(data):
    """
    Calculate time differences between specified columns.

    Parameters:
    data (dask.dataframe.DataFrame): The data for which time differences should be calculated.

    Returns:
    dask.dataframe.DataFrame: The data with calculated time differences, or None if an error occurred.
    """
    def calculate_time_difference(data, col1, col2, new_col_name):
        try:
            data[col1] = dd.to_datetime(data[col1], format='mixed', dayfirst=True)
            data[col2] = dd.to_datetime(data[col2], format='mixed', dayfirst=True)
            data[new_col_name] = (data[col1] - data[col2]).dt.total_seconds().astype('int64')
            logging.debug(f"Calculated {new_col_name} values: {data[new_col_name].head()}")
            print(f"\033[92m✔ {new_col_name} calculated successfully.\033[0m")
        except Exception as e:
            logging.error(f"Error calculating time difference for {new_col_name}: {e}")
            return None
        return data
    
    data = calculate_time_difference(data, 'ANKUNFTSZEIT', 'AN_PROGNOSE', 'ARRIVAL_TIME_DIFF_SECONDS')
    if data is None:
        return None
    
    data = calculate_time_difference(data, 'ABFAHRTSZEIT', 'AB_PROGNOSE', 'DEPARTURE_TIME_DIFF_SECONDS')
    if data is None:
        return None

    logging.debug(f"Total partitions: {data.npartitions}")
    logging.debug(f"Columns: {data.columns}")
    logging.debug(f"Data types: {data.dtypes}")
    logging.debug(f"Number of rows: {len(data)}")

    return data

def _encode_categorical_columns(data):
    """
    Encode all columns to int64 using Dask's parallel processing.

    Parameters:
    data (dask.dataframe.DataFrame): The data to be encoded.

    Returns:
    dask.dataframe.DataFrame: The encoded data, or None if an error occurred.
    """
    try:
        logging.info("Encoding all columns to int64.")
        
        def encode_partition(df):
            for column in df.columns:
                df[column] = df[column].astype('category').cat.codes
                df[column] = df[column].astype('int64')
            return df
        
        data = data.map_partitions(encode_partition)
        logging.info("All columns encoded to int64.")
    except Exception as e:
        logging.error(f"Error encoding columns: {e}")
        return None
    return data

def _transform_betriebstag(data):
    """
    Extract day, month, year, and day of the week from the BETRIEBSTAG column.

    Parameters:
    data (dask.dataframe.DataFrame): The data to be processed.

    Returns:
    dask.dataframe.DataFrame: The processed data, or None if an error occurred.
    """
    try:
        data['BETRIEBSTAG'] = dd.to_datetime(data['BETRIEBSTAG'], errors='coerce', dayfirst=True)
        data['DAY'] = data['BETRIEBSTAG'].dt.day
        data['MONTH'] = data['BETRIEBSTAG'].dt.month
        data['YEAR'] = data['BETRIEBSTAG'].dt.year
        data['DAY_OF_WEEK'] = data['BETRIEBSTAG'].dt.dayofweek
        logging.info("Extracted day, month, year, and day of the week from BETRIEBSTAG column.")
        data = data.drop(columns=['BETRIEBSTAG'], errors='ignore')
    except Exception as e:
        logging.error(f"Error extracting date components: {e}")
        return None
    
    print(f"\033[92m✔ BETRIEBSTAGE transformed successfully.\033[0m")

    logging.debug(f"Total partitions: {data.npartitions}")
    logging.debug(f"Columns: {data.columns}")
    logging.debug(f"Data types: {data.dtypes}")
    logging.debug(f"Number of rows: {len(data)}")
    
    return data

def _save_data(data, output_file_path):
    """
    Save the processed data to a Parquet file.

    Parameters:
    data (dask.dataframe.DataFrame): The processed data to be saved.
    output_file_path (str): The file path where the data should be saved.

    Returns:
    str: The path to the saved Parquet file, or None if an error occurred.
    """
    try:
        logging.info("Repartitioning data into 20 partitions")
        data = data.repartition(npartitions=20)
            
        output_file_path = output_file_path.replace('.csv', '.parquet')
        logging.info(f"Output file path changed to {output_file_path}")
            
        logging.info("Starting to write data to Parquet file")
        data.to_parquet(output_file_path, 
                        engine='pyarrow', 
                        compression='snappy', 
                        write_metadata_file=False)
            
        logging.info("Successfully processed and saved data to a Parquet file.")
        return output_file_path
    except Exception as e:
        logging.error(f"Error saving processed data to Parquet file: {e}")
        return None
