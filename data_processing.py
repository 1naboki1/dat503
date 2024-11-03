import os
import logging
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd

# Set pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

def load_and_preprocess_data(train_folder, filters, output_file_path, delimiter=';'):
    logging.info(f"Starting to load data from {train_folder}")
    
    data_files = _get_csv_files(train_folder)
    logging.info(f"Found {len(data_files)} CSV files to process.")
    
    data = _load_data(data_files, delimiter)
    if data is None:
        return None
    
    return preprocess_and_save_data(data, filters, output_file_path)

def _get_csv_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def _load_data(data_files, delimiter):
    try:
        with ProgressBar():
            logging.info("Loading data into Dask DataFrame.")
            data = dd.read_csv(data_files, delimiter=delimiter, dtype={'LINIEN_ID': 'object', 'UMLAUF_ID': 'object'}, low_memory=False)
        logging.info("Data loaded into Dask DataFrame.")
        logging.info(f"Data columns: {data.columns}")
        return data
    except Exception as e:
        logging.error(f"Error loading data into Dask DataFrame: {e}")
        return None

def preprocess_and_save_data(data, filters, output_file_path):
    data = _apply_filters(data, filters)
    if data is None:
        return None
    
    data = _preprocess_data(data)
    if data is None:
        return None
    
    return _save_data(data, output_file_path)

def _apply_filters(data, filters):
    if filters:
        for column, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            logging.info(f"Applying filter: {column} in {values}")
            try:
                if column in data.columns:
                    with ProgressBar():
                        data = data[data[column].isin(values)]
                    logging.info(f"Filter applied on column: {column}")
                else:
                    logging.error(f"Column {column} does not exist in the data.")
                    return None
            except Exception as e:
                logging.error(f"Error applying filter on column {column}: {e}")
                return None
    return data

def _preprocess_data(data):
    try:
        logging.info("Filling missing values with NaN.")
        with ProgressBar():
            data = data.fillna(np.nan)
            data = data.map_partitions(lambda df: df.infer_objects(copy=False))
        logging.info("Missing values filled with NaN and inferred object types.")
        return data
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return None

def _save_data(data, output_file_path):
    try:
        logging.info(f"Saving processed data to {output_file_path}")
        with ProgressBar():
            data.to_csv(output_file_path, single_file=True, index=False)
        logging.info("Successfully processed and saved data to a CSV file.")
        return output_file_path
    except Exception as e:
        logging.error(f"Error saving processed data to CSV file: {e}")
        return None
