import os
import logging
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import numpy as np

def load_and_preprocess_data(train_folder, filters=None, output_file='processed_data.csv', delimiter=';'):
    # Create the working directory if it doesn't exist
    working_dir = os.path.join(train_folder, 'working')
    os.makedirs(working_dir, exist_ok=True)
    
    # Define the full path for the output file
    output_file_path = os.path.join(working_dir, output_file)
    
    logging.info(f"Starting to load data from {train_folder}")
    
    # Load the data into a Dask DataFrame
    data_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    logging.info(f"Found {len(data_files)} CSV files to process.")
    
    try:
        data = dd.read_csv(data_files, delimiter=delimiter, dtype={'LINIEN_ID': 'object', 'UMLAUF_ID': 'object'}, low_memory=False)
        logging.info("Data loaded into Dask DataFrame.")
    except Exception as e:
        logging.error(f"Error loading data into Dask DataFrame: {e}")
        return None
    
    # Apply filters if provided
    if filters:
        for column, value in filters.items():
            logging.info(f"Applying filter: {column} == {value}")
            data = data[data[column] == value]
    
    # Preprocess the data (example: handle missing values)
    try:
        data = data.fillna(np.nan)  # Example: fill missing values with NaN
        logging.info("Missing values filled with NaN.")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return None
    
    # Save the processed data to a CSV file
    try:
        logging.info(f"Saving processed data to {output_file_path}")
        with ProgressBar():
            data.to_csv(output_file_path, single_file=True, index=False)
        logging.info("Successfully processed and saved data to a CSV file.")
    except Exception as e:
        logging.error(f"Error saving processed data to CSV file: {e}")
        return None
    
    return output_file_path
