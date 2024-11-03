import os
import logging
import dask.dataframe as dd
from tqdm import tqdm

def load_and_preprocess_data(train_folder, output_file='processed_data.csv', delimiter=';', log_interval=100):
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
        ddf = dd.read_csv(data_files, delimiter=delimiter, assume_missing=True)
        logging.info("Data loaded into Dask DataFrame.")
    except Exception as e:
        logging.error(f"Error loading data into Dask DataFrame: {e}")
        return None
    
    # Preprocess the data (example: handle missing values)
    try:
        ddf = ddf.fillna(0)  # Example: fill missing values with 0
        logging.info("Missing values filled with 0.")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return None
    
    # Save the processed data to a CSV file with a progress bar
    try:
        logging.info(f"Saving processed data to {output_file_path}")
        with tqdm(total=ddf.npartitions, desc="Saving CSV") as pbar:
            ddf.to_csv(output_file_path, single_file=True, index=False, compute_kwargs={'scheduler': 'threads'})
            pbar.update(ddf.npartitions)
        logging.info("Successfully processed and saved data to a CSV file.")
    except Exception as e:
        logging.error(f"Error saving processed data to CSV file: {e}")
        return None
    
    return output_file_path
