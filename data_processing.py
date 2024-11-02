import os
import pandas as pd
from tqdm import tqdm
import logging
import warnings
import psutil

# Configure logging to capture warnings
logging.captureWarnings(True)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Memory usage: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB")

def load_and_preprocess_data(train_folder, output_file='processed_data.csv', delimiter=';', chunk_size=10000, log_interval=100):
    # Load the data into a pandas DataFrame
    data_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    chunk_counter = 0
    first_chunk = True

    for file in tqdm(data_files, desc="Loading CSV files"):
        try:
            logging.info(f"Processing file: {file}")
            # Read the CSV file in chunks
            chunks = pd.read_csv(file, delimiter=delimiter, low_memory=False, chunksize=chunk_size)
            for chunk in chunks:
                # Preprocess the chunk (example: handle missing values)
                chunk.fillna(0, inplace=True)  # Example: fill missing values with 0
                
                # Append chunk to the output file
                chunk.to_csv(output_file, mode='a', header=first_chunk, index=False)
                first_chunk = False
                
                chunk_counter += 1
                if chunk_counter % log_interval == 0:
                    log_memory_usage()  # Log memory usage after processing every 'log_interval' chunks
        except pd.errors.ParserError as e:
            logging.error(f"Error reading {file}: {e}")

    logging.info("Successfully processed and saved data to a CSV file.")
    return output_file
