"""
Train Data Processing Script

This script processes large CSV datasets containing train scheduling data.
It handles data loading, filtering, timestamp processing, and metric calculations
in a memory-efficient way using Dask for distributed computing.

Key Features:
- Distributed processing using Dask
- Memory-efficient handling of large datasets
- Progress tracking for all operations
- Automatic cleanup of temporary files
- Comprehensive error handling

Requirements:
- Python 3.8+
- dask
- pandas
- numpy
- tqdm

Memory Usage:
- Designed for systems with 32GB RAM
- Uses 4 workers with ~7GB each
- Processes data in chunks to manage memory

Author: [Your Name]
Date: November 2024
"""

import os
import logging
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm
import pandas as pd
import gc
from typing import List, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    """
    A class to handle the processing of train scheduling data.
    
    This class manages the loading, processing, and saving of large CSV datasets
    containing train scheduling information. It uses Dask for distributed computing
    and handles memory efficiently through chunked processing.
    """
    
    def __init__(self, n_workers: int = 4, memory_per_worker: int = 7):
        """
        Initialize the data processor.
        
        Args:
            n_workers (int): Number of Dask workers to use
            memory_per_worker (int): Memory limit per worker in GB
        """
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        self.temp_files = []
        
        # Data type definitions for CSV columns
        self.dtype_definitions = {
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
            'DURCHFAHRT_TF': 'object'
        }
    
    def initialize_cluster(self) -> None:
        """Initialize the Dask distributed cluster."""
        try:
            print(f"\n=== Initializing Processing Environment ===")
            print(f"Workers: {self.n_workers}")
            print(f"Memory per worker: {self.memory_per_worker}GB")
            
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit=f"{self.memory_per_worker}GB",
                silence_logs=logging.WARNING
            )
            self.client = Client(self.cluster)
            print(f"Dashboard: {self.client.dashboard_link}")
            
        except Exception as e:
            logging.error(f"Failed to initialize cluster: {str(e)}")
            raise
    
    def get_files(self, folder: str) -> List[str]:
        """
        Get list of CSV files from the specified folder.
        
        Args:
            folder (str): Path to folder containing CSV files
            
        Returns:
            List[str]: List of file paths
        """
        try:
            files = [os.path.join(folder, f) for f in os.listdir(folder) 
                    if f.endswith('.csv')]
            if not files:
                raise ValueError("No CSV files found in specified folder")
            return files
        except Exception as e:
            logging.error(f"Error getting CSV files: {str(e)}")
            raise
    
    def process_file(self, file_path: str, chunk_size: int, 
                    exclude_columns: List[str], delimiter: str) -> Optional[str]:
        """
        Process a single CSV file in chunks.
        
        Args:
            file_path (str): Path to CSV file
            chunk_size (int): Size of chunks to process
            exclude_columns (List[str]): Columns to exclude
            delimiter (str): CSV delimiter
            
        Returns:
            Optional[str]: Path to temporary parquet file or None if processing failed
        """
        try:
            # Create reader for chunked processing
            df_reader = pd.read_csv(
                file_path,
                delimiter=delimiter,
                dtype=self.dtype_definitions,
                chunksize=chunk_size
            )
            
            # Process chunks
            chunks = []
            for chunk in tqdm(df_reader, 
                            desc=f"Processing {os.path.basename(file_path)}", 
                            leave=False):
                if exclude_columns:
                    chunk = chunk.drop(columns=exclude_columns, errors='ignore')
                chunks.append(chunk)
            
            # Save temporary result
            temp_file = f"temp_{len(self.temp_files)}.parquet"
            pd.concat(chunks).to_parquet(temp_file, index=False)
            self.temp_files.append(temp_file)
            
            del chunks
            gc.collect()
            
            return temp_file
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def apply_filters(self, ddf: dd.DataFrame, 
                     filters: Dict[str, Union[str, List[str]]]) -> dd.DataFrame:
        """
        Apply filters to the dataset.
        
        Args:
            ddf (dd.DataFrame): Input Dask DataFrame
            filters (Dict): Filters to apply
            
        Returns:
            dd.DataFrame: Filtered DataFrame
        """
        try:
            print("\nApplying filters...")
            
            # Apply custom filters
            if filters:
                with tqdm(filters.items(), desc="Processing filters") as pbar:
                    for column, values in pbar:
                        if column in ddf.columns:
                            values = [values] if not isinstance(values, list) else values
                            ddf = ddf[ddf[column].isin(values)]
                            pbar.set_description(f"Filtered {column}")
            
            # Apply status filters
            if "AN_PROGNOSE_STATUS" in ddf.columns and "AB_PROGNOSE_STATUS" in ddf.columns:
                ddf = ddf[
                    (ddf["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (ddf["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                ddf = ddf.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
            
            # Persist and show results
            ddf = ddf.persist()
            with ProgressBar():
                print(f"\nRows after filtering: {len(ddf.compute()):,}")
            gc.collect()
            
            return ddf
            
        except Exception as e:
            logging.error(f"Error applying filters: {str(e)}")
            raise
    
    def process_timestamps(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Process timestamp columns and extract components.
        
        Args:
            ddf (dd.DataFrame): Input DataFrame
            
        Returns:
            dd.DataFrame: Processed DataFrame
        """
        try:
            timestamp_cols = ['ANKUNFTSZEIT', 'AN_PROGNOSE', 'ABFAHRTSZEIT', 'AB_PROGNOSE']
            print("\nProcessing timestamps...")
            
            for col in tqdm(timestamp_cols, desc="Processing datetime columns"):
                if col in ddf.columns:
                    # Convert to datetime
                    ddf[col] = dd.to_datetime(ddf[col], format='mixed', dayfirst=True)
                    
                    # Extract components
                    print(f"\nExtracting components from {col}...")
                    components = {
                        f'{col}_DAY': ddf[col].dt.day,
                        f'{col}_MONTH': ddf[col].dt.month,
                        f'{col}_YEAR': ddf[col].dt.year,
                        f'{col}_DAY_OF_WEEK': ddf[col].dt.dayofweek,
                        f'{col}_HOUR': ddf[col].dt.hour,
                        f'{col}_MINUTE': ddf[col].dt.minute
                    }
                    
                    for comp_name, comp_data in tqdm(components.items(), 
                                                   desc="Creating components",
                                                   leave=False):
                        ddf[comp_name] = comp_data
            
            # Drop original columns and persist
            ddf = ddf.drop(columns=[col for col in timestamp_cols if col in ddf.columns])
            ddf = ddf.persist()
            gc.collect()
            
            return ddf
            
        except Exception as e:
            logging.error(f"Error processing timestamps: {str(e)}")
            raise
    
    def calculate_metrics(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Calculate time difference metrics.
        
        Args:
            ddf (dd.DataFrame): Input DataFrame
            
        Returns:
            dd.DataFrame: DataFrame with calculated metrics
        """
        try:
            print("\nCalculating time differences...")
            
            # Calculate arrival time differences
            if all(col in ddf.columns for col in ['ANKUNFTSZEIT', 'AN_PROGNOSE']):
                print("Processing arrival times...")
                ddf['ARRIVAL_TIME_DIFF_SECONDS'] = (
                    ddf['ANKUNFTSZEIT'] - ddf['AN_PROGNOSE']
                ).dt.total_seconds()
            
            # Calculate departure time differences
            if all(col in ddf.columns for col in ['ABFAHRTSZEIT', 'AB_PROGNOSE']):
                print("Processing departure times...")
                ddf['DEPARTURE_TIME_DIFF_SECONDS'] = (
                    ddf['ABFAHRTSZEIT'] - ddf['AB_PROGNOSE']
                ).dt.total_seconds()
            
            ddf = ddf.persist()
            gc.collect()
            
            return ddf
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary files and close Dask cluster."""
        print("\nCleaning up...")
        
        # Remove temporary files
        for file in self.temp_files:
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {file}: {str(e)}")
        
        # Close Dask cluster
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
        except Exception as e:
            logging.warning(f"Error closing Dask cluster: {str(e)}")
    
    def process_data(self, train_folder: str, train_filters: Dict,
                    output_file_path: str, exclude_columns: List[str],
                    delimiter: str = ';') -> Optional[str]:
        """
        Main processing function.
        
        Args:
            train_folder (str): Folder containing CSV files
            train_filters (Dict): Filters to apply
            output_file_path (str): Path for output file
            exclude_columns (List[str]): Columns to exclude
            delimiter (str): CSV delimiter
            
        Returns:
            Optional[str]: Path to output file or None if processing failed
        """
        try:
            # Initialize cluster
            self.initialize_cluster()
            
            # Get files
            data_files = self.get_files(train_folder)
            
            # Calculate processing parameters
            total_size = sum(os.path.getsize(f) for f in data_files) / (1024 * 1024 * 1024)
            chunk_size = int((total_size / self.n_workers) * 1024 / 2)
            
            print(f"\n=== Dataset Information ===")
            print(f"Files found: {len(data_files)}")
            print(f"Total size: {total_size:.2f} GB")
            print(f"Chunk size: {chunk_size} MB")
            
            # Process files in batches
            print("\n=== Loading Data ===")
            for batch_idx in range(0, len(data_files), self.n_workers):
                batch_files = data_files[batch_idx:batch_idx + self.n_workers]
                print(f"\nProcessing batch {(batch_idx//self.n_workers)+1}/"
                      f"{(len(data_files)+self.n_workers-1)//self.n_workers}")
                
                for file in tqdm(batch_files, desc="Loading files"):
                    self.process_file(file, chunk_size, exclude_columns, delimiter)
            
            # Combine and process data
            print("\n=== Processing Data ===")
            
            # Read parquet files
            print("\nCombining processed files...")
            ddf = dd.read_parquet(self.temp_files)
            ddf = ddf.repartition(npartitions=self.n_workers * 2)
            
            # Apply processing steps
            ddf = self.apply_filters(ddf, train_filters)
            ddf = self.process_timestamps(ddf)
            ddf = self.calculate_metrics(ddf)
            
            # Save results
            print("\n=== Saving Results ===")
            output_file_path = output_file_path.replace('.csv', '.parquet')
            
            with tqdm(total=100, desc="Saving data") as pbar:
                ddf.to_parquet(
                    output_file_path,
                    engine='pyarrow',
                    compression='snappy',
                    write_metadata_file=False,
                    write_index=False
                )
                pbar.update(100)
            
            print("\n✔ Processing completed successfully")
            return output_file_path
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return None
            
        finally:
            self.cleanup()

def load_and_preprocess_data(train_folder: str, train_filters: Dict,
                           output_file_path: str, exclude_columns: List[str],
                           delimiter: str = ';') -> Optional[str]:
    """
    Main entry point for data processing.
    
    Args:
        train_folder (str): Folder containing CSV files
        train_filters (Dict): Filters to apply
        output_file_path (str): Path for output file
        exclude_columns (List[str]): Columns to exclude
        delimiter (str): CSV delimiter
        
    Returns:
        Optional[str]: Path to output file or None if processing failed
    """
    processor = DataProcessor()
    return
