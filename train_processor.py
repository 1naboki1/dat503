"""
Train Data Processing System

A complete solution for downloading, processing, and analyzing train scheduling data.
Includes parallel downloading, memory-efficient processing, and comprehensive error handling.

Author: Roman Sernetz
Date: November 2024
"""

import os
import logging
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse
import requests
from zipfile import ZipFile
from io import BytesIO
import concurrent.futures
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG = {
    'base_url': 'https://opentransportdata.swiss/wp-content/uploads/ist-daten-archive',
    'data_path': 'data',
    'download_threads': 4,
    'process_workers': 3,
    'memory_per_worker': 8,  # GB
    'months_history': 3,
    'chunk_size': 100000,
    'exclude_columns': ['PRODUKT_ID', 'BETREIBER_NAME', 'BETREIBER_ID', 'UMLAUF_ID', 'VERKEHRSMITTEL_TEXT', 'HALTESTELLEN_NAME', 'BETRIEBSTAG'],
    'filters': {'LINIEN_TEXT': ['IC2', 'IC3', 'IC5', 'IC6', 'IC8', 'IC21']}
}

class DataValidation:
    """Handles data validation and statistics."""
    
    @staticmethod
    def validate_parquet(file_path: str) -> Tuple[bool, Dict]:
        """
        Validates the parquet file and returns statistics.
        Returns (success, stats_dict)
        """
        try:
            ddf = dd.read_parquet(file_path)
            stats = {
                'total_rows': int(ddf.shape[0].compute()),
                'columns': list(ddf.columns),
                'memory_usage': f"{ddf.memory_usage(deep=True).sum().compute() / (1024**2):.2f} MB",
                'file_size': f"{os.path.getsize(file_path) / (1024**2):.2f} MB"
            }
            
            # Add basic statistics for numeric columns
            numeric_stats = {}
            numeric_cols = [
                'ARRIVAL_TIME_DIFF_SECONDS', 
                'DEPARTURE_TIME_DIFF_SECONDS',
                'ANKUNFTSZEIT_DAY', 'ANKUNFTSZEIT_MONTH', 'ANKUNFTSZEIT_YEAR',
                'ANKUNFTSZEIT_DAY_OF_WEEK', 'ANKUNFTSZEIT_HOUR', 'ANKUNFTSZEIT_MINUTE',
                'ABFAHRTSZEIT_DAY', 'ABFAHRTSZEIT_MONTH', 'ABFAHRTSZEIT_YEAR',
                'ABFAHRTSZEIT_DAY_OF_WEEK', 'ABFAHRTSZEIT_HOUR', 'ABFAHRTSZEIT_MINUTE'
            ]
            
            for col in numeric_cols:
                if col in ddf.columns:
                    try:
                        series_stats = ddf[col].describe().compute()
                        numeric_stats[col] = {
                            'mean': float(series_stats['mean']),
                            'std': float(series_stats['std']),
                            'min': float(series_stats['min']),
                            'max': float(series_stats['max'])
                        }
                    except Exception as e:
                        logging.warning(f"Could not compute statistics for column {col}: {str(e)}")
            
            stats['numeric_stats'] = numeric_stats
            
            return True, stats
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return False, {'error': str(e)}

class DataDownloader:
    """Handles downloading and extracting of train data files."""
    
    def __init__(self, base_url: str, target_folder: str):
        self.base_url = base_url
        self.target_folder = target_folder
    
    def download_extract(self, month: str) -> bool:
        try:
            file_url = f"{self.base_url}/ist-daten-{month}.zip"
            print(f"Attempting to download: {file_url}")  # Added for debugging
            
            response = requests.get(file_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                
                t = tqdm(total=total_size, unit='iB', unit_scale=True, 
                        desc=f"Downloading {month}")
                file_content = BytesIO()
                
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    file_content.write(data)
                t.close()
                
                if total_size != 0 and t.n != total_size:
                    logging.error(f"Download incomplete for {month}")
                    return False
                
                with ZipFile(file_content) as zip_file:
                    file_list = zip_file.namelist()
                    with tqdm(total=len(file_list), 
                            desc=f"Extracting {month}") as pbar:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = [
                                executor.submit(zip_file.extract, file, self.target_folder)
                                for file in file_list
                            ]
                            for future in concurrent.futures.as_completed(futures):
                                pbar.update(1)
                
                logging.info(f"Successfully processed: {file_url}")
                return True
            else:
                logging.error(f"Failed to download: {file_url} "
                            f"(Status: {response.status_code})")
                return False
                
        except Exception as e:
            logging.error(f"Error processing {month}: {str(e)}")
            return False
    
    def download_months(self, months: List[str], max_workers: int = 4) -> bool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_extract, month) for month in months]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return all(results)

class DataProcessor:
    """Handles data processing using Dask for distributed computing."""
    
    def __init__(self, n_workers: int = 4, memory_per_worker: int = 7):
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        
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
    
    def process_data(self, train_folder: str, train_filters: Dict,
                    output_file_path: str, exclude_columns: List[str],
                    delimiter: str = ';') -> Optional[str]:
        try:
            self.initialize_cluster()
            
            # Get files
            data_files = [os.path.join(train_folder, f) 
                         for f in os.listdir(train_folder) 
                         if f.endswith('.csv')]
            
            if not data_files:
                raise Exception("No CSV files found")
            
            # Calculate processing parameters
            total_size = sum(os.path.getsize(f) for f in data_files) / (1024**3)
            chunk_size = CONFIG['chunk_size']
            
            print(f"\n=== Dataset Information ===")
            print(f"Files found: {len(data_files)}")
            print(f"Total size: {total_size:.2f} GB")
            
            # Load and process data
            print("\n=== Loading Data ===")
            ddf = dd.read_csv(
                data_files,
                delimiter=delimiter,
                dtype=self.dtype_definitions,
                blocksize=f"{chunk_size}MB",
                assume_missing=True
            )
            
            # Add row count before processing
            initial_rows = int(ddf.shape[0].compute())
            print(f"\nInitial row count: {initial_rows:,}")
            
            if exclude_columns:
                ddf = ddf.drop(columns=exclude_columns, errors='ignore')
            
            # Apply filters with progress tracking
            print("\n=== Applying Filters ===")
            if train_filters:
                for column, values in train_filters.items():
                    if column in ddf.columns:
                        values = [values] if not isinstance(values, list) else values
                        ddf = ddf[ddf[column].isin(values)]
                        # Show impact of each filter
                        filtered_rows = int(ddf.shape[0].compute())
                        print(f"Rows after filtering {column}: {filtered_rows:,} "
                              f"({filtered_rows/initial_rows*100:.1f}% remaining)")
            
            if "AN_PROGNOSE_STATUS" in ddf.columns and "AB_PROGNOSE_STATUS" in ddf.columns:
                ddf = ddf[
                    (ddf["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (ddf["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                ddf = ddf.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
            
            ddf = ddf.persist()
            
            # Process timestamps
            print("\n=== Processing Timestamps ===")
            timestamp_cols = ['ANKUNFTSZEIT', 'AN_PROGNOSE', 'ABFAHRTSZEIT', 'AB_PROGNOSE']
            
            for col in timestamp_cols:
                if col in ddf.columns:
                    print(f"Processing {col}...")
                    ddf[col] = dd.to_datetime(ddf[col], format='mixed', dayfirst=True)
                    
                    ddf = ddf.assign(**{
                        f'{col}_DAY': ddf[col].dt.day,
                        f'{col}_MONTH': ddf[col].dt.month,
                        f'{col}_YEAR': ddf[col].dt.year,
                        f'{col}_DAY_OF_WEEK': ddf[col].dt.dayofweek,
                        f'{col}_HOUR': ddf[col].dt.hour,
                        f'{col}_MINUTE': ddf[col].dt.minute
                    })
            
            ddf = ddf.drop(columns=timestamp_cols)
            ddf = ddf.persist()
            
            # Calculate metrics
            print("\n=== Calculating Metrics ===")
            if all(col in ddf.columns for col in ['ANKUNFTSZEIT', 'AN_PROGNOSE']):
                ddf['ARRIVAL_TIME_DIFF_SECONDS'] = (
                    ddf['ANKUNFTSZEIT'] - ddf['AN_PROGNOSE']
                ).dt.total_seconds()
            
            if all(col in ddf.columns for col in ['ABFAHRTSZEIT', 'AB_PROGNOSE']):
                ddf['DEPARTURE_TIME_DIFF_SECONDS'] = (
                    ddf['ABFAHRTSZEIT'] - ddf['AB_PROGNOSE']
                ).dt.total_seconds()
            
             # Save results
            print("\n=== Saving Results ===")
            output_file_path = output_file_path.replace('.csv', '.parquet')
            
            print("Writing to parquet file...")
            pbar = ProgressBar()
            pbar.register()  # This ensures the progress bar is displayed
            try:
                ddf.to_parquet(
                    output_file_path,
                    engine='pyarrow',
                    compression='snappy',
                    write_metadata_file=False,
                    write_index=False
                )
            finally:
                pbar.unregister()
            
            # Force computation of numeric columns before validation
            print("\nComputing final statistics...")
            if 'ARRIVAL_TIME_DIFF_SECONDS' in ddf.columns:
                _ = ddf['ARRIVAL_TIME_DIFF_SECONDS'].compute()
            if 'DEPARTURE_TIME_DIFF_SECONDS' in ddf.columns:
                _ = ddf['DEPARTURE_TIME_DIFF_SECONDS'].compute()
            
            # Validate the output
            success, stats = DataValidation.validate_parquet(output_file_path)
            if success:
                print("\n=== Output Statistics ===")
                print(f"Total rows: {stats['total_rows']:,}")
                print(f"File size: {stats['file_size']}")
                print(f"Memory usage: {stats['memory_usage']}")
                
                if stats['numeric_stats']:
                    print("\nNumeric column statistics:")
                    for col, col_stats in stats['numeric_stats'].items():
                        print(f"\n{col}:")
                        for stat_name, value in col_stats.items():
                            print(f"  {stat_name}: {value:,.2f}")
                else:
                    print("\nNo numeric columns found in the dataset")
            
            print(f"\n✔ Processing completed successfully")
            return output_file_path
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return None
        finally:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()

class DataManager:
    """Manages the overall data processing workflow."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.train_folder = os.path.join(base_path, "train")
        self.processed_folder = os.path.join(base_path, "processed")
        
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        
        self.downloader = DataDownloader(CONFIG['base_url'], self.train_folder)
        self.processor = DataProcessor(
            n_workers=CONFIG['process_workers'],
            memory_per_worker=CONFIG['memory_per_worker']
        )

    def get_months_to_download(self) -> List[str]:
        """Get list of months to download based on CONFIG['months_history']."""
        months = []
        current_date = datetime.now()
        
        for i in range(CONFIG['months_history']):
            date = current_date - timedelta(days=30*i)
            month_str = date.strftime("%Y-%m")
            months.append(month_str)
        
        return months
    
    def process_all(self, force_download: bool = False) -> Optional[str]:
        try:
            # Download data if needed
            if force_download or not os.listdir(self.train_folder):
                months = self.get_months_to_download()
                print(f"\nAttempting to download {len(months)} months of data: {', '.join(months)}")
                success = self.downloader.download_months(
                    months, 
                    max_workers=CONFIG['download_threads']
                )
                if not success:
                    raise Exception("Download failed")
            
            # Process data with static output filename
            output_file = os.path.join(
                self.processed_folder,
                "processed_data.parquet"  # Static filename
            )
            
            result = self.processor.process_data(
                train_folder=self.train_folder,
                train_filters=CONFIG['filters'],
                output_file_path=output_file,
                exclude_columns=CONFIG['exclude_columns']
            )
            
            if result:
                # Validate the final output
                success, stats = DataValidation.validate_parquet(result)
                if not success:
                    raise Exception(f"Validation failed: {stats.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logging.error(f"Processing pipeline error: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files and folders."""
        try:
            shutil.rmtree(self.train_folder)
            os.makedirs(self.train_folder)
        except Exception as e:
            logging.warning(f"Cleanup error: {str(e)}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Data Processing System")
    parser.add_argument('--force-download', action='store_true',
                      help='Force download of data files')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up after processing')
    parser.add_argument('--skip-preprocessing', action='store_true',
                      help='Skip preprocessing and use existing parquet file')
    args = parser.parse_args()
    
    try:
        manager = DataManager()
        
        if args.skip_preprocessing:
            # Check if processed file exists
            static_output_file = os.path.join(manager.processed_folder, "processed_data.parquet")
            if os.path.exists(static_output_file):
                print(f"\nUsing existing processed file: {static_output_file}")
                success, stats = DataValidation.validate_parquet(static_output_file)
                if success:
                    print("\n=== Output Statistics ===")
                    print(f"Total rows: {stats['total_rows']:,}")
                    print(f"File size: {stats['file_size']}")
                    print(f"Memory usage: {stats['memory_usage']}")
                    print("\nNumeric column statistics:")
                    for col, col_stats in stats['numeric_stats'].items():
                        print(f"\n{col}:")
                        for stat_name, value in col_stats.items():
                            print(f"  {stat_name}: {value:,.2f}")
                return
            else:
                print("No existing processed file found. Proceeding with preprocessing...")
        
        result = manager.process_all(force_download=args.force_download)
        
        if result:
            print(f"\nProcessing completed successfully")
            print(f"Output file: {result}")
        else:
            print("\nProcessing failed")
        
        if args.cleanup:
            manager.cleanup()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
