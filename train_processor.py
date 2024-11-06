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
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd

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
    'months_history': 5,
    'chunk_size': 100000,
    'exclude_columns': ['PRODUKT_ID', 'BETREIBER_NAME', 'BETREIBER_ID', 'UMLAUF_ID', 'VERKEHRSMITTEL_TEXT', 'HALTESTELLEN_NAME', 'BETRIEBSTAG'],
    'filters': {'LINIEN_TEXT': ['IC2', 'IC3', 'IC5', 'IC6', 'IC8', 'IC21']}
}

class DataDownloader:
    """Handles downloading and extracting of train data files."""
    
    def __init__(self, base_url: str, target_folder: str):
        self.base_url = base_url
        self.target_folder = target_folder
    
    def download_extract(self, month: str) -> bool:
        try:
            file_url = f"{self.base_url}/ist-daten-{month}.zip"
            print(f"Attempting to download: {file_url}")
            
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
            
            # Get statistics for encoded columns
            encoded_cols = [col for col in ddf.columns if col.endswith('_encoded')]
            stats['encoded_columns'] = {}
            
            for col in encoded_cols:
                try:
                    value_counts = ddf[col].value_counts().compute()
                    unique_values = len(value_counts)
                    stats['encoded_columns'][col] = {
                        'unique_values': unique_values,
                        'value_range': f"0 to {unique_values - 1}"
                    }
                except Exception as e:
                    logging.warning(f"Could not compute statistics for {col}: {str(e)}")
            
            return True, stats
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return False, {'error': str(e)}

class DelayAnalyzer:
    """Advanced train delay analysis using Rainbow Forest approach with Dask."""
    
    def __init__(self, processed_folder: str, n_workers: int = 4, memory_per_worker: int = 8):
        self.processed_folder = processed_folder
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        
        # Define model parameters
        self.model_params = {
            'base': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42,
                'warm_start': True
            },
            'tuning_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            }
        }
        
        self.feature_cols = None
    
    def initialize_cluster(self) -> None:
        """Initialize Dask cluster for distributed processing."""
        print(f"\n=== Initializing Analysis Environment ===")
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

    def preprocess_features(self, ddf):
        """Enhanced feature engineering for delay analysis."""
        print("\nEngineering additional features...")
        
        # Time-based features
        for col in ['ANKUNFTSZEIT', 'ABFAHRTSZEIT']:
            if col in ddf.columns:
                # Is weekend
                ddf[f'{col}_IS_WEEKEND'] = ddf[f'{col}_DAY_OF_WEEK'].isin([5, 6])
                
                # Time of day categories
                ddf[f'{col}_TIME_CATEGORY'] = ddf[f'{col}_HOUR'].map({
                    **dict.fromkeys(range(5), 'night'),
                    **dict.fromkeys(range(5, 10), 'morning_rush'),
                    **dict.fromkeys(range(10, 16), 'midday'),
                    **dict.fromkeys(range(16, 20), 'evening_rush'),
                    **dict.fromkeys(range(20, 24), 'evening')
                })
                
                # Season calculation
                ddf[f'{col}_SEASON'] = ddf[f'{col}_MONTH'].map({
                    **dict.fromkeys([12, 1, 2], 'winter'),
                    **dict.fromkeys([3, 4, 5], 'spring'),
                    **dict.fromkeys([6, 7, 8], 'summer'),
                    **dict.fromkeys([9, 10, 11], 'autumn')
                })

        # Interaction features
        if 'LINIEN_TEXT_encoded' in ddf.columns:
            ddf['LINE_HOUR_INTERACTION'] = ddf['LINIEN_TEXT_encoded'] * ddf['ANKUNFTSZEIT_HOUR']
        
        return ddf

    def create_delay_categories(self, delay_seconds):
        """Create meaningful delay categories."""
        bins = [-np.inf, -300, -60, 60, 300, 600, np.inf]
        labels = ['very_early', 'early', 'on_time', 'slight_delay', 'moderate_delay', 'severe_delay']
        return pd.cut(delay_seconds, bins=bins, labels=labels)

    def calculate_advanced_statistics(self, ddf):
        """Calculate comprehensive delay statistics."""
        stats = {}
        
        # Basic statistics
        for col in ['ARRIVAL_TIME_DIFF_SECONDS', 'DEPARTURE_TIME_DIFF_SECONDS']:
            stats[col] = {
                'mean': ddf[col].mean().compute(),
                'median': ddf[col].quantile(0.5).compute(),
                'std': ddf[col].std().compute(),
                'skew': ddf[col].skew().compute(),
                'kurtosis': ddf[col].kurtosis().compute(),
                'q90': ddf[col].quantile(0.9).compute(),
                'q95': ddf[col].quantile(0.95).compute()
            }
        
        # Time-based patterns
        for time_unit in ['HOUR', 'DAY_OF_WEEK', 'MONTH']:
            stats[f'delays_by_{time_unit}'] = ddf.groupby(f'ANKUNFTSZEIT_{time_unit}')[
                'ARRIVAL_TIME_DIFF_SECONDS'
            ].mean().compute()
        
        return stats

    def print_advanced_statistics(self, stats):
        """Print comprehensive statistics in a readable format."""
        print("\n=== Advanced Delay Statistics ===")
        
        for col, stat_dict in stats.items():
            if col.endswith('_DIFF_SECONDS'):
                print(f"\n{col.replace('_DIFF_SECONDS', '')} Delays:")
                print(f"Mean delay: {stat_dict['mean']/60:.2f} minutes")
                print(f"Median delay: {stat_dict['median']/60:.2f} minutes")
                print(f"Standard deviation: {stat_dict['std']/60:.2f} minutes")
                print(f"90th percentile: {stat_dict['q90']/60:.2f} minutes")
                print(f"95th percentile: {stat_dict['q95']/60:.2f} minutes")
                print(f"Skewness: {stat_dict['skew']:.2f}")
                print(f"Kurtosis: {stat_dict['kurtosis']:.2f}")

    def plot_delay_distributions(self, df):
        """Create delay distribution visualizations."""
        fig = plt.figure(figsize=(15, 10))
    
        # Ensure numeric type
        df['DEPARTURE_TIME_DIFF_SECONDS'] = pd.to_numeric(df['DEPARTURE_TIME_DIFF_SECONDS'], errors='coerce')
        df['ARRIVAL_TIME_DIFF_SECONDS'] = pd.to_numeric(df['ARRIVAL_TIME_DIFF_SECONDS'], errors='coerce')
    
        # Plot 1: Overall distribution
        plt.subplot(2, 2, 1)
        sns.kdeplot(data=df, x=df['DEPARTURE_TIME_DIFF_SECONDS'].div(60), label='Departure', alpha=0.5)
        sns.kdeplot(data=df, x=df['ARRIVAL_TIME_DIFF_SECONDS'].div(60), label='Arrival', alpha=0.5)
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Density')
        plt.title('Distribution of Delays')
        plt.legend()
    
        # Plot 2: Box plots by time category (if time categories exist)
        if 'ANKUNFTSZEIT_TIME_CATEGORY' in df.columns:
            plt.subplot(2, 2, 2)
            sns.boxplot(data=df, x='ANKUNFTSZEIT_TIME_CATEGORY', 
                       y=df['ARRIVAL_TIME_DIFF_SECONDS'].div(60))
            plt.xticks(rotation=45)
            plt.xlabel('Time Category')
            plt.ylabel('Delay (minutes)')
            plt.title('Delays by Time of Day')
    
        return fig

    def plot_time_patterns(self, stats):
        """Create time-based pattern visualizations."""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot delays by hour
        plt.subplot(2, 2, 1)
        delays_by_hour = stats['delays_by_HOUR']
        plt.plot(delays_by_hour.index, delays_by_hour.values/60)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Delay (minutes)')
        plt.title('Average Delays by Hour')
        
        # Plot delays by day of week
        plt.subplot(2, 2, 2)
        delays_by_dow = stats['delays_by_DAY_OF_WEEK']
        plt.bar(delays_by_dow.index, delays_by_dow.values/60)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Delay (minutes)')
        plt.title('Average Delays by Day of Week')
        
        return fig

    def plot_feature_importance(self, models):
        """Create feature importance visualizations."""
        fig = plt.figure(figsize=(15, 10))
        
        for i, (name, model) in enumerate(models.items(), 1):
            plt.subplot(2, 1, i)
            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=importances.head(10), x='importance', y='feature')
            plt.title(f'Top Features for {name.title()} Delay Prediction')
            
        return fig

    def plot_model_performance(self, df, models):
        """Create model performance visualizations."""
        fig = plt.figure(figsize=(15, 10))
        
        for i, (name, model) in enumerate(models.items(), 1):
            plt.subplot(1, 2, i)
            y_true = df[f'{name.upper()}_TIME_DIFF_SECONDS']
            y_pred = model.predict(df[self.feature_cols])
            
            plt.scatter(y_true/60, y_pred/60, alpha=0.5)
            plt.plot([-60, 60], [-60, 60], 'r--')
            plt.xlabel('Actual Delay (minutes)')
            plt.ylabel('Predicted Delay (minutes)')
            plt.title(f'{name.title()} Delay Predictions')
            
        return fig

    def analyze_delays(self, parquet_file: str):
        """Main analysis method."""
        try:
            self.initialize_cluster()
            print("\n=== Starting Enhanced Delay Analysis ===")
        
            # Load data
            print("Loading data...")
            ddf = dd.read_parquet(parquet_file)
        
            # Verify required columns and convert to numeric
            required_cols = ['DEPARTURE_TIME_DIFF_SECONDS', 'ARRIVAL_TIME_DIFF_SECONDS']
            if not all(col in ddf.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Available columns: {ddf.columns.tolist()}")
        
            # Convert delay columns to numeric
            print("Converting delay columns to numeric...")
            for col in required_cols:
                ddf[col] = dd.to_numeric(ddf[col], errors='coerce')
        
            # Preprocess and engineer features
            ddf = self.preprocess_features(ddf)
        
            # Calculate statistics
            print("\nCalculating statistics...")
            stats = self.calculate_advanced_statistics(ddf)
            self.print_advanced_statistics(stats)
        
            # Prepare for modeling
            print("\nPreparing for modeling...")
            self.feature_cols = [col for col in ddf.columns if 
                col.endswith(('_DAY', '_MONTH', '_YEAR', '_DAY_OF_WEEK', '_HOUR', 
                             '_MINUTE', '_encoded', '_IS_WEEKEND', '_TIME_CATEGORY', '_SEASON'))
                and col not in required_cols]
        
            # Convert to pandas for modeling
            print("Converting to pandas DataFrame...")
            with ProgressBar():
                df = ddf.compute()
        
            # Train models
            print("\nTraining models...")
            models = self.train_models(df)
        
            # Create and save visualizations
            print("\nGenerating visualizations...")
            figures = {
                'delay_distribution': self.plot_delay_distributions(df),
                'time_patterns': self.plot_time_patterns(stats),
                'feature_importance': self.plot_feature_importance(models),
                'model_performance': self.plot_model_performance(df, models)
            }
        
            # Save figures
            print("\nSaving visualizations...")
            for name, fig in figures.items():
                fig_path = os.path.join(self.processed_folder, f'{name}.png')
                fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Saved {name} plot to: {fig_path}")
        
            # Save models
            print("\nSaving models...")
            for name, model in models.items():
                model_path = os.path.join(self.processed_folder, f'rf_{name}.joblib')
                joblib.dump(model, model_path)
                print(f"Saved {name} model to: {model_path}")
        
            print("\nAnalysis complete!")
            return models, figures
        
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            raise
        finally:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()

    def train_models(self, df):
        """Train delay prediction models."""
        X = df[self.feature_cols]
        y_dep = df['DEPARTURE_TIME_DIFF_SECONDS']
        y_arr = df['ARRIVAL_TIME_DIFF_SECONDS']
    
        # Split data
        X_train, X_test, y_train_dep, y_test_dep, y_train_arr, y_test_arr = train_test_split(
        X, y_dep, y_arr, test_size=0.2, random_state=42
        )
    
        # Train models
        models = {}
        for name, (y_train, y_test) in [('departure', (y_train_dep, y_test_dep)), 
                                   ('arrival', (y_train_arr, y_test_arr))]:
            print(f"\nTraining {name} model...")
            model = RandomForestRegressor(**self.model_params['base'])
            model.fit(X_train, y_train)
        
            # Evaluate
            y_pred = model.predict(X_test)
            print(f"{name.title()} Model Performance:")
            print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))/60:.2f} minutes")
        
            models[name] = model
    
        return models

class DataProcessor:
    """Handles data processing using Dask for distributed computing."""
    
    def __init__(self, n_workers: int = 4, memory_per_worker: int = 7):
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        self.encoding_maps = {}
        
        self.bool_columns = ['ZUSATZFAHRT_TF', 'FAELLT_AUS_TF', 'DURCHFAHRT_TF']
        self.categorical_columns = [
            'FAHRT_BEZEICHNER', 
            'BETREIBER_ABK', 
            'LINIEN_ID', 
            'LINIEN_TEXT',
            'BPUIC'
        ]
        
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

    def encode_categorical_columns(self, ddf):
        """Encode both categorical and boolean columns with proper metadata."""
        print("\n=== Encoding Columns ===")
        
        # Handle boolean columns first
        print("\nEncoding boolean columns...")
        bool_mapping = {'false': 0, 'true': 1, 'False': 0, 'True': 1}
        for col in self.bool_columns:
            if col in ddf.columns:
                print(f"Encoding {col}...")
                # Create the encoded column
                ddf[f'{col}_encoded'] = ddf[col].astype(str).str.lower().map(
                    bool_mapping,
                    meta=(f'{col}_encoded', 'int8')
                )
                self.encoding_maps[col] = bool_mapping
                print(f"Encoded {col} to 0/1")
        
        # Handle categorical columns
        print("\nEncoding categorical columns...")
        for col in self.categorical_columns:
            if col in ddf.columns:
                print(f"Encoding {col}...")
                # Get unique values and create mapping
                unique_values = ddf[col].unique().compute()
                mapping = {str(val): idx for idx, val in enumerate(sorted(str(v) for v in unique_values))}
                self.encoding_maps[col] = mapping
                
                # Create the encoded column
                ddf[f'{col}_encoded'] = ddf[col].astype(str).map(
                    mapping,
                    meta=(f'{col}_encoded', 'int32')
                )
                print(f"Created {len(mapping)} unique encodings for {col}")
        
        # Save encodings
        try:
            # Save binary mapping file
            encoder_file = os.path.join(os.path.dirname(self.output_file_path), 'category_encodings.pkl')
            with open(encoder_file, 'wb') as f:
                pickle.dump(self.encoding_maps, f)
            print(f"\nEncoder mappings saved to: {encoder_file}")
            
            # Save human-readable mapping file
            mapping_file = os.path.join(os.path.dirname(self.output_file_path), 'category_mappings.txt')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                f.write("Category Encodings:\n\n")
                
                # Boolean columns
                f.write("Boolean Columns:\n")
                for col in self.bool_columns:
                    if col in self.encoding_maps:
                        f.write(f"\n{col}:\n")
                        for val in ['false', 'true']:
                            f.write(f"{val} -> {bool_mapping[val]}\n")
                
                # Categorical columns
                f.write("\nCategorical Columns:\n")
                for col in self.categorical_columns:
                    if col in self.encoding_maps:
                        f.write(f"\n{col}:\n")
                        for val, idx in sorted(self.encoding_maps[col].items(), key=lambda x: x[1]):
                            f.write(f"{val} -> {idx}\n")
            
            print(f"Human-readable mappings saved to: {mapping_file}")
            
        except Exception as e:
            print(f"Warning: Error saving mappings: {str(e)}")
        
        return ddf
    
    def process_data(self, train_folder: str, train_filters: Dict,
                    output_file_path: str, exclude_columns: List[str],
                    delimiter: str = ';') -> Optional[str]:
        try:
            self.output_file_path = output_file_path
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
        
            # First, read all columns from the first file to get complete column list
            print("\nReading column names from first file...")
            with open(data_files[0], 'r', encoding='utf-8') as f:
                all_columns = f.readline().strip().split(delimiter)
            print(f"Available columns: {', '.join(all_columns)}")
        
            # Start with all columns except those explicitly excluded
            needed_columns = [col for col in all_columns if col not in exclude_columns]
            print(f"\nColumns to be processed: {', '.join(needed_columns)}")
        
            # Load and process data
            print("\n=== Loading Data ===")
            ddf = dd.read_csv(
                data_files,
                delimiter=delimiter,
                dtype=self.dtype_definitions,
                blocksize=f"{chunk_size}MB",
                assume_missing=True,
                usecols=needed_columns
            )
        
            # Apply filters early
            print("\n=== Applying Filters ===")
            if train_filters:
                for column, values in train_filters.items():
                    if column in ddf.columns:
                        values = [values] if not isinstance(values, list) else values
                        print(f"Filtering {column} for values: {values}")
                        ddf = ddf[ddf[column].isin(values)]
                        ddf = ddf.persist()
        
            print("\nFiltering for REAL status...")
            if "AN_PROGNOSE_STATUS" in ddf.columns and "AB_PROGNOSE_STATUS" in ddf.columns:
                ddf = ddf[
                    (ddf["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (ddf["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                ddf = ddf.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
                ddf = ddf.persist()
        
            # Process timestamps and calculate differences in batches
            print("\n=== Processing Timestamps ===")
            timestamp_pairs = [
                ('ANKUNFTSZEIT', 'AN_PROGNOSE', 'ARRIVAL_TIME_DIFF_SECONDS'),
                ('ABFAHRTSZEIT', 'AB_PROGNOSE', 'DEPARTURE_TIME_DIFF_SECONDS')
            ]

            for actual_col, pred_col, diff_col in timestamp_pairs:
                if actual_col in ddf.columns and pred_col in ddf.columns:
                    print(f"\nProcessing {actual_col} and {pred_col}...")
        
                    # Convert to datetime
                    print(f"Converting {actual_col} to datetime...")
                    ddf[actual_col] = dd.to_datetime(ddf[actual_col], format='mixed', dayfirst=True)
                    print(f"Converting {pred_col} to datetime...")
                    ddf[pred_col] = dd.to_datetime(ddf[pred_col], format='mixed', dayfirst=True)
        
                    # Calculate time difference (predicted - actual)
                    print(f"Calculating {diff_col}...")
                    ddf[diff_col] = (ddf[pred_col] - ddf[actual_col]).dt.total_seconds()   
                
                    # Extract time components
                    for col in [actual_col, pred_col]:
                        print(f"Extracting components for {col}...")
                        ddf = ddf.assign(**{
                            f'{col}_DAY': ddf[col].dt.day,
                            f'{col}_MONTH': ddf[col].dt.month,
                            f'{col}_YEAR': ddf[col].dt.year,
                            f'{col}_DAY_OF_WEEK': ddf[col].dt.dayofweek,
                            f'{col}_HOUR': ddf[col].dt.hour,
                            f'{col}_MINUTE': ddf[col].dt.minute
                        })

                    # Drop original timestamp columns to free memory
                    ddf = ddf.drop(columns=[actual_col, pred_col])
                    ddf = ddf.persist()
                    print(f"Completed processing {diff_col}")
        
            # Encode categorical columns
            print("\n=== Encoding Columns ===")
            ddf = self.encode_categorical_columns(ddf)
        
            # Drop original categorical columns after encoding
            columns_to_drop = []
            for col in self.bool_columns + self.categorical_columns:
                if col in ddf.columns:
                    columns_to_drop.append(col)
            if columns_to_drop:
                print("\nDropping original categorical columns...")
                ddf = ddf.drop(columns=columns_to_drop)
                ddf = ddf.persist()
        
            # Print final column list
            final_columns = list(ddf.columns)
            print("\nFinal columns in dataset:")
            print(', '.join(final_columns))
        
            # Save results
            print("\n=== Saving Results ===")
            print("Writing to parquet file...")
        
            # Calculate number of partitions based on data size
            n_partitions = max(1, int(total_size * 2))  # 2 partitions per GB
            print(f"Using {n_partitions} partitions for writing")
        
            ddf = ddf.repartition(npartitions=n_partitions)
        
            with ProgressBar():
                ddf.to_parquet(
                    output_file_path,
                    engine='pyarrow',
                    compression='snappy',
                    write_metadata_file=True,
                    write_index=False
                )
        
            # Validate output
            success, stats = DataValidation.validate_parquet(output_file_path)
            if success:
                print("\n=== Output Statistics ===")
                print(f"Total rows: {stats['total_rows']:,}")
                print(f"File size: {stats['file_size']}")
                print(f"Memory usage: {stats['memory_usage']}")
            
                if 'encoded_columns' in stats:
                    print("\nEncoded Columns Statistics:")
                    for col, col_stats in stats['encoded_columns'].items():
                        print(f"\n{col}:")
                        for stat_name, value in col_stats.items():
                            print(f"  {stat_name}: {value}")
        
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
        self.analyzer = DelayAnalyzer(self.processed_folder)
    
    def get_months_to_download(self) -> List[str]:
        months = []
        current_date = datetime.now()
        
        for i in range(CONFIG['months_history']):
            date = current_date - timedelta(days=30*i)
            month_str = date.strftime("%Y-%m")
            months.append(month_str)
        
        return months
    
    def process_all(self, force_download: bool = False, skip_analysis: bool = False) -> Optional[str]:
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
            
            # Process data
            output_file = os.path.join(
                self.processed_folder,
                "processed_data.parquet"
            )
            
            result = self.processor.process_data(
                train_folder=self.train_folder,
                train_filters=CONFIG['filters'],
                output_file_path=output_file,
                exclude_columns=CONFIG['exclude_columns']
            )
            
            if result and not skip_analysis:
                print("\nStarting delay analysis...")
                self.analyzer.analyze_delays(result)
            
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

# Update the main function to include the new analysis option
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Data Processing System")
    parser.add_argument('--force-download', action='store_true',
                      help='Force download of data files')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up after processing')
    parser.add_argument('--skip-preprocessing', action='store_true',
                      help='Skip preprocessing and use existing parquet file')
    parser.add_argument('--skip-analysis', action='store_true',
                      help='Skip delay analysis')
    args = parser.parse_args()
    
    try:
        manager = DataManager()
        
        if args.skip_preprocessing:
            static_output_file = os.path.join(manager.processed_folder, "processed_data.parquet")
            if os.path.exists(static_output_file):
                print(f"\nUsing existing processed file: {static_output_file}")
                if not args.skip_analysis:
                    manager.analyzer.analyze_delays(static_output_file)
                return
            else:
                print("No existing processed file found. Proceeding with preprocessing...")
        
        result = manager.process_all(
            force_download=args.force_download,
            skip_analysis=args.skip_analysis
        )
        
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
