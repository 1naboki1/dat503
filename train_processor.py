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
import time
import json

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
    'process_workers': 5,
    'memory_per_worker': 6,  # GB
    'months_history': 8,
    'chunk_size': 100000,
    'exclude_columns': ['BETREIBER_ABK','FAHRT_BEZEICHNER','PRODUKT_ID', 'BETREIBER_NAME', 'BETREIBER_ID', 'UMLAUF_ID', 'VERKEHRSMITTEL_TEXT', 'BPUIC', 'BETRIEBSTAG'],
    'filters': {'LINIEN_TEXT': ['IC2', 'IC3', 'IC5', 'IC6', 'IC8', 'IC21', 'IC51', 'IC61', 'IC81']}
}

class DataDownloader:
    """Handles downloading and extracting of train data files with robust error handling."""
    
    def __init__(self, base_url: str, target_folder: str, max_retries: int = 3, chunk_size: int = 8192):
        self.base_url = base_url
        self.target_folder = target_folder
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.session = requests.Session()
        # Configure longer timeouts
        self.session.timeout = (30, 300)  # (connect timeout, read timeout)
        
    def _download_with_resume(self, url: str, temp_file: str) -> bool:
        """Download file with resume capability."""
        headers = {}
        temp_file_path = f"{temp_file}.partial"
        mode = 'ab'
        
        # Check if partial download exists
        if os.path.exists(temp_file_path):
            temp_size = os.path.getsize(temp_file_path)
            headers['Range'] = f'bytes={temp_size}-'
            print(f"Resuming download from byte {temp_size}")
        else:
            temp_size = 0
            mode = 'wb'
        
        try:
            # Get file size
            response = self.session.head(url, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            # Start download
            response = self.session.get(url, headers=headers, stream=True, timeout=300)
            
            if response.status_code == 416:  # Range not satisfiable
                print("Invalid range request, starting fresh download")
                temp_size = 0
                mode = 'wb'
                headers = {}
                response = self.session.get(url, stream=True, timeout=300)
            
            response.raise_for_status()
            
            # Setup progress bar
            progress = tqdm(
                total=total_size,
                initial=temp_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {os.path.basename(url)}"
            )
            
            with open(temp_file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        progress.update(size)
            
            progress.close()
            
            # Verify download
            if os.path.getsize(temp_file_path) >= total_size:
                os.rename(temp_file_path, temp_file)
                return True
            else:
                print(f"Download incomplete. Expected {total_size} bytes, got {os.path.getsize(temp_file_path)} bytes")
                return False
                
        except Exception as e:
            print(f"Download error: {str(e)}")
            return False
    
    def _extract_with_progress(self, zip_path: str) -> bool:
        """Extract ZIP file with progress tracking."""
        try:
            with ZipFile(zip_path) as zip_file:
                # Get total size for progress bar
                total_size = sum(info.file_size for info in zip_file.filelist)
                extracted_size = 0
                
                with tqdm(total=total_size, unit='iB', unit_scale=True, 
                         desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
                    for info in zip_file.filelist:
                        zip_file.extract(info, self.target_folder)
                        extracted_size += info.file_size
                        pbar.update(info.file_size)
                
                return True
        except Exception as e:
            print(f"Extraction error: {str(e)}")
            return False
    
    def download_extract(self, month: str) -> bool:
        """Download and extract data with retries."""
        file_url = f"{self.base_url}/ist-daten-{month}.zip"
        temp_zip = os.path.join(self.target_folder, f"ist-daten-{month}.zip")
        
        print(f"\nProcessing {month}...")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{self.max_retries}")
                
                # Download
                if not self._download_with_resume(file_url, temp_zip):
                    continue
                
                # Extract
                if not self._extract_with_progress(temp_zip):
                    continue
                
                # Clean up
                try:
                    os.remove(temp_zip)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_zip}: {str(e)}")
                
                return True
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    print(f"Waiting {sleep_time} seconds before retry...")
                    time.sleep(sleep_time)
        
        print(f"Failed to process {month} after {self.max_retries} attempts")
        return False
    
    def download_months(self, months: List[str], max_workers: int = 4) -> bool:
        """Download multiple months with proper resource management."""
        results = []
        failed_months = []
        
        print(f"\n=== Downloading {len(months)} months of data ===")
        print(f"Workers: {max_workers}")
        print(f"Max retries per download: {self.max_retries}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_month = {
                executor.submit(self.download_extract, month): month 
                for month in months
            }
            
            for future in concurrent.futures.as_completed(future_to_month):
                month = future_to_month[future]
                try:
                    success = future.result()
                    results.append(success)
                    if not success:
                        failed_months.append(month)
                except Exception as e:
                    print(f"Unexpected error processing {month}: {str(e)}")
                    results.append(False)
                    failed_months.append(month)
        
        # Summary
        success_count = sum(results)
        print(f"\n=== Download Summary ===")
        print(f"Successfully processed: {success_count}/{len(months)} months")
        
        if failed_months:
            print(f"Failed months: {', '.join(failed_months)}")
        
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
    
    def __init__(self, processed_folder: str, encoding_maps: Dict = None, n_workers: int = 4, memory_per_worker: int = 8):
        self.processed_folder = processed_folder
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        self.encoding_maps = encoding_maps or {}
        
        # Define model parameters
        self.model_params = {
            'base': {
                'n_estimators': 500,
                'max_depth': 20,
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
        
        # Update feature columns to include geospatial data
        self.feature_cols = None
        self.geo_cols = ['STATION_LAT', 'STATION_LON']

    def analyze_station_delays(self, df):
        """Analyze delays by station."""
        station_stats = df.groupby('HALTESTELLEN_NAME_encoded').agg({
            'ARRIVAL_TIME_DIFF_SECONDS': ['count', 'mean', 'std'],
            'DEPARTURE_TIME_DIFF_SECONDS': ['count', 'mean', 'std']
        }).compute()
        
        # Flatten column names
        station_stats.columns = [f'{col[0]}_{col[1]}'.lower() for col in station_stats.columns]
        
        # Get station names mapping
        station_mapping = {v: k for k, v in self.encoding_maps['HALTESTELLEN_NAME'].items()}
        station_stats['station_name'] = station_stats.index.map(station_mapping)
        
        return station_stats
    
    def plot_station_delays(self, df):
        """Create comprehensive visualizations for station-based delays."""
        print("\nAnalyzing station delays...")
        station_stats = self.analyze_station_delays(df)
        
        # Create figure with better size and spacing
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
        
        # Plot 1: Top 15 stations by average arrival delay (horizontal bars with error bars)
        ax1 = fig.add_subplot(gs[0, :])
        station_stats_sorted = station_stats.nlargest(15, 'arrival_time_diff_seconds_mean')
        
        # Create horizontal bar chart
        bars = ax1.barh(y=range(len(station_stats_sorted)),
                    width=station_stats_sorted['arrival_time_diff_seconds_mean'] / 60,
                    height=0.7,
                    color='#2196F3',
                    alpha=0.7)
        
        # Add error bars
        ax1.errorbar(station_stats_sorted['arrival_time_diff_seconds_mean'] / 60,
                    range(len(station_stats_sorted)),
                    xerr=station_stats_sorted['arrival_time_diff_seconds_std'] / 60,
                    fmt='none',
                    color='#1976D2',
                    alpha=0.5,
                    capsize=5)
        
        # Customize appearance
        ax1.set_yticks(range(len(station_stats_sorted)))
        ax1.set_yticklabels(station_stats_sorted['station_name'], fontsize=10)
        ax1.set_xlabel('Average Delay (minutes)', fontsize=12)
        ax1.set_title('Top 15 Stations by Average Arrival Delay', fontsize=14, pad=20)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            std = station_stats_sorted.iloc[i]['arrival_time_diff_seconds_std'] / 60
            label = f'{width:.1f}±{std:.1f} min'
            ax1.text(width, i, f' {label}',
                    va='center',
                    fontsize=9,
                    color='#1565C0')
        
        # Add grid for better readability
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.set_axisbelow(True)
        
        # Plot 2: Arrival vs Departure delays scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Create scatter plot with transparency and size based on count
        sizes = station_stats['arrival_time_diff_seconds_count'] / \
                station_stats['arrival_time_diff_seconds_count'].max() * 300
        
        scatter = ax2.scatter(station_stats['arrival_time_diff_seconds_mean'] / 60,
                            station_stats['departure_time_diff_seconds_mean'] / 60,
                            alpha=0.6,
                            s=sizes,
                            c=station_stats['arrival_time_diff_seconds_std'] / 60,
                            cmap='YlOrRd')
        
        # Add diagonal line
        max_delay = max(station_stats['arrival_time_diff_seconds_mean'].max(),
                    station_stats['departure_time_diff_seconds_mean'].max()) / 60
        ax2.plot([0, max_delay], [0, max_delay], 'k--', alpha=0.5, label='Equal Delays')
        
        # Customize appearance
        ax2.set_xlabel('Average Arrival Delay (minutes)', fontsize=10)
        ax2.set_ylabel('Average Departure Delay (minutes)', fontsize=10)
        ax2.set_title('Station Arrival vs Departure Delays', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Delay Standard Deviation (minutes)', fontsize=9)
        
        # Plot 3: Delay distribution violin plot
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Prepare data for violin plot
        violin_data = [
            station_stats['arrival_time_diff_seconds_mean'] / 60,
            station_stats['departure_time_diff_seconds_mean'] / 60
        ]
        
        # Create violin plot
        violins = ax3.violinplot(violin_data,
                                showmeans=True,
                                showmedians=True)
        
        # Customize violin plot colors
        for i, pc in enumerate(violins['bodies']):
            pc.set_facecolor(['#2196F3', '#FFA726'][i])
            pc.set_alpha(0.7)
        
        # Customize appearance
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Arrival', 'Departure'])
        ax3.set_ylabel('Average Delay (minutes)', fontsize=10)
        ax3.set_title('Distribution of Station Delays', fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add statistics annotations
        stats_text = (
            f"Arrival Delays:\n"
            f"Mean: {station_stats['arrival_time_diff_seconds_mean'].mean()/60:.1f} min\n"
            f"Median: {station_stats['arrival_time_diff_seconds_mean'].median()/60:.1f} min\n"
            f"Std: {station_stats['arrival_time_diff_seconds_mean'].std()/60:.1f} min\n\n"
            f"Departure Delays:\n"
            f"Mean: {station_stats['departure_time_diff_seconds_mean'].mean()/60:.1f} min\n"
            f"Median: {station_stats['departure_time_diff_seconds_mean'].median()/60:.1f} min\n"
            f"Std: {station_stats['departure_time_diff_seconds_mean'].std()/60:.1f} min"
        )
        
        # Add text box with statistics
        ax3.text(1.45, ax3.get_ylim()[0], stats_text,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=9,
                verticalalignment='bottom')
        
        # Add annotations for outlier stations in scatter plot
        outlier_threshold = station_stats['arrival_time_diff_seconds_mean'].quantile(0.95)
        outliers = station_stats[
            (station_stats['arrival_time_diff_seconds_mean'] > outlier_threshold) |
            (station_stats['departure_time_diff_seconds_mean'] > outlier_threshold)
        ]
        
        for _, row in outliers.iterrows():
            ax2.annotate(
                row['station_name'],
                (row['arrival_time_diff_seconds_mean'] / 60,
                row['departure_time_diff_seconds_mean'] / 60),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
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
        # Time-based features are now handled in the DataProcessor class
        return ddf

    def create_delay_categories(self, delay_seconds):
        """Create meaningful delay categories."""
        bins = [-np.inf, -300, -60, 60, 300, 600, np.inf]
        labels = ['very_early', 'early', 'on_time', 'slight_delay', 'moderate_delay', 'severe_delay']
        return pd.cut(delay_seconds, bins=bins, labels=labels)

    def calculate_advanced_statistics(self, ddf):
        """Calculate comprehensive delay statistics with time patterns."""
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
    
        # Time patterns
        for time_col in ['ANKUNFTSZEIT_MINUTES', 'ABFAHRTSZEIT_MINUTES']:
            if time_col in ddf.columns:
                # Group by minutes and calculate mean delays
                delay_col = 'ARRIVAL_TIME_DIFF_SECONDS' if 'ANKUNFT' in time_col else 'DEPARTURE_TIME_DIFF_SECONDS'
                stats[time_col] = ddf.groupby(time_col)[delay_col].mean().compute()
    
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
    
        # Plot 1: Overall distribution
        plt.subplot(2, 2, 1)
        sns.kdeplot(data=df, x=df['DEPARTURE_TIME_DIFF_SECONDS'].div(60), label='Departure', alpha=0.5)
        sns.kdeplot(data=df, x=df['ARRIVAL_TIME_DIFF_SECONDS'].div(60), label='Arrival', alpha=0.5)
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Density')
        plt.title('Distribution of Delays')
        plt.legend()
        
        return fig

    def plot_time_patterns(self, stats):
        """Create enhanced time-based pattern visualizations."""
        fig = plt.figure(figsize=(15, 10))
        
        # Setup subplots
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Plot 1: Delays by hour
        ax1 = fig.add_subplot(gs[0])
        
        time_patterns = []
        labels = []
        colors = ['#1f77b4', '#ff7f0e']  # Blue for arrival, Orange for departure
        
        for time_col in ['ANKUNFTSZEIT_MINUTES', 'ABFAHRTSZEIT_MINUTES']:
            if time_col in stats and not stats[time_col].empty:
                # Group by hour instead of minutes for smoother visualization
                hours = stats[time_col].index / 60
                delays = stats[time_col].values / 60  # Convert to minutes
                
                # Create hourly averages
                df_hourly = pd.DataFrame({
                    'hour': hours,
                    'delay': delays
                })
                df_hourly = df_hourly.groupby(df_hourly['hour'].astype(int)).mean()
                
                time_patterns.append(df_hourly)
                labels.append('Arrival' if 'ANKUNFT' in time_col else 'Departure')
        
        if time_patterns:
            for pattern, label, color in zip(time_patterns, labels, colors):
                # Plot with error bands
                ax1.plot(pattern.index, pattern['delay'],
                        label=label, color=color, linewidth=2)
                
                # Add rolling mean for trend
                rolling_mean = pattern['delay'].rolling(window=3, center=True).mean()
                ax1.plot(pattern.index, rolling_mean, 
                        '--', color=color, alpha=0.5, 
                        label=f'{label} Trend')
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Delay (minutes)')
        ax1.set_title('Average Delays Throughout the Day')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Customize x-axis
        ax1.set_xticks(range(0, 25, 2))
        ax1.set_xlim(0, 23)
        
        # Add horizontal line at y=0
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add annotations for key times
        peak_times = {
            'Morning Rush': 8,
            'Lunch Time': 12,
            'Evening Rush': 17
        }
        
        for label, hour in peak_times.items():
            ax1.axvline(x=hour, color='gray', linestyle=':', alpha=0.3)
            ax1.text(hour, ax1.get_ylim()[1], label,
                    rotation=90, ha='right', va='top')
        
        # Plot 2: Delay Distribution by Time Period
        ax2 = fig.add_subplot(gs[1])
        
        # Define time periods
        time_periods = {
            'Early Morning (4-7)': (4, 7),
            'Morning Rush (7-10)': (7, 10),
            'Midday (10-16)': (10, 16),
            'Evening Rush (16-19)': (16, 19),
            'Evening (19-23)': (19, 23),
            'Night (23-4)': (23, 4)
        }
        
        period_stats = []
        
        for pattern, label in zip(time_patterns, labels):
            period_means = []
            period_labels = []
            
            for period_name, (start, end) in time_periods.items():
                if start < end:
                    mask = (pattern.index >= start) & (pattern.index < end)
                else:  # Handle overnight period
                    mask = (pattern.index >= start) | (pattern.index < end)
                
                mean_delay = pattern.loc[mask, 'delay'].mean()
                period_means.append(mean_delay)
                period_labels.append(period_name)
            
            df_periods = pd.DataFrame({
                'Period': period_labels,
                'Delay': period_means,
                'Type': label
            })
            period_stats.append(df_periods)
        
        # Combine stats and plot
        all_periods = pd.concat(period_stats)
        sns.barplot(data=all_periods, x='Period', y='Delay', hue='Type', ax=ax2)
        
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Average Delay (minutes)')
        ax2.set_title('Average Delays by Time Period')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f', padding=3)
        
        # Find max and min delays properly
        max_idx = all_periods['Delay'].idxmax()
        min_idx = all_periods['Delay'].idxmin()
        
        highest_delay = all_periods.iloc[max_idx]
        lowest_delay = all_periods.iloc[min_idx]
        
        # Format the delay patterns text
        delay_patterns = (
            f"Highest delays: {highest_delay['Period']} "
            f"({highest_delay['Type']}: {highest_delay['Delay']:.1f} min)\n"
            f"Lowest delays: {lowest_delay['Period']} "
            f"({lowest_delay['Type']}: {lowest_delay['Delay']:.1f} min)"
        )
        
        # Add text box with properly formatted values
        plt.figtext(0.02, 0.02, delay_patterns,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8, family='monospace')
        
        # Adjust layout with specified padding
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text box
        
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

    def train_models(self, df):
        """Train delay prediction models with proper error handling."""
        try:
            # Update feature columns to include geospatial features
            if not self.feature_cols:
                self.feature_cols = [col for col in df.columns if 
                    col.endswith(('_MINUTES', '_DAY_OF_WEEK', '_MONTH', '_IS_WEEKEND', '_encoded'))
                    or col in self.geo_cols]
            
            print("\nFeatures used in model:")
            for col in sorted(self.feature_cols):
                print(f"- {col}")
            
            X = df[self.feature_cols]
            y_dep = df['DEPARTURE_TIME_DIFF_SECONDS']
            y_arr = df['ARRIVAL_TIME_DIFF_SECONDS']
        
            # Verify data
            if X.isna().any().any():
                logging.warning("NaN values found in features. Filling with appropriate values...")
                # Fill NaN values appropriately for each column type
                for col in X.columns:
                    if col in self.geo_cols:
                        # For geospatial columns, use median values
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        # For other columns, use 0
                        X[col] = X[col].fillna(0)
        
            # Split data
            X_train, X_test, y_train_dep, y_test_dep, y_train_arr, y_test_arr = train_test_split(
                X, y_dep, y_arr, test_size=0.2, random_state=42
            )
        
            models = {}
            feature_importance_data = {}
            
            for name, (y_train, y_test) in [('departure', (y_train_dep, y_test_dep)), 
                                           ('arrival', (y_train_arr, y_test_arr))]:
                print(f"\nTraining {name} model...")
                model = RandomForestRegressor(**self.model_params['base'])
            
                # Handle potential memory issues
                try:
                    model.fit(X_train, y_train)
                except MemoryError:
                    logging.warning(f"Memory error during training. Reducing estimators for {name} model...")
                    model = RandomForestRegressor(
                        **{**self.model_params['base'], 
                           'n_estimators': self.model_params['base']['n_estimators'] // 2}
                    )
                    model.fit(X_train, y_train)
            
                # Evaluate
                y_pred = model.predict(X_test)
                print(f"{name.title()} Model Performance:")
                print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
                print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))/60:.2f} minutes")
                
                # Calculate and store feature importances
                importances = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 10 most important features for {name} delays:")
                for _, row in importances.head(10).iterrows():
                    print(f"- {row['feature']}: {row['importance']:.4f}")
                
                models[name] = model
                feature_importance_data[name] = importances
            
            # Save feature importance plots
            self._save_feature_importance_plots(feature_importance_data)
            
            return models
        
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def _save_feature_importance_plots(self, feature_importance_data):
        """Save detailed feature importance visualizations."""
        try:
            # Create plots directory
            plots_dir = os.path.join(self.processed_folder, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            for model_name, importances in feature_importance_data.items():
                plt.figure(figsize=(12, 8))
                
                # Plot feature importances
                sns.barplot(data=importances.head(15), x='importance', y='feature')
                plt.title(f'Top 15 Features for {model_name.title()} Delay Prediction')
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                
                # Add value labels
                for i, v in enumerate(importances.head(15)['importance']):
                    plt.text(v, i, f'{v:.4f}', va='center')
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'feature_importance_{model_name}.png'))
                plt.close()
                
                # Save detailed feature importance data
                importances.to_csv(os.path.join(plots_dir, f'feature_importance_{model_name}.csv'))
                
        except Exception as e:
            logging.error(f"Error saving feature importance plots: {str(e)}")

    def analyze_geo_importance(self, df, models):
        """Analyze the importance of geographical features."""
        print("\n=== Geographical Feature Analysis ===")
        
        try:
            for name, model in models.items():
                print(f"\n{name.title()} Model Geographical Analysis:")
                
                # Get feature importances
                importances = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                })
                
                # Filter for geographical features
                geo_importances = importances[importances['feature'].isin(self.geo_cols)]
                
                if not geo_importances.empty:
                    print("\nGeographical Feature Importances:")
                    for _, row in geo_importances.iterrows():
                        print(f"- {row['feature']}: {row['importance']:.4f}")
                    
                    # Calculate total geographical impact
                    total_geo_importance = geo_importances['importance'].sum()
                    print(f"\nTotal geographical feature importance: {total_geo_importance:.4f}")
                    print(f"Percentage of model decisions: {total_geo_importance * 100:.2f}%")
                    
                    # Create geographical importance visualization
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=geo_importances, x='feature', y='importance')
                    plt.title(f'Geographical Feature Importance - {name.title()} Model')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot
                    plots_dir = os.path.join(self.processed_folder, 'plots')
                    plt.savefig(os.path.join(plots_dir, f'geo_importance_{name}.png'))
                    plt.close()
                
        except Exception as e:
            logging.error(f"Error in geographical analysis: {str(e)}")

    def plot_feature_importance(self, models):
        """Create feature importance visualizations."""
        fig = plt.figure(figsize=(15, 10))
        
        # Calculate average feature importance across models
        all_importances = []
        for name, model in models.items():
            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': model.feature_importances_,
                'model': name
            })
            all_importances.append(importances)
        
        combined_importances = pd.concat(all_importances)
        
        # Calculate average importance
        avg_importances = combined_importances.groupby('feature')['importance'].mean().sort_values(ascending=True)
        
        # Get top 15 features
        top_features = avg_importances.tail(15)
        
        # Create main importance plot
        ax1 = plt.subplot(2, 1, 1)
        bars = ax1.barh(range(len(top_features)), top_features.values)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index)
        ax1.set_title('Top 15 Most Important Features (Average)')
        ax1.set_xlabel('Feature Importance')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, i, f'{width:.4f}', 
                    va='center', fontsize=8)
        
        # Create comparison plot
        ax2 = plt.subplot(2, 1, 2)
        
        # Get top 10 features for comparison
        top_10_features = avg_importances.tail(10).index
        comparison_data = combined_importances[
            combined_importances['feature'].isin(top_10_features)
        ]
        
        # Create grouped bar plot
        sns.barplot(
            data=comparison_data,
            y='feature',
            x='importance',
            hue='model',
            ax=ax2
        )
        
        ax2.set_title('Top 10 Feature Importance by Model')
        ax2.set_xlabel('Feature Importance')
        ax2.set_ylabel('Feature')
        
        # Add feature categories explanation
        feature_categories = {
            '_encoded': 'Categorical features (stations, lines, etc.)',
            '_MINUTES': 'Time of day (minutes from midnight)',
            '_DAY_OF_WEEK': 'Day of the week (0=Monday to 6=Sunday)',
            '_MONTH': 'Month of the year',
            '_IS_WEEKEND': 'Weekend indicator',
            'STATION_LAT': 'Station latitude',
            'STATION_LON': 'Station longitude'
        }
        
        # Create feature category explanation text
        explanation_text = "Feature Categories:\n\n"
        for suffix, description in feature_categories.items():
            matching_features = [f for f in self.feature_cols if suffix in f]
            if matching_features or suffix in ['STATION_LAT', 'STATION_LON']:
                explanation_text += f"• {description}\n"
        
        # Add text box with feature categories
        plt.figtext(0.02, 0.02, explanation_text,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8, family='monospace')
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        
        # Add analysis summary
        summary_text = "Key Findings:\n"
        for name, model in models.items():
            # Get top 3 features for this model
            top_3 = pd.Series(
                model.feature_importances_,
                index=self.feature_cols
            ).nlargest(3)
            
            summary_text += f"\n{name.title()} model top features:\n"
            for feat, imp in top_3.items():
                summary_text += f"  • {feat}: {imp:.4f}\n"
        
        plt.figtext(0.98, 0.02, summary_text,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8, family='monospace',
                    ha='right')
        
        return fig

    def analyze_delays(self, parquet_file: str):
        """Main analysis method with proper resource cleanup."""
        try:
            self.initialize_cluster()
            print("\n=== Starting Enhanced Delay Analysis ===")
    
            # Load data
            print("Loading data...")
            ddf = dd.read_parquet(parquet_file)
    
            # Verify required columns
            required_cols = ['DEPARTURE_TIME_DIFF_SECONDS', 'ARRIVAL_TIME_DIFF_SECONDS']
            geo_cols = ['STATION_LAT', 'STATION_LON', 'STATION_GEOID']
        
            if not all(col in ddf.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Available columns: {ddf.columns.tolist()}")
        
            # Check for geospatial columns
            has_geo = all(col in ddf.columns for col in geo_cols)
            if has_geo:
                print("\nGeospatial columns found - including in analysis")
                self.geo_cols = ['STATION_LAT', 'STATION_LON']  # Not using GEOID in the model
            else:
                print("\nWarning: Geospatial columns not found - proceeding without geographical features")
                self.geo_cols = []
    
            # Convert delay columns to numeric
            print("\nConverting delay columns to numeric...")
            for col in required_cols:
                ddf[col] = dd.to_numeric(ddf[col], errors='coerce')
    
            # Calculate statistics
            print("\nCalculating statistics...")
            stats = self.calculate_advanced_statistics(ddf)
            self.print_advanced_statistics(stats)
        
            # Prepare for modeling
            print("\nPreparing for modeling...")
            self.feature_cols = [col for col in ddf.columns if 
                col.endswith(('_MINUTES', '_DAY_OF_WEEK', '_MONTH', '_IS_WEEKEND', '_encoded'))
                or col in self.geo_cols]
        
            print("\nFeatures to be used:")
            for col in sorted(self.feature_cols):
                print(f"- {col}")
        
            # Convert to pandas for modeling
            print("\nConverting to pandas DataFrame...")
            with ProgressBar():
                df = ddf.compute()
        
            # Train models
            print("\nTraining models...")
            models = self.train_models(df)
        
            # Geographical analysis if available
            if has_geo:
                print("\nPerforming geographical analysis...")
                self.analyze_geo_importance(df, models)
        
            # Create visualizations
            print("\nGenerating visualizations...")
            figures = {
                'delay_distribution': self.plot_delay_distributions(df),
                'time_patterns': self.plot_time_patterns(stats),
                'feature_importance': self.plot_feature_importance(models),
                'model_performance': self.plot_model_performance(df, models)
            }

            # Add station delay analysis
            print("\nAnalyzing station delays...")
            station_fig = self.plot_station_delays(ddf)
            figures['station_delays'] = station_fig
        
            # Create plots directory
            plots_dir = os.path.join(self.processed_folder, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
        
            # Save figures
            print("\nSaving visualizations...")
            for name, fig in figures.items():
                fig_path = os.path.join(plots_dir, f'{name}.png')
                fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Saved {name} plot to: {fig_path}")
        
            # Save station statistics
            station_stats = self.analyze_station_delays(ddf)
            stats_path = os.path.join(self.processed_folder, 'station_stats.csv')
            station_stats.to_csv(stats_path)
            print(f"Saved station statistics to: {stats_path}")

            # Save models
            print("\nSaving models...")
            for name, model in models.items():
                model_path = os.path.join(self.processed_folder, f'rf_{name}.joblib')
                joblib.dump(model, model_path)
                print(f"Saved {name} model to: {model_path}")
        
            # Save feature importance data
            print("\nSaving feature importance data...")
            for name, model in models.items():
                importances = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
                importance_path = os.path.join(self.processed_folder, f'feature_importance_{name}.csv')
                importances.to_csv(importance_path, index=False)
                print(f"Saved {name} feature importance to: {importance_path}")
        
            # Save analysis summary
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'features_used': self.feature_cols,
                'has_geo_features': has_geo,
                'model_params': self.model_params['base'],
                'performance': {}
            }
        
            for name, model in models.items():
                y_true = df[f'{name.upper()}_TIME_DIFF_SECONDS']
                y_pred = model.predict(df[self.feature_cols])
            
                summary['performance'][name] = {
                    'r2_score': float(r2_score(y_true, y_pred)),
                    'rmse_minutes': float(np.sqrt(mean_squared_error(y_true, y_pred))/60),
                    'feature_importance': {
                        feat: float(imp) 
                        for feat, imp in zip(self.feature_cols, model.feature_importances_)
                    }
                }
        
            summary_path = os.path.join(self.processed_folder, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
            print("\nAnalysis complete!")
            return models, figures
        
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
            raise
    
        finally:
            # Ensure proper cleanup
            plt.close('all')  # Close all matplotlib figures
    
            if self.cluster:
                try:
                    self.cluster.close()
                    print("\nClosed Dask cluster")
                except Exception as e:
                    logging.warning(f"Error closing Dask cluster: {str(e)}")
            
            # Clear any remaining memory
            import gc
            gc.collect()

class DataProcessor:
    """Handles data processing using Dask for distributed computing."""
    
    def __init__(self, n_workers: int = 4, memory_per_worker: int = 7):
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        self.encoding_maps = {}
        self.output_file_path = None
        self.geospatial_data = {}
        
        self.bool_columns = ['ZUSATZFAHRT_TF', 'FAELLT_AUS_TF', 'DURCHFAHRT_TF']
        self.categorical_columns = [
            'LINIEN_ID', 
            'LINIEN_TEXT',
            'HALTESTELLEN_NAME'
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

    def load_geospatial_data(self, geospatial_file: str) -> None:
        """Load geospatial data from the provided file."""
        try:
            print("\n=== Loading Geospatial Data ===")
            with open(geospatial_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # Parse tab-separated values
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        geonameid, name, ascii_name, alt_names, latitude, longitude = parts[:6]
                        
                        # Create set of all possible names (including alternatives)
                        all_names = set([name, ascii_name] + alt_names.split(','))
                        
                        # Store the data with all possible names as keys
                        for station_name in all_names:
                            self.geospatial_data[station_name] = {
                                'latitude': float(latitude),
                                'longitude': float(longitude),
                                'geonameid': int(geonameid)
                            }
            
            print(f"Loaded geospatial data for {len(self.geospatial_data)} locations")
            
        except Exception as e:
            logging.error(f"Error loading geospatial data: {str(e)}")
            raise

    def _add_geospatial_columns(self, ddf):
        """Add geospatial information to the dataframe."""
        print("\n=== Adding Geospatial Information ===")
    
        if not self.geospatial_data:
            print("Warning: No geospatial data loaded")
            return ddf
    
        try:
            # Create mappings for station names to coordinates
            station_to_lat = {}
            station_to_lon = {}
            station_to_geoid = {}
        
            # Get unique station names from the data
            unique_stations = ddf['HALTESTELLEN_NAME'].unique().compute()
        
            # Match station names with geospatial data
            matches = 0
            for station in unique_stations:
                if station in self.geospatial_data:
                    geo_info = self.geospatial_data[station]
                    station_to_lat[station] = float(geo_info['latitude'])
                    station_to_lon[station] = float(geo_info['longitude'])
                    station_to_geoid[station] = int(geo_info['geonameid'])
                    matches += 1
        
            print(f"Matched {matches} stations with geospatial data")
        
            # Add new columns using map operation with explicit dtypes and meta
            print("Adding geospatial columns...")
            ddf['STATION_LAT'] = ddf['HALTESTELLEN_NAME'].map(
                station_to_lat,
                meta=('STATION_LAT', 'float64')
            ).fillna(-999.0).astype('float64')
        
            ddf['STATION_LON'] = ddf['HALTESTELLEN_NAME'].map(
                station_to_lon,
                meta=('STATION_LON', 'float64')
            ).fillna(-999.0).astype('float64')
        
            ddf['STATION_GEOID'] = ddf['HALTESTELLEN_NAME'].map(
                station_to_geoid,
                meta=('STATION_GEOID', 'int64')
            ).fillna(-1).astype('int64')
        
            # Verify the data types
            print("\nVerifying geospatial column data types...")
            for col, dtype in {
                'STATION_LAT': 'float64',
                'STATION_LON': 'float64',
                'STATION_GEOID': 'int64'
            }.items():
                actual_dtype = ddf[col].dtype
                print(f"{col}: {actual_dtype}")
                if str(actual_dtype) != dtype:
                    print(f"Warning: {col} has dtype {actual_dtype}, expected {dtype}")
                    ddf[col] = ddf[col].astype(dtype)
        
            print("Geospatial columns added successfully")
            return ddf
        
        except Exception as e:
            logging.error(f"Error adding geospatial data: {str(e)}", exc_info=True)
            raise
    
    def initialize_cluster(self) -> None:
        """Initialize the Dask cluster for distributed processing."""
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
        """Encode categorical and boolean columns and process timestamps."""
        print("\n=== Encoding Columns ===")
        
        # Store columns to drop later
        columns_to_drop = []
        
        # Handle boolean columns
        print("\nEncoding boolean columns...")
        bool_mapping = {'false': 0, 'true': 1, 'False': 0, 'True': 1}
        for col in self.bool_columns:
            if col in ddf.columns:
                print(f"Encoding {col}...")
                ddf[f'{col}_encoded'] = ddf[col].astype(str).str.lower().map(
                    bool_mapping,
                    meta=(f'{col}_encoded', 'int8')
                )
                self.encoding_maps[col] = bool_mapping
                columns_to_drop.append(col)
        
        # Handle categorical columns
        print("\nEncoding categorical columns...")
        for col in self.categorical_columns:
            if col in ddf.columns:
                print(f"Encoding {col}...")
                unique_values = ddf[col].unique().compute()
                mapping = {str(val): idx for idx, val in enumerate(sorted(str(v) for v in unique_values))}
                self.encoding_maps[col] = mapping
                
                ddf[f'{col}_encoded'] = ddf[col].astype(str).map(
                    mapping,
                    meta=(f'{col}_encoded', 'int32')
                )
                # Don't drop HALTESTELLEN_NAME yet
                if col != 'HALTESTELLEN_NAME':
                    columns_to_drop.append(col)
        
        # Process timestamps
        print("\n=== Processing Timestamps ===")
        timestamp_pairs = [
            ('ANKUNFTSZEIT', 'AN_PROGNOSE', 'ARRIVAL_TIME_DIFF_SECONDS'),
            ('ABFAHRTSZEIT', 'AB_PROGNOSE', 'DEPARTURE_TIME_DIFF_SECONDS')
        ]
        
        for actual_col, pred_col, diff_col in timestamp_pairs:
            if actual_col in ddf.columns and pred_col in ddf.columns:
                print(f"\nProcessing {actual_col} and {pred_col}...")
                
                print(f"Converting timestamps to datetime...")
                ddf[actual_col] = dd.to_datetime(ddf[actual_col], format='mixed', dayfirst=True)
                ddf[pred_col] = dd.to_datetime(ddf[pred_col], format='mixed', dayfirst=True)
                
                print(f"Calculating {diff_col}...")
                ddf[diff_col] = (ddf[pred_col] - ddf[actual_col]).dt.total_seconds()
                
                print(f"Converting {actual_col} to minutes since midnight...")
                ddf[f'{actual_col}_MINUTES'] = (ddf[actual_col].dt.hour * 60 + 
                                              ddf[actual_col].dt.minute).astype('int16')
                
                ddf[f'{actual_col}_DAY_OF_WEEK'] = ddf[actual_col].dt.dayofweek.astype('int8')
                ddf[f'{actual_col}_MONTH'] = ddf[actual_col].dt.month.astype('int8')
                ddf[f'{actual_col}_IS_WEEKEND'] = (ddf[f'{actual_col}_DAY_OF_WEEK'] >= 5).astype('int8')
                
                columns_to_drop.extend([actual_col, pred_col])
        
        self._save_encodings()
        
        return ddf, columns_to_drop
    
    def _save_encodings(self):
        """Save encoding mappings to files."""
        try:
            # Save pickle file
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
                        for val, idx in sorted(self.encoding_maps[col].items(), key=lambda x: x[1]):
                            f.write(f"{val} -> {idx}\n")
                
                # Categorical columns
                f.write("\nCategorical Columns:\n")
                for col in self.categorical_columns:
                    if col in self.encoding_maps:
                        f.write(f"\n{col}:\n")
                        for val, idx in sorted(self.encoding_maps[col].items(), key=lambda x: x[1]):
                            f.write(f"{val} -> {idx}\n")
                
                # Time features explanation
                f.write("\nTime Features:\n")
                f.write("MINUTES: Minutes since midnight (0-1439)\n")
                f.write("  Example: 11:52 -> 712 minutes\n")
                f.write("  Example: 00:15 -> 15 minutes\n")
                f.write("  Example: 23:45 -> 1425 minutes\n")
            
            print(f"Human-readable mappings saved to: {mapping_file}")
            
        except Exception as e:
            print(f"Warning: Error saving mappings: {str(e)}")

    def process_data(self, train_folder: str, train_filters: Dict,
                    output_file_path: str, exclude_columns: List[str],
                    geospatial_file: str = None, delimiter: str = ';') -> Optional[str]:
        """Process the train data files."""
        try:
            self.output_file_path = output_file_path
            self.initialize_cluster()
        
            # Load geospatial data if provided
            if geospatial_file and os.path.exists(geospatial_file):
                self.load_geospatial_data(geospatial_file)
    
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
    
            # Read columns
            print("\nReading column names from first file...")
            with open(data_files[0], 'r', encoding='utf-8') as f:
                all_columns = f.readline().strip().split(delimiter)
            print(f"Available columns: {', '.join(all_columns)}")
    
            needed_columns = [col for col in all_columns if col not in exclude_columns]
            print(f"\nColumns to be processed: {', '.join(needed_columns)}")
    
            # Load data
            print("\n=== Loading Data ===")
            ddf = dd.read_csv(
                data_files,
                delimiter=delimiter,
                dtype=self.dtype_definitions,
                blocksize=f"{chunk_size}MB",
                assume_missing=True,
                usecols=needed_columns
            )
    
            # Apply filters
            print("\n=== Applying Filters ===")
            if train_filters:
                for column, values in train_filters.items():
                    if column in ddf.columns:
                        values = [values] if not isinstance(values, list) else values
                        print(f"Filtering {column} for values: {values}")
                        ddf = ddf[ddf[column].isin(values)]
                        ddf = ddf.persist()
    
            # Filter for REAL status
            print("\nFiltering for REAL status...")
            if "AN_PROGNOSE_STATUS" in ddf.columns and "AB_PROGNOSE_STATUS" in ddf.columns:
                ddf = ddf[
                    (ddf["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (ddf["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                ddf = ddf.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
                ddf = ddf.persist()
    
            # Process and encode - now returns columns to drop
            print("\n=== Encoding categorical columns ===")
            ddf, columns_to_drop = self.encode_categorical_columns(ddf)
        
            # Add geospatial data before dropping HALTESTELLEN_NAME
            print("\n=== Adding geospatial information ===")
            ddf = self._add_geospatial_columns(ddf)

            # Now safe to drop all columns including HALTESTELLEN_NAME
            print("\nDropping original columns...")
            if 'HALTESTELLEN_NAME' not in columns_to_drop:
                columns_to_drop.append('HALTESTELLEN_NAME')
            ddf = ddf.drop(columns=columns_to_drop)
    
            # Save results
            print("\n=== Saving Results ===")
            print("Writing to parquet file...")
    
            n_partitions = max(1, int(total_size * 2))
            print(f"Using {n_partitions} partitions for writing")
    
            ddf = ddf.repartition(npartitions=n_partitions)
    
            # Prepare metadata as strings
            meta = {
                'has_geospatial': str(bool(self.geospatial_data)),
                'processing_date': datetime.now().isoformat(),
                'original_files': str(len(data_files)),
                'filters_applied': str(train_filters)
            }

            # Convert all metadata values to strings
            meta = {k: str(v) for k, v in meta.items()}

            # Ensure all columns have correct dtypes before saving
            dtypes = {
                'ZUSATZFAHRT_TF_encoded': 'int8',
                'FAELLT_AUS_TF_encoded': 'int8',
                'DURCHFAHRT_TF_encoded': 'int8',
                'LINIEN_ID_encoded': 'int32',
                'LINIEN_TEXT_encoded': 'int32',
                'HALTESTELLEN_NAME_encoded': 'int32',
                'STATION_LAT': 'float64',
                'STATION_LON': 'float64',
                'STATION_GEOID': 'int64',
                'ARRIVAL_TIME_DIFF_SECONDS': 'float64',
                'DEPARTURE_TIME_DIFF_SECONDS': 'float64',
                'ANKUNFTSZEIT_MINUTES': 'int16',
                'ABFAHRTSZEIT_MINUTES': 'int16',
                'ANKUNFTSZEIT_DAY_OF_WEEK': 'int8',
                'ANKUNFTSZEIT_MONTH': 'int8',
                'ANKUNFTSZEIT_IS_WEEKEND': 'int8',
                'ABFAHRTSZEIT_DAY_OF_WEEK': 'int8',
                'ABFAHRTSZEIT_MONTH': 'int8',
                'ABFAHRTSZEIT_IS_WEEKEND': 'int8'
            }

            print("\nEnsuring correct data types...")
            for col, dtype in dtypes.items():
                if col in ddf.columns:
                    ddf[col] = ddf[col].astype(dtype)

            print("\nSaving to parquet...")
            with ProgressBar():
                ddf.to_parquet(
                output_file_path,
                engine='pyarrow',
                compression='snappy',
                write_metadata_file=True,
                write_index=False
            )

            # Save metadata separately
            metadata = {
                'has_geospatial': str(bool(self.geospatial_data)),
                'processing_date': datetime.now().isoformat(),
                'original_files': str(len(data_files)),
                'filters_applied': str(train_filters),
                'columns': ','.join(ddf.columns),
                'dtypes': str(ddf.dtypes.to_dict())
            }
    
            # Verify the saved file
            print("\nVerifying saved file...")
            test_df = dd.read_parquet(output_file_path)
            print("Columns in saved file:", test_df.columns.tolist())
            print("Number of partitions:", test_df.npartitions)
        
            # Print geospatial column stats
            geo_cols = ['STATION_LAT', 'STATION_LON', 'STATION_GEOID']
            if all(col in test_df.columns for col in geo_cols):
                print("\nGeospatial column statistics:")
                for col in geo_cols:
                    stats = test_df[col].describe().compute()
                    print(f"\n{col}:")
                    print(f"  Count: {stats['count']:,}")
                    print(f"  Non-missing: {(stats['count'] - test_df[col].isna().sum().compute()):,}")
                    print(f"  Mean: {stats['mean']:.6f}")
                    print(f"  Std: {stats['std']:.6f}")
                    print(f"  Min: {stats['min']:.6f}")
                    print(f"  Max: {stats['max']:.6f}")
        
            print(f"\n✔ Processing completed successfully")
            return output_file_path
        
        except Exception as e:
            logging.error(f"Processing error: {str(e)}", exc_info=True)
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
        self.geospatial_folder = os.path.join(base_path, "geospatial")
        
        # Create necessary directories
        for folder in [self.base_path, self.train_folder, 
                      self.processed_folder, self.geospatial_folder]:
            os.makedirs(folder, exist_ok=True)
        
        self.downloader = DataDownloader(CONFIG['base_url'], self.train_folder)
        self.processor = DataProcessor(
            n_workers=CONFIG['process_workers'],
            memory_per_worker=CONFIG['memory_per_worker']
        )
        self.analyzer = None
    
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
            
            # Check for geospatial data
            geospatial_file = os.path.join(self.geospatial_folder, "CH.txt")
            
            result = self.processor.process_data(
                train_folder=self.train_folder,
                train_filters=CONFIG['filters'],
                output_file_path=output_file,
                exclude_columns=CONFIG['exclude_columns'],
                geospatial_file=geospatial_file
            )
            
            if result and not skip_analysis:
                print("\nStarting delay analysis...")
                # Load encoding maps from the processed data
                encoding_maps_file = os.path.join(self.processed_folder, 'category_encodings.pkl')
                try:
                    with open(encoding_maps_file, 'rb') as f:
                        encoding_maps = pickle.load(f)
                    print("Successfully loaded encoding maps")
                except Exception as e:
                    print(f"Warning: Could not load encoding maps: {str(e)}")
                    encoding_maps = {}
                
                # Initialize analyzer with encoding maps
                self.analyzer = DelayAnalyzer(
                    self.processed_folder,
                    encoding_maps=encoding_maps,
                    n_workers=CONFIG['process_workers'],
                    memory_per_worker=CONFIG['memory_per_worker']
                )
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
                    # Load encoding maps
                    encoding_maps_file = os.path.join(manager.processed_folder, 'category_encodings.pkl')
                    try:
                        with open(encoding_maps_file, 'rb') as f:
                            encoding_maps = pickle.load(f)
                        print("Successfully loaded encoding maps")
                    except Exception as e:
                        print(f"Warning: Could not load encoding maps: {str(e)}")
                        encoding_maps = {}
                    
                    # Initialize analyzer with encoding maps
                    manager.analyzer = DelayAnalyzer(
                        manager.processed_folder,
                        encoding_maps=encoding_maps,
                        n_workers=CONFIG['process_workers'],
                        memory_per_worker=CONFIG['memory_per_worker']
                    )
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
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
