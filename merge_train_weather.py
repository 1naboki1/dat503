import dask.dataframe as dd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dask.distributed import Client, LocalCluster
import warnings
import json
from pathlib import Path
from dask.diagnostics import ProgressBar
import glob
warnings.filterwarnings('ignore')

class ValidationReport:
    def __init__(self):
        self.total_records = 0
        self.records_with_weather = 0
        self.records_missing_weather = 0
        self.stations_without_weather = set()
        self.time_diff_stats = {
            'mean': 0,
            'median': 0,
            'max': 0,
            'min': 0
        }
        self.monthly_coverage = {}
        self.station_coverage = {}
        
    def to_dict(self):
        # Convert numpy types to Python types for JSON serialization
        return {
            'total_records': int(self.total_records),
            'records_with_weather': int(self.records_with_weather),
            'records_missing_weather': int(self.records_missing_weather),
            'stations_without_weather': [int(x) for x in self.stations_without_weather],
            'time_diff_stats': {
                'mean': float(self.time_diff_stats['mean']),
                'median': float(self.time_diff_stats['median']),
                'max': float(self.time_diff_stats['max']),
                'min': float(self.time_diff_stats['min'])
            },
            'monthly_coverage': {
                int(k): {
                    'total': int(v['total']),
                    'with_weather': int(v['with_weather']),
                    'coverage_percentage': float(v['coverage_percentage'])
                } for k, v in self.monthly_coverage.items()
            },
            'station_coverage': {
                int(k): {
                    'total': int(v['total']),
                    'with_weather': int(v['with_weather']),
                    'coverage_percentage': float(v['coverage_percentage'])
                } for k, v in self.station_coverage.items()
            }
        }
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

def validate_weather_coverage(df):
    report = ValidationReport()
    
    # Basic counts
    report.total_records = len(df)
    report.records_with_weather = df['weather_station_found'].sum()
    report.records_missing_weather = report.total_records - report.records_with_weather
    
    # Find stations without weather data
    stations_missing = df[~df['weather_station_found']]['HALTESTELLEN_NAME_encoded'].unique()
    report.stations_without_weather = set(stations_missing.tolist())  # Convert numpy array to list
    
    # Calculate time difference statistics for valid records
    valid_records = df[df['weather_station_found']]
    if len(valid_records) > 0:
        report.time_diff_stats = {
            'mean': float(valid_records['time_diff_minutes'].mean()),  # Convert numpy types to Python types
            'median': float(valid_records['time_diff_minutes'].median()),
            'max': float(valid_records['time_diff_minutes'].max()),
            'min': float(valid_records['time_diff_minutes'].min())
        }
    
    # Calculate monthly coverage
    for month in df['ABFAHRTSZEIT_MONTH'].unique():
        month_data = df[df['ABFAHRTSZEIT_MONTH'] == month]
        report.monthly_coverage[int(month)] = {
            'total': int(len(month_data)),  # Convert numpy types to Python types
            'with_weather': int(month_data['weather_station_found'].sum()),
            'coverage_percentage': float((month_data['weather_station_found'].sum() / len(month_data) * 100))
        }
    
    # Calculate station coverage
    for station in df['HALTESTELLEN_NAME_encoded'].unique():
        station_data = df[df['HALTESTELLEN_NAME_encoded'] == station]
        report.station_coverage[int(station)] = {
            'total': int(len(station_data)),  # Convert numpy types to Python types
            'with_weather': int(station_data['weather_station_found'].sum()),
            'coverage_percentage': float((station_data['weather_station_found'].sum() / len(station_data) * 100))
        }
    
    return report

def create_datetime(row):
    try:
        # Explicitly convert to integers, handling NaN values
        month = pd.to_numeric(row['ABFAHRTSZEIT_MONTH'], downcast='integer')
        day_of_week = pd.to_numeric(row['ABFAHRTSZEIT_DAY_OF_WEEK'], downcast='integer')
        minutes = pd.to_numeric(row['ABFAHRTSZEIT_MINUTES'], downcast='integer')
        
        # Check for NaN values
        if pd.isna(month) or pd.isna(day_of_week) or pd.isna(minutes):
            return pd.NaT
            
        base_date = datetime(2024, int(month), 1)
        days_to_add = int(day_of_week) - base_date.isoweekday()
        if days_to_add < 0:
            days_to_add += 7
        date = base_date + timedelta(days=days_to_add)
        time = date + timedelta(minutes=int(minutes))
        return time
    except (ValueError, TypeError) as e:
        print(f"Error processing row: {row}")
        print(f"Error details: {str(e)}")
        return pd.NaT

def find_closest_weather(transport_row, weather_df):
    station_weather = weather_df[
        weather_df['station_id'] == transport_row['HALTESTELLEN_NAME_encoded']
    ][['valid_time', 'temperature_2m', 'dewpoint_2m', 'wind_speed', 
       'wind_direction', 'surface_pressure', 'solar_radiation',
       'total_precipitation', 'snow_cover', 'snowfall', 'soil_temperature_1']]
    
    if len(station_weather) == 0:
        return pd.Series({
            'temperature_2m': np.nan,
            'dewpoint_2m': np.nan,
            'wind_speed': np.nan,
            'wind_direction': np.nan,
            'surface_pressure': np.nan,
            'solar_radiation': np.nan,
            'total_precipitation': np.nan,
            'snow_cover': np.nan,
            'snowfall': np.nan,
            'soil_temperature_1': np.nan,
            'time_diff_minutes': np.nan,
            'weather_station_found': False
        })
    
    # Calculate time differences more efficiently
    time_diffs = abs(station_weather['valid_time'] - transport_row['datetime'])
    min_idx = time_diffs.idxmin()
    
    closest_weather = station_weather.loc[min_idx]
    time_diff_minutes = time_diffs[min_idx].total_seconds() / 60
    
    result = closest_weather.to_dict()
    result['time_diff_minutes'] = time_diff_minutes
    result['weather_station_found'] = True
    
    return pd.Series(result)

def process_partition(partition, weather_df):
    try:
        # Convert columns to numeric first
        for col in ['ABFAHRTSZEIT_MONTH', 'ABFAHRTSZEIT_MINUTES', 'ABFAHRTSZEIT_DAY_OF_WEEK']:
            partition[col] = pd.to_numeric(partition[col], errors='coerce')
        
        # Create datetime column
        partition['datetime'] = partition.apply(create_datetime, axis=1)
        
        # Drop rows where datetime creation failed
        valid_rows = ~partition['datetime'].isna()
        if valid_rows.sum() == 0:
            # If no valid rows, return empty DataFrame with correct columns
            empty_result = partition.copy()
            for col in weather_df.columns:
                empty_result[col] = pd.Series(dtype=weather_df[col].dtype)
            return empty_result
            
        partition = partition[valid_rows]
        
        # Process weather data
        weather_data = partition.apply(
            lambda row: find_closest_weather(row, weather_df), 
            axis=1
        )
        return pd.concat([partition, weather_data], axis=1)
    except Exception as e:
        print(f"Error processing partition: {str(e)}")
        raise

def find_closest_weather(transport_row, weather_df):
    station_weather = weather_df[
        weather_df['station_id'] == transport_row['HALTESTELLEN_NAME_encoded']
    ]
    
    if len(station_weather) == 0:
        return pd.Series({
            'temperature_2m': np.nan,
            'dewpoint_2m': np.nan,
            'wind_speed': np.nan,
            'wind_direction': np.nan,
            'surface_pressure': np.nan,
            'solar_radiation': np.nan,
            'total_precipitation': np.nan,
            'snow_cover': np.nan,
            'snowfall': np.nan,
            'soil_temperature_1': np.nan,
            'time_diff_minutes': np.nan,
            'weather_station_found': False
        })
    
    # Calculate time differences
    time_diffs = abs(station_weather['valid_time'] - transport_row['datetime'])
    min_idx = time_diffs.idxmin()
    
    # Select only the columns we want
    result = {
        'temperature_2m': station_weather.loc[min_idx, 'temperature_2m'],
        'dewpoint_2m': station_weather.loc[min_idx, 'dewpoint_2m'],
        'wind_speed': station_weather.loc[min_idx, 'wind_speed'],
        'wind_direction': station_weather.loc[min_idx, 'wind_direction'],
        'surface_pressure': station_weather.loc[min_idx, 'surface_pressure'],
        'solar_radiation': station_weather.loc[min_idx, 'solar_radiation'],
        'total_precipitation': station_weather.loc[min_idx, 'total_precipitation'],
        'snow_cover': station_weather.loc[min_idx, 'snow_cover'],
        'snowfall': station_weather.loc[min_idx, 'snowfall'],
        'soil_temperature_1': station_weather.loc[min_idx, 'soil_temperature_1'],
        'time_diff_minutes': time_diffs[min_idx].total_seconds() / 60,
        'weather_station_found': True
    }
    
    return pd.Series(result)

def save_dataframe_chunked(df, output_dir):
    """
    Save DataFrame to multiple parquet files with a fixed small chunk size
    """
    chunk_size = 10000  # Small fixed chunk size
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    total_rows = len(df)
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"Saving {total_rows:,} rows in {n_chunks} files...")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        
        chunk_filename = f'transport_weather_part_{i:05d}.parquet'
        chunk_path = output_dir / chunk_filename
        
        chunk.to_parquet(
            chunk_path,
            engine='pyarrow',
            compression='snappy',
            row_group_size=1000
        )
        
        file_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(f"Saved {chunk_filename}: {end_idx-start_idx:,} rows, {file_size_mb:.1f}MB")

def process_data():
    # Configure Dask cluster for 32GB machine
    memory_limit = '24GB'
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit=memory_limit,
        dashboard_address=':8787'
    )
    client = Client(cluster)
    
    try:
        print("Reading parquet files...")
        df_transport = dd.read_parquet(
            'data/processed/processed_data.parquet/*.parquet',
            engine='pyarrow',
            split_row_groups=True
        )
        
        print(f"Total number of rows to process: {len(df_transport):,}")
        
        # Calculate optimal partitions
        total_memory = float(memory_limit[:-2])
        target_partition_size = 0.5
        optimal_partitions = max(20, min(df_transport.npartitions, 
                                       int(total_memory / target_partition_size)))
        
        print(f"Repartitioning from {df_transport.npartitions} to {optimal_partitions} partitions...")
        df_transport = df_transport.repartition(npartitions=optimal_partitions)
        
        print("\nLoading weather data...")
        # Load only necessary columns from weather data
        usecols = ['station_id', 'valid_time', 'temperature_2m', 'dewpoint_2m',
                  'wind_speed', 'wind_direction', 'surface_pressure', 
                  'solar_radiation', 'total_precipitation', 'snow_cover',
                  'snowfall', 'soil_temperature_1']
        
        weather_dtypes = {
            'station_id': 'float32',
            'temperature_2m': 'float32',
            'dewpoint_2m': 'float32',
            'wind_speed': 'float32',
            'wind_direction': 'float32',
            'surface_pressure': 'float32',
            'solar_radiation': 'float32',
            'total_precipitation': 'float32',
            'snow_cover': 'float32',
            'snowfall': 'float32',
            'soil_temperature_1': 'float32'
        }
        
        df_weather = pd.read_csv(
            'data/weather/all_station_data_combined.csv',
            usecols=usecols,
            dtype=weather_dtypes,
            parse_dates=['valid_time']
        )
        
        # Create metadata for output DataFrame
        meta = df_transport.dtypes.to_dict()
        meta.update({
            'datetime': 'datetime64[ns]',
            'temperature_2m': 'float32',
            'dewpoint_2m': 'float32',
            'wind_speed': 'float32',
            'wind_direction': 'float32',
            'surface_pressure': 'float32',
            'solar_radiation': 'float32',
            'total_precipitation': 'float32',
            'snow_cover': 'float32',
            'snowfall': 'float32',
            'soil_temperature_1': 'float32',
            'time_diff_minutes': 'float32',
            'weather_station_found': 'bool'
        })
        
        print("\nProcessing data...")
        df_combined = df_transport.map_partitions(
            process_partition,
            weather_df=df_weather,
            meta=meta
        )
        
        print("\nComputing final results...")
        with ProgressBar():
            df_combined = df_combined.compute()
        
        print(f"\nProcessed {len(df_combined):,} records")
        
        print("\nSaving results...")
        output_dir = Path('data/processed/transport_weather_combined')
        if output_dir.exists():
            # Clean up any existing files
            for file in output_dir.glob('*.parquet'):
                file.unlink()
        else:
            output_dir.mkdir(parents=True)
        
        save_dataframe_chunked(df_combined, output_dir)
        
        print("Generating validation report...")
        validation_report = validate_weather_coverage(df_combined)
        validation_report.save(output_dir.parent / 'weather_coverage_report.json')
        
        print("\nValidation Summary:")
        print(f"Total records: {validation_report.total_records:,}")
        print(f"Records with weather: {validation_report.records_with_weather:,} "
              f"({validation_report.records_with_weather/validation_report.total_records*100:.2f}%)")
        print(f"Records missing weather: {validation_report.records_missing_weather:,}")
        print(f"Stations without weather: {len(validation_report.stations_without_weather)}")
        print(f"Time diff stats: mean={validation_report.time_diff_stats['mean']:.1f}min, "
              f"median={validation_report.time_diff_stats['median']:.1f}min")
        
        return df_combined, validation_report
        
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        raise
    finally:
        print("Cleaning up Dask cluster...")
        client.close()
        cluster.close()

if __name__ == "__main__":
    df_combined, validation_report = process_data()
