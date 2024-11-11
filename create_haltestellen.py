import pandas as pd
import pyarrow.parquet as pq
import glob
import os
from pathlib import Path

def create_station_mapping(parquet_folder_path, output_file='station_mapping.csv'):
    """
    Creates a CSV file containing unique stations with their coordinates.
    
    Parameters:
    parquet_folder_path (str): Path to the folder containing parquet files
    output_file (str): Name of the output CSV file
    """
    
    # Get all parquet files in the directory
    parquet_files = glob.glob(os.path.join(parquet_folder_path, '*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder_path}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Create a set to store unique station records
    unique_stations = set()
    
    # Process each parquet file
    for i, file in enumerate(parquet_files, 1):
        # Read the parquet file
        df = pq.read_table(file).to_pandas()
        
        # Create tuples of (encoded_id, lat, lon) for each unique station
        station_records = df[['HALTESTELLEN_NAME_encoded', 'STATION_LAT', 'STATION_LON']].drop_duplicates().values.tolist()
        
        # Add to our set of unique stations
        unique_stations.update(map(tuple, station_records))
        
        # Print progress
        if i % 10 == 0:
            print(f"Processed {i}/{len(parquet_files)} files")
    
    # Convert to DataFrame
    stations_df = pd.DataFrame(list(unique_stations), 
                             columns=['station_id', 'latitude', 'longitude'])
    
    # Sort by station_id
    stations_df = stations_df.sort_values('station_id')
    
    # Save to CSV
    stations_df.to_csv(output_file, index=False)
    
    print(f"\nCreated station mapping file: {output_file}")
    print(f"Total unique stations: {len(stations_df)}")
    
    return stations_df

def main():
    # Get the project root directory (assuming script is in project root)
    project_root = Path(__file__).parent
    
    # Define input and output paths
    parquet_folder = project_root / 'data' / 'processed'
    output_file = project_root / 'data' / 'station_mapping.csv'
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create the station mapping
        stations_df = create_station_mapping(
            parquet_folder_path=str(parquet_folder),
            output_file=str(output_file)
        )
        
        # Print first few rows as a preview
        print("\nFirst few stations:")
        print(stations_df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
