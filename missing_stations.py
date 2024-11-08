import dask.dataframe as dd
import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_or_create_missing_stations_csv(csv_path):
    """Load existing missing stations CSV or create a new one if it doesn't exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Update status to 'updated' for entries with valid coordinates
        mask = (df['status'] == 'pending') & df['new_latitude'].notna() & df['new_longitude'].notna()
        df.loc[mask, 'status'] = 'updated'
        df.to_csv(csv_path, index=False)
        return df
    else:
        df = pd.DataFrame(columns=[
            'encoded_id', 
            'new_latitude',
            'new_longitude',
            'date_added',
            'status'
        ])
        df.to_csv(csv_path, index=False)
        return df

def detect_missing_stations(parquet_file, missing_stations_csv):
    """Detect stations with missing coordinates (-999)."""
    try:
        logging.info("Loading parquet file...")
        ddf = dd.read_parquet(parquet_file)
        
        logging.info(f"Available columns: {list(ddf.columns)}")
        
        missing_coords = ddf[
            (ddf['STATION_LAT'] == -999.0) | 
            (ddf['STATION_LON'] == -999.0)
        ]
        
        missing_stations = missing_coords['HALTESTELLEN_NAME_encoded'].drop_duplicates().compute()
        
        tracking_df = load_or_create_missing_stations_csv(missing_stations_csv)
        new_stations_count = 0
        
        for encoded_id in missing_stations:
            if encoded_id not in tracking_df['encoded_id'].values:
                new_row = {
                    'encoded_id': encoded_id,
                    'new_latitude': None,
                    'new_longitude': None,
                    'date_added': datetime.now().strftime('%Y-%m-%d'),
                    'status': 'pending'
                }
                tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
                new_stations_count += 1
                logging.info(f"Added new station to tracking (ID: {encoded_id})")
        
        tracking_df.to_csv(missing_stations_csv, index=False)
        
        logging.info(f"\nMissing Stations Summary:")
        logging.info(f"- Total stations in tracking: {len(tracking_df)}")
        logging.info(f"- New stations found: {new_stations_count}")
        logging.info(f"- Pending coordinates: {len(tracking_df[tracking_df['status'] == 'pending'])}")
        logging.info(f"- Updated coordinates: {len(tracking_df[tracking_df['status'] == 'updated'])}")
        logging.info(f"- Invalid coordinates: {len(tracking_df[tracking_df['status'] == 'invalid'])}")
        
        return new_stations_count
        
    except Exception as e:
        logging.error(f"Error detecting missing stations: {str(e)}", exc_info=True)
        raise

def apply_coordinate_updates(parquet_file, missing_stations_csv, output_file=None):
    """Apply coordinate updates from CSV to the parquet file."""
    try:
        # Load the updates CSV
        logging.info("Loading coordinate updates...")
        updates_df = pd.read_csv(missing_stations_csv)
        
        # Filter for stations with status 'updated'
        valid_updates = updates_df[updates_df['status'] == 'updated'].copy()
        
        if len(valid_updates) == 0:
            logging.info("No valid updates found in CSV")
            return False
        
        # Create a mapping dictionary for quick lookups
        coord_updates = {}
        for _, row in valid_updates.iterrows():
            coord_updates[row['encoded_id']] = {
                'lat': row['new_latitude'],
                'lon': row['new_longitude']
            }
        
        logging.info(f"Found {len(coord_updates)} stations to update")
        
        # Load parquet file
        logging.info("Loading parquet file...")
        ddf = dd.read_parquet(parquet_file)
        
        # Create backup
        backup_file = f"{parquet_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Creating backup: {backup_file}")
        ddf.to_parquet(backup_file, write_index=False)
        
        # Update coordinates using map_partitions
        def update_partition(df):
            df = df.copy()
            for encoded_id, coords in coord_updates.items():
                mask = df['HALTESTELLEN_NAME_encoded'] == encoded_id
                df.loc[mask, 'STATION_LAT'] = coords['lat']
                df.loc[mask, 'STATION_LON'] = coords['lon']
            return df
        
        logging.info("Applying coordinate updates...")
        updated_ddf = ddf.map_partitions(update_partition)
        
        # Save updated dataset
        if output_file is None:
            output_file = parquet_file
            
        logging.info(f"Saving updated dataset to: {output_file}")
        updated_ddf.to_parquet(
            output_file,
            compression='snappy',
            write_metadata_file=True,
            write_index=False
        )
        
        logging.info(f"Successfully applied {len(valid_updates)} coordinate updates")
        return True
        
    except Exception as e:
        logging.error(f"Error applying updates: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # File paths
    parquet_file = "data/processed/processed_data.parquet"
    missing_stations_csv = "data/geospatial/missing_stations.csv"
    
    try:
        # First, check for any new missing stations
        print("\nChecking for missing stations...")
        new_stations = detect_missing_stations(parquet_file, missing_stations_csv)
        
        if new_stations > 0:
            print(f"\nFound {new_stations} new stations with missing coordinates!")
            print("Please update their coordinates in the CSV file")
        
        # Then, apply any updates from the CSV
        print("\nApplying any coordinate updates from CSV...")
        if apply_coordinate_updates(parquet_file, missing_stations_csv):
            print("Successfully applied coordinate updates!")
        else:
            print("No updates were applied (check if there are any stations with 'updated' status in CSV)")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
