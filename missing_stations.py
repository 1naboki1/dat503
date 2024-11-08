import dask.dataframe as dd
import pandas as pd
import os
import logging
import shutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_coordinates(updates_df):
    """Validate the new coordinates before applying updates."""
    valid_lat = (updates_df['new_latitude'].notna() & 
                updates_df['new_latitude'].astype(str).str.match(r'-?\d+\.?\d*'))
    valid_lon = (updates_df['new_longitude'].notna() & 
                updates_df['new_longitude'].astype(str).str.match(r'-?\d+\.?\d*'))
    
    updates_df.loc[valid_lat, 'new_latitude'] = pd.to_numeric(
        updates_df.loc[valid_lat, 'new_latitude'], errors='coerce'
    )
    updates_df.loc[valid_lon, 'new_longitude'] = pd.to_numeric(
        updates_df.loc[valid_lon, 'new_longitude'], errors='coerce'
    )
    
    valid_lat &= (updates_df['new_latitude'] >= 45.8) & (updates_df['new_latitude'] <= 47.9)
    valid_lon &= (updates_df['new_longitude'] >= 5.9) & (updates_df['new_longitude'] <= 10.5)
    
    valid_updates = updates_df[valid_lat & valid_lon].copy()
    invalid_updates = updates_df[~(valid_lat & valid_lon)].copy()
    
    return valid_updates, invalid_updates

def safe_remove_parquet(path):
    """Safely remove a parquet file/directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    except Exception as e:
        print(f"Warning: Could not remove {path}: {e}")

def update_coordinates(parquet_file, updates_csv, output_file=None, delete_backup=False):
    """Update station coordinates in the parquet file based on the updates CSV."""
    backup_file = None
    try:
        # Load updates
        print("\nLoading updates from CSV...")
        updates_df = pd.read_csv(updates_csv)
        
        # Validate updates
        print("\nValidating coordinate updates...")
        valid_updates, invalid_updates = validate_coordinates(updates_df)
        
        if len(valid_updates) == 0:
            raise ValueError("No valid coordinate updates found in CSV")
        
        print(f"\nFound {len(valid_updates)} valid updates")
        if len(invalid_updates) > 0:
            print(f"Warning: {len(invalid_updates)} invalid updates were skipped:")
            for _, row in invalid_updates.iterrows():
                print(f"- {row['station_name']} (ID: {row['encoded_id']}): "
                      f"lat={row['new_latitude']}, lon={row['new_longitude']}")
        
        # Create updates dictionary
        coord_updates = {}
        for _, row in valid_updates.iterrows():
            coord_updates[row['encoded_id']] = {
                'lat': float(row['new_latitude']),
                'lon': float(row['new_longitude'])
            }
        
        # Load parquet file
        print("\nLoading parquet file...")
        ddf = dd.read_parquet(parquet_file)
        
        # Create backup
        backup_file = f"{parquet_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\nCreating backup: {backup_file}")
        ddf.to_parquet(backup_file, write_index=False)
        
        # Update coordinates using map_partitions
        def update_partition(df):
            df = df.copy()
            for encoded_id, coords in coord_updates.items():
                mask = df['HALTESTELLEN_NAME_encoded'] == encoded_id
                df.loc[mask, 'STATION_LAT'] = coords['lat']
                df.loc[mask, 'STATION_LON'] = coords['lon']
            return df
        
        print("\nApplying coordinate updates...")
        updated_ddf = ddf.map_partitions(update_partition)
        
        # Save updated dataset
        if output_file is None:
            output_file = parquet_file
            
        print(f"\nSaving updated dataset to: {output_file}")
        updated_ddf.to_parquet(
            output_file,
            compression='snappy',
            write_metadata_file=True,
            write_index=False
        )
        
        # Verify updates
        print("\nVerifying updates...")
        verification_ddf = dd.read_parquet(output_file)
        
        # Check for remaining -999 values
        missing_coords = verification_ddf[
            (verification_ddf['STATION_LAT'] == -999.0) | 
            (verification_ddf['STATION_LON'] == -999.0)
        ]
        
        remaining_missing = missing_coords['HALTESTELLEN_NAME_encoded'].nunique().compute()
        
        print("\nUpdate Summary:")
        print(f"- Successfully applied {len(valid_updates)} coordinate updates")
        print(f"- Skipped {len(invalid_updates)} invalid updates")
        print(f"- Remaining stations with missing coordinates: {remaining_missing}")
        
        if remaining_missing > 0:
            print("\nNote: Some stations still have missing coordinates. "
                  "You may want to run the missing stations script again.")
        
        # Clean up backup if requested
        if delete_backup and backup_file and os.path.exists(backup_file):
            print(f"\nRemoving backup file: {backup_file}")
            safe_remove_parquet(backup_file)
            print("Backup file deleted")
        else:
            print(f"\nBackup file retained at: {backup_file}")
        
        return output_file
        
    except Exception as e:
        if backup_file:
            print(f"\nError occurred. Backup file preserved at: {backup_file}")
        logging.error(f"Error updating coordinates: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # File paths
    parquet_file = "data/processed/processed_data.parquet"
    updates_csv = "data/geospatial/missing_stations.csv"
    
    try:
        print("Starting coordinate update process...")
        updated_file = update_coordinates(parquet_file, updates_csv, delete_backup=True)
        print("\nCoordinate update process completed successfully!")
        
    except Exception as e:
        print(f"\nError during update process: {str(e)}")
        raise
