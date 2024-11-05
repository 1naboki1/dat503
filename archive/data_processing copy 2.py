import os
import logging
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
import gc

def _get_csv_files(folder):
    """Get list of CSV files from folder"""
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        print(f"Found {len(files)} CSV files in {folder}")
        return files
    except Exception as e:
        logging.error(f"Error getting CSV files from {folder}: {str(e)}")
        return []

def load_and_preprocess_data(train_folder, train_filters, output_file_path, exclude_columns, delimiter=';'):
    """Main processing function with optimized resource usage"""
    
    # Set up a local cluster with optimized resources
    n_workers = max(1, cpu_count() - 1)  # Leave one CPU core free
    memory_limit = '25GB'  # Use most of available memory on 32GB system
    
    print(f"\nInitializing cluster with {n_workers} workers and {memory_limit} memory limit per worker")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit=memory_limit
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    
    try:
        # Get files
        data_files = _get_csv_files(train_folder)
        if not data_files:
            raise Exception("No CSV files found")
        print(f"\nFound {len(data_files)} CSV files to process")
        
        # Load data
        print("\nStarting data load...")
        data = _load_data_parallel(data_files, delimiter, exclude_columns, n_workers)
        if data is None:
            raise Exception("Data loading failed")
        print(f"\n✔ {len(data_files)} files loaded successfully")
        
        # Process pipeline
        print("\nStarting data processing pipeline...")
        try:
            # Process in steps with monitoring
            steps = [
                ('Applying filters', lambda df: _apply_filters(df, train_filters)),
                ('Processing timestamps', _process_timestamps),
                ('Calculating metrics', _calculate_metrics),
            ]
            
            for step_name, step_func in steps:
                print(f"\nExecuting: {step_name}")
                with ProgressBar():
                    data = step_func(data)
                    if data is None:
                        raise Exception(f"{step_name} failed")
                    # Persist with optimized partitions
                    data = data.repartition(npartitions=n_workers * 4)
                    data = data.persist()
                print(f"✔ {step_name} completed")
                
                # Monitor progress
                with ProgressBar():
                    mem_usage = data.memory_usage(deep=True).sum().compute() / 1e9
                    print(f"Current memory usage: {mem_usage:.2f} GB")
                    print(f"Current partition count: {data.npartitions}")
                    n_rows = len(data.compute())
                    print(f"Current row count: {n_rows:,}")
            
            # Save results
            result = _save_optimized(data, output_file_path, n_workers)
            
            # Cleanup
            client.close()
            cluster.close()
            
            return result
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            client.close()
            cluster.close()
            return None
            
    except Exception as e:
        logging.error(f"Error in processing pipeline: {str(e)}")
        try:
            client.close()
            cluster.close()
        except:
            pass
        return None

def _load_data_parallel(data_files, delimiter, exclude_columns, n_workers):
    """Simplified data loading approach"""
    try:
        # Calculate total size and chunk size
        total_size = sum(os.path.getsize(f) for f in data_files) / (1024 * 1024 * 1024)  # GB
        chunk_size = int((total_size / n_workers) * 1024)  # MB
        
        print(f"Total data size: {total_size:.2f} GB")
        print(f"Chunk size per worker: {chunk_size} MB")
        
        # Set up dtype dictionary
        dtype = {
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
        
        # Direct loading approach
        print("\nLoading CSV files...")
        ddf = dd.read_csv(
            data_files,
            delimiter=delimiter,
            dtype=dtype,
            blocksize=f"{chunk_size}MB",
            assume_missing=True
        )
        
        # Handle excluded columns
        if exclude_columns:
            ddf = ddf.drop(columns=exclude_columns, errors='ignore')
        
        # Optimize partitions
        n_partitions = max(n_workers * 4, int(total_size * 2))
        print(f"\nRepartitioning to {n_partitions} partitions...")
        
        with ProgressBar():
            ddf = ddf.repartition(npartitions=n_partitions)
            ddf = ddf.persist()
            
            # Get statistics
            stats = {
                'rows': len(ddf.compute()),
                'columns': len(ddf.columns),
                'partitions': ddf.npartitions,
                'memory': ddf.memory_usage(deep=True).sum().compute() / 1e9
            }
            
            print(f"\nDataFrame Statistics:")
            print(f"- Rows: {stats['rows']:,}")
            print(f"- Columns: {stats['columns']}")
            print(f"- Partitions: {stats['partitions']}")
            print(f"- Memory Usage: {stats['memory']:.2f} GB")
            
            return ddf
            
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None

def load_and_preprocess_data(train_folder, train_filters, output_file_path, exclude_columns, delimiter=';'):
    """Main processing function with better memory distribution"""
    
    # Use fewer workers but give them more memory
    n_workers = 4  # Reduced from 23
    memory_per_worker = int(28 / n_workers)  # Distribute 28GB among workers
    
    print(f"\nInitializing cluster with {n_workers} workers")
    print(f"Memory limit per worker: {memory_per_worker}GB")
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit=f"{memory_per_worker}GB",
        silence_logs=logging.WARNING  # Reduce log noise
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    
    try:
        # Get CSV files
        data_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) 
                     if f.endswith('.csv')]
        
        if not data_files:
            raise Exception("No CSV files found")
        
        # Calculate chunk size based on available memory
        total_size = sum(os.path.getsize(f) for f in data_files) / (1024 * 1024 * 1024)  # GB
        chunk_size = int((total_size / n_workers) * 1024 / 2)  # Half the data per worker in MB
        
        print(f"\nTotal data size: {total_size:.2f} GB")
        print(f"Chunk size: {chunk_size} MB")
        
        # Load data in chunks
        print("\nLoading CSV files...")
        ddf = dd.read_csv(
            data_files,
            delimiter=delimiter,
            dtype={
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
            },
            blocksize=f"{chunk_size}MB",
            assume_missing=True,
            sample=10000  # Use smaller sample for partition estimation
        )
        
        if exclude_columns:
            ddf = ddf.drop(columns=exclude_columns, errors='ignore')
        
        # Use fewer, larger partitions
        n_partitions = n_workers * 4  # 4 partitions per worker
        print(f"\nRepartitioning to {n_partitions} partitions...")
        
        with ProgressBar():
            ddf = ddf.repartition(npartitions=n_partitions)
            
            # Process in steps with explicit memory management
            
            # 1. Apply filters
            print("\nApplying filters...")
            if train_filters:
                for column, values in train_filters.items():
                    if column in ddf.columns:
                        values = [values] if not isinstance(values, list) else values
                        ddf = ddf[ddf[column].isin(values)]
            
            if "AN_PROGNOSE_STATUS" in ddf.columns and "AB_PROGNOSE_STATUS" in ddf.columns:
                ddf = ddf[
                    (ddf["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (ddf["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                ddf = ddf.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
            
            ddf = ddf.persist()
            gc.collect()
            
            # 2. Process timestamps
            print("\nProcessing timestamps...")
            timestamp_cols = ['ANKUNFTSZEIT', 'AN_PROGNOSE', 'ABFAHRTSZEIT', 'AB_PROGNOSE']
            
            for col in timestamp_cols:
                if col in ddf.columns:
                    print(f"Processing {col}...")
                    ddf[col] = dd.to_datetime(ddf[col], format='mixed', dayfirst=True)
                    
                    # Extract components
                    ddf = ddf.assign(
                        **{
                            f'{col}_DAY': ddf[col].dt.day,
                            f'{col}_MONTH': ddf[col].dt.month,
                            f'{col}_YEAR': ddf[col].dt.year,
                            f'{col}_DAY_OF_WEEK': ddf[col].dt.dayofweek,
                            f'{col}_HOUR': ddf[col].dt.hour,
                            f'{col}_MINUTE': ddf[col].dt.minute
                        }
                    )
            
            ddf = ddf.drop(columns=timestamp_cols)
            ddf = ddf.persist()
            gc.collect()
            
            # 3. Calculate time differences
            print("\nCalculating time differences...")
            if all(col in ddf.columns for col in ['ANKUNFTSZEIT', 'AN_PROGNOSE']):
                ddf['ARRIVAL_TIME_DIFF_SECONDS'] = (
                    ddf['ANKUNFTSZEIT'] - ddf['AN_PROGNOSE']
                ).dt.total_seconds()
            
            if all(col in ddf.columns for col in ['ABFAHRTSZEIT', 'AB_PROGNOSE']):
                ddf['DEPARTURE_TIME_DIFF_SECONDS'] = (
                    ddf['ABFAHRTSZEIT'] - ddf['AB_PROGNOSE']
                ).dt.total_seconds()
            
            ddf = ddf.persist()
            gc.collect()
            
            # Save results
            print("\nSaving results...")
            output_file_path = output_file_path.replace('.csv', '.parquet')
            
            ddf.to_parquet(
                output_file_path,
                engine='pyarrow',
                compression='snappy',
                write_metadata_file=False,
                write_index=False
            )
            
            print("\n✔ Processing completed successfully")
            return output_file_path
            
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return None
    finally:
        # Cleanup
        try:
            client.close()
            cluster.close()
        except:
            pass

def _process_timestamps(data):
    """Process timestamps in parallel"""
    try:
        print("\nProcessing timestamp columns...")
        timestamp_cols = ['ANKUNFTSZEIT', 'AN_PROGNOSE', 'ABFAHRTSZEIT', 'AB_PROGNOSE']
        
        for col in timestamp_cols:
            if col in data.columns:
                print(f"Converting {col}...")
                with ProgressBar():
                    data[col] = dd.to_datetime(data[col], format='mixed', dayfirst=True)
                    
                    print(f"Extracting components from {col}...")
                    new_cols = {
                        f'{col}_DAY': data[col].dt.day,
                        f'{col}_MONTH': data[col].dt.month,
                        f'{col}_YEAR': data[col].dt.year,
                        f'{col}_DAY_OF_WEEK': data[col].dt.dayofweek,
                        f'{col}_HOUR': data[col].dt.hour,
                        f'{col}_MINUTE': data[col].dt.minute
                    }
                    data = data.assign(**new_cols)
                    data = data.persist()
        
        print("Dropping original timestamp columns...")
        data = data.drop(columns=[col for col in timestamp_cols if col in data.columns])
        return data
        
    except Exception as e:
        logging.error(f"Error processing timestamps: {str(e)}")
        return None

def _calculate_metrics(data):
    """Calculate metrics in parallel"""
    try:
        print("\nCalculating time differences...")
        
        if all(col in data.columns for col in ['ANKUNFTSZEIT', 'AN_PROGNOSE']):
            print("Calculating arrival time differences...")
            with ProgressBar():
                data['ARRIVAL_TIME_DIFF_SECONDS'] = (
                    data['ANKUNFTSZEIT'] - data['AN_PROGNOSE']
                ).dt.total_seconds()
                data = data.persist()
        
        if all(col in data.columns for col in ['ABFAHRTSZEIT', 'AB_PROGNOSE']):
            print("Calculating departure time differences...")
            with ProgressBar():
                data['DEPARTURE_TIME_DIFF_SECONDS'] = (
                    data['ABFAHRTSZEIT'] - data['AB_PROGNOSE']
                ).dt.total_seconds()
                data = data.persist()
        
        return data
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None

def _save_optimized(data, output_file_path, n_workers):
    """Optimized parallel saving"""
    try:
        print("\nPreparing to save data...")
        output_file_path = output_file_path.replace('.csv', '.parquet')
        
        with ProgressBar():
            # Get current size
            size_gb = data.memory_usage(deep=True).sum().compute() / 1e9
            print(f"Final data size: {size_gb:.2f} GB")
            
            # Calculate optimal partitions
            partition_size = 0.5  # GB
            n_partitions = max(n_workers * 2, int(size_gb / partition_size))
            print(f"Repartitioning to {n_partitions} partitions...")
            data = data.repartition(npartitions=n_partitions)
            
            print(f"Saving to {output_file_path}...")
            data.to_parquet(
                output_file_path,
                engine='pyarrow',
                compression='snappy',
                write_metadata_file=False,
                write_index=False
            )
        
        print("\n✔ Data saved successfully")
        return output_file_path
        
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        return None

def _apply_filters(data, train_filters):
    """Apply filters in parallel"""
    try:
        if train_filters:
            print("\nApplying custom filters...")
            for column, values in train_filters.items():
                if column in data.columns:
                    print(f"Filtering {column}...")
                    values = [values] if not isinstance(values, list) else values
                    with ProgressBar():
                        data = data[data[column].isin(values)]
                        data = data.persist()
        
        print("\nApplying status filters...")
        if "AN_PROGNOSE_STATUS" in data.columns and "AB_PROGNOSE_STATUS" in data.columns:
            with ProgressBar():
                data = data[
                    (data["AN_PROGNOSE_STATUS"] == "REAL") & 
                    (data["AB_PROGNOSE_STATUS"] == "REAL")
                ]
                data = data.persist()
            
            print("Dropping status columns...")
            data = data.drop(columns=["AN_PROGNOSE_STATUS", "AB_PROGNOSE_STATUS"])
        
        return data
        
    except Exception as e:
        logging.error(f"Error applying filters: {str(e)}")
        return None
