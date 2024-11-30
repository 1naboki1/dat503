import os
import logging
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import pyarrow as pa
from archive.swiss_holidays import get_past_three_months_holidays


# Set pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

def load_and_preprocess_data(train_folder, filters, output_file_path, exclude_columns, delimiter=';'):
    """Load and preprocess data from the specified folder."""
    logging.info(f"Starting to load data from {train_folder}")
    
    data_files = _get_csv_files(train_folder)
    logging.info(f"Found {len(data_files)} CSV files to process.")
    
    data = _load_data(data_files, delimiter)
    if data is None:
        return None
    
    return preprocess_and_save_data(data, filters, exclude_columns, output_file_path)

def _get_csv_files(folder):
    """Get a list of CSV files in the specified folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def _load_data(data_files, delimiter):
    """Load data from CSV files into a Dask DataFrame."""
    try:
        with ProgressBar():
            logging.info("Loading data into Dask DataFrame.")
            data = dd.read_csv(data_files, delimiter=delimiter, dtype={'LINIEN_ID': 'object', 'UMLAUF_ID': 'object', 'BPUIC': 'float64'}, low_memory=False)
        logging.info("Data loaded into Dask DataFrame.")
        logging.info(f"Data columns: {data.columns}")
        return data
    except Exception as e:
        logging.error(f"Error loading data into Dask DataFrame: {e}")
        return None

def _exclude_columns(data, exclude_columns):
    """Exclude specified columns from the data."""
    try:
        logging.info(f"Excluding columns: {exclude_columns}")
        data = data.drop(columns=exclude_columns, errors='ignore')
        logging.info(f"Columns after exclusion: {data.columns}")
        return data
    except Exception as e:
        logging.error(f"Error excluding columns: {e}")
        return None
        
def _add_holidays(data):
    logging.info("adding holidays")
    try:
        schweizer_feiertage = get_past_three_months_holidays()
        logging.info(f"add new coloumn with holiday data {schweizer_feiertage}")
        # data['BETRIEBSTAG'] = dd.to_datetime(data['BETRIEBSTAG'], errors='coerce')
        def _ist_feiertag(BETRIEBSTAG):
            return BETRIEBSTAG in schweizer_feiertage
        
        data['ist_feiertag'] = data['BETRIEBSTAG'].map(_ist_feiertag, meta=('BETRIEBSTAG', 'bool'))
        data = data.compute()
        logging.info("iiiiiiiiiiiis gooooooooooooooooooooooooooooooooooooooooooooooooooooooood")
        return data
    except Exception as e:
        logging.error(f"Conny hat an blödsinn gmacht: {e}")
        return None



def preprocess_and_save_data(data, filters, exclude_columns, output_file_path):
    """Preprocess and save data to a Parquet file."""
    data = _apply_filters(data, filters)
    if data is None:
        return None
    
    data = _exclude_columns(data, exclude_columns)
    if data is None:
        return None
    
    data = _add_holidays(data)

    data = _preprocess_data(data)
    if data is None:
        return None
    
    return _save_data(data, output_file_path)

def _apply_filters(data, filters):
    """Apply filters to the data."""
    try:
        logging.info("Applying custom filters for AN_PROGNOSE_STATUS and AB_PROGNOSE_STATUS.")
        with ProgressBar():
            data = data[(data["AN_PROGNOSE_STATUS"] == "REAL") & (data["AB_PROGNOSE_STATUS"] == "REAL")]
        logging.info("Custom filters applied.")
    except Exception as e:
        logging.error(f"Error applying custom filters: {e}")
        return None

    if filters:
        for column, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            logging.info(f"Applying filter: {column} in {values}")
            try:
                if column in data.columns:
                    with ProgressBar():
                        data = data[data[column].isin(values)]
                    logging.info(f"Filter applied on column: {column}")
                else:
                    logging.error(f"Column {column} does not exist in the data.")
                    return None
            except Exception as e:
                logging.error(f"Error applying filter on column {column}: {e}")
                return None
    return data

def _preprocess_data(data):
    """Preprocess the data by filling missing values and inferring object types."""
    try:
        logging.info("Filling missing values with NaN.")
        with ProgressBar():
            data = data.fillna(np.nan)
            data = data.map_partitions(lambda df: df.infer_objects(copy=False))
        logging.info("Missing values filled with NaN and inferred object types.")
        return data
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return None

def _save_data(data, output_file_path):
    """Save the processed data to a Parquet file."""
    try:
        logging.info(f"Saving processed data to {output_file_path}")
        with ProgressBar():
            data = data.repartition(npartitions=20)
            schema = pa.schema([
                ('BETRIEBSTAG', pa.string()),
                ('FAHRT_BEZEICHNER', pa.string()),
                ('BETREIBER_ABK', pa.string()),
                ('LINIEN_ID', pa.string()),
                ('LINIEN_TEXT', pa.string()),
                ('ZUSATZFAHRT_TF', pa.bool_()),
                ('FAELLT_AUS_TF', pa.bool_()),
                ('BPUIC', pa.int64()),
                ('ANKUNFTSZEIT', pa.string()),
                ('AN_PROGNOSE', pa.string()),
                ('ABFAHRTSZEIT', pa.string()),
                ('AB_PROGNOSE', pa.string()),
                ('DURCHFAHRT_TF', pa.bool_()),
                ('ist_feiertag', pa.bool_()),
                ('__null_dask_index__', pa.int64())
            ])
            output_file_path = output_file_path.replace('.csv', '.parquet')
            data.to_parquet(output_file_path, engine='pyarrow', compression='snappy', schema=schema, write_metadata_file=False)
        logging.info("Successfully processed and saved data to a Parquet file.")
        return output_file_path
    except Exception as e:
        logging.error(f"Error saving processed data to Parquet file: {e}")
        return None
