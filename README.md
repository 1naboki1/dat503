# dat503


## Requirements
- Python 3.7+
- Dask
- Dask-ML
- scikit-learn
- pandas
- pyarrow
- tqdm
- requests

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/dat503.git
    cd dat503
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Configure the logging settings in `dat503.py`:
    ```python
    def configure_logging():
        """Configure logging settings."""
        logging.basicConfig(
            filename='dat503.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
        )
        logging.captureWarnings(True)
    ```

2. Set the constants in `dat503.py`:
    ```python
    BASE_URL = "https://opentransportdata.swiss/wp-content/uploads/ist-daten-archive/"
    TRAIN_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'train')
    FORCE_DOWNLOAD = False  # Set to True to download the data
    NUM_MONTHS = 3  # Number of months to download
    TRAIN_FILTERS = {'LINIEN_TEXT': ['IC2', 'IC3', 'IC5', 'IC6', 'IC8', 'IC21']}
    TRAIN_OUTPUT_FILE_PATH = os.path.join(TRAIN_FOLDER, 'working', 'processed_data.parquet')
    TRAIN_EXCLUDE_COLUMNS = ['PRODUKT_ID', 'BETREIBER_NAME', 'BETREIBER_ID', 'UMLAUF_ID', 'VERKEHRSMITTEL_TEXT', 'AN_PROGNOSE_STATUS', 'AB_PROGNOSE_STATUS', 'HALTESTELLEN_NAME']
    ```

3. Run the main script:
    ```sh
    python dat503.py
    ```

## Functions
### `dat503.py`
- `configure_logging()`: Configures logging settings.
- `remove_existing_data(folder)`: Removes existing data in the specified folder.
- `calculate_months(num_months)`: Calculates the list of months to download data for.
- `download_data_if_needed(base_url, train_folder, months, force_download)`: Downloads and extracts data if needed.
- `main()`: Main function to orchestrate data processing and model training.

### `data_processing.py`
- `load_and_preprocess_data(train_folder, filters, output_file_path, exclude_columns, delimiter=';')`: Loads and preprocesses data from the specified folder.
- `_get_csv_files(folder)`: Gets a list of CSV files in the specified folder.
- `_load_data(data_files, delimiter)`: Loads data from CSV files into a Dask DataFrame.
- `_exclude_columns(data, exclude_columns)`: Excludes specified columns from the data.
- `_encode_categorical_columns(data)`: Encodes all columns to int64 using Dask's parallel processing.
- `preprocess_and_save_data(data, filters, exclude_columns, output_file_path)`: Preprocesses and saves data to a Parquet file.
- `_apply_filters(data, filters)`: Applies filters to the data.
- `_preprocess_data(data)`: Preprocesses the data by filling missing values and inferring object types.
- `save_processed_data(data, output_file_path)`: Saves the processed data to a Parquet file.

### `download_extract.py`
- `download_extract(base_url, target_folder, month)`: Downloads and extracts data from the specified URL.

## License
This project is licensed under the MIT License.
