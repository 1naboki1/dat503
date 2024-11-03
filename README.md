# Data Processing with Dask

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
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
            filemode='w'  # Overwrite the log file on each run
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

3. Run the data processing script:
    ```sh
    python dat503.py
    ```

## Functions Overview

### `download_extract(base_url, target_folder, month)`
Download and extract a ZIP file from the specified URL.

**Parameters:**
- `base_url` (str): The base URL for the data files.
- `target_folder` (str): The folder where the extracted files should be saved.
- `month` (str): The month for which data should be downloaded.

### `load_and_preprocess_data(train_folder, filters, output_file_path, exclude_columns, delimiter=';')`
Load and preprocess data from the specified folder.

**Parameters:**
- `train_folder` (str): The folder containing the CSV files to be processed.
- `filters` (dict): A dictionary of filters to apply to the data.
- `output_file_path` (str): The file path where the processed data should be saved.
- `exclude_columns` (list): A list of columns to exclude from the data.
- `delimiter` (str): The delimiter used in the CSV files.

**Returns:**
- `str`: The path to the saved Parquet file, or `None` if an error occurred.

### `_get_csv_files(folder)`
Get a list of CSV files in the specified folder.

**Parameters:**
- `folder` (str): The folder to search for CSV files.

**Returns:**
- `list`: A list of file paths to the CSV files in the folder.

### `_load_data(data_files, delimiter)`
Load data from CSV files into a Dask DataFrame.

**Parameters:**
- `data_files` (list): A list of file paths to the CSV files.
- `delimiter` (str): The delimiter used in the CSV files.

**Returns:**
- `dask.dataframe.DataFrame`: The loaded data, or `None` if an error occurred.

### `_exclude_columns(data, exclude_columns)`
Exclude specified columns from the data.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data from which columns should be excluded.
- `exclude_columns` (list): A list of columns to exclude.

**Returns:**
- `dask.dataframe.DataFrame`: The data with specified columns excluded, or `None` if an error occurred.

### `_encode_categorical_columns(data)`
Encode all columns to int64 using Dask's parallel processing.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data to be encoded.

**Returns:**
- `dask.dataframe.DataFrame`: The encoded data, or `None` if an error occurred.

### `preprocess_and_save_data(data, filters, exclude_columns, output_file_path)`
Preprocess and save data to a Parquet file.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data to be processed.
- `filters` (dict): A dictionary of filters to apply to the data.
- `exclude_columns` (list): A list of columns to exclude from the data.
- `output_file_path` (str): The file path where the processed data should be saved.

**Returns:**
- `str`: The path to the saved Parquet file, or `None` if an error occurred.

### `_apply_filters(data, filters)`
Apply filters to the data.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data to be filtered.
- `filters` (dict): A dictionary of filters to apply.

**Returns:**
- `dask.dataframe.DataFrame`: The filtered data, or `None` if an error occurred.

### `_preprocess_data(data)`
Preprocess the data by filling missing values and inferring object types.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data to be preprocessed.

**Returns:**
- `dask.dataframe.DataFrame`: The preprocessed data, or `None` if an error occurred.

### `_calculate_time_differences(data)`
Calculate time differences between specified columns.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The data for which time differences should be calculated.

**Returns:**
- `dask.dataframe.DataFrame`: The data with calculated time differences, or `None` if an error occurred.

### `save_processed_data(data, output_file_path)`
Save the processed data to a Parquet file.

**Parameters:**
- `data` (dask.dataframe.DataFrame): The processed data to be saved.
- `output_file_path` (str): The file path where the data should be saved.

**Returns:**
- `str`: The path to the saved Parquet file, or `None` if an error occurred.

## Example Workflow
1. **Configure Logging**: Set up logging to capture the process details.
2. **Define Constants**: Set the base URL, folder paths, and other constants.
3. **Download Data**: Download and extract data files if needed.
4. **Load and Preprocess Data**: Load CSV files into a Dask DataFrame, apply filters, exclude columns, preprocess data, calculate time differences, and encode categorical columns.
5. **Save Processed Data**: Save the processed data to a Parquet file.
6. **Train Model**: Load the processed data, split it into training and testing sets, train a RandomForest model, and evaluate its accuracy.

## Logging
Logs are saved to `dat503.log` with detailed information about each step, including any errors encountered.

## Notes
- Ensure that the `target_column` in the `dat503.py` script is replaced with the actual target column name from your dataset.
- Adjust the filters and excluded columns as needed based on your specific requirements.

## License
This project is licensed under the MIT License.
