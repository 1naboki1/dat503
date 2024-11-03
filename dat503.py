import os
import shutil
import logging
from datetime import datetime, timedelta
from download_extract import download_extract
from data_processing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dask.dataframe as dd

# Configure logging
def configure_logging():
    logging.basicConfig(
        filename='dat503.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
    )
    logging.captureWarnings(True)

# Define constants
BASE_URL = "https://opentransportdata.swiss/wp-content/uploads/ist-daten-archive/"
TRAIN_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'train')
FORCE_DOWNLOAD = False  # Set to True to download the data
NUM_MONTHS = 3  # Number of months to download
TRAIN_FILTERS = {'LINIEN_TEXT': ['IC2', 'IC3', 'IC5', 'IC6', 'IC8', 'IC21']}  # IC4 is cross-border and not in dataset
TRAIN_OUTPUT_FILE_PATH = os.path.join(TRAIN_FOLDER, 'working', 'processed_data.parquet')
TRAIN_EXCLUDE_COLUMNS = ['PRODUKT_ID', 'BETREIBER_NAME', 'BETREIBER_ID', 'UMLAUF_ID', 'VERKEHRSMITTEL_TEXT', 'AN_PROGNOSE_STATUS', 'AB_PROGNOSE_STATUS', 'HALTESTELLEN_NAME']  # Columns to exclude from processing

def remove_existing_data(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

def calculate_months(num_months):
    today = datetime.today()
    return [(today - timedelta(days=30 * i)).strftime("%Y-%m") for i in range(num_months)]

def download_data_if_needed(base_url, train_folder, months, force_download):
    if force_download:
        os.makedirs(train_folder, exist_ok=True)
        for month in months:
            download_extract(base_url, train_folder, month)
    else:
        logging.info("Skipping download and extraction as force_download is set to False.")

def main():
    configure_logging()

    if FORCE_DOWNLOAD:
        remove_existing_data(TRAIN_FOLDER)

    months = calculate_months(NUM_MONTHS)
    download_data_if_needed(BASE_URL, TRAIN_FOLDER, months, FORCE_DOWNLOAD)

    processed_data_file = load_and_preprocess_data(TRAIN_FOLDER, TRAIN_FILTERS, TRAIN_OUTPUT_FILE_PATH, TRAIN_EXCLUDE_COLUMNS)

    if processed_data_file is None:
        logging.error("Processed data file is None. Exiting.")
        exit(1)

    try:
        data = dd.read_parquet(processed_data_file).compute()
    except Exception as e:
        logging.error(f"Error loading processed data file: {e}")
        exit(1)

    if data is not None:
        X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
        y = data['target_column']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy}")
    else:
        logging.error("Data is None after loading the processed data file.")
        exit(1)

if __name__ == "__main__":
    main()
