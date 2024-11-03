import os
import shutil
import logging
from datetime import datetime, timedelta
from download_extract import download_extract
from data_processing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Configure logging
logging.basicConfig(filename='dat503.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)

# Define the base URL and the target folder
base_url = "https://opentransportdata.swiss/wp-content/uploads/ist-daten-archive/"
train_folder = os.path.join(os.path.dirname(__file__), 'data', 'train')

# Variables to control the download process
force_download = True  # Set to True to download the data
num_months = 2  # Number of months to download

# If force_download is True, remove all existing data
if force_download and os.path.exists(train_folder):
    shutil.rmtree(train_folder)

# Calculate the last 'num_months' months
today = datetime.today()
months = [(today - timedelta(days=30 * i)).strftime("%Y-%m") for i in range(num_months)]

# Download and extract the files if force_download is True
if force_download:
    os.makedirs(train_folder, exist_ok=True)
    for month in months:
        download_extract(base_url, train_folder, month)
else:
    logging.info("Skipping download and extraction as force_download is set to False.")

# Load and preprocess the data
processed_data_file = load_and_preprocess_data(train_folder)

# Load the processed data
data = pd.read_csv(processed_data_file)

if data is not None:
    # Define features and target variable
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy}")
else:
    logging.error("Data loading and preprocessing failed.")
