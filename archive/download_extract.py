import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
import logging
import concurrent.futures

def download_extract(base_url, target_folder, month):
    file_url = f"{base_url}ist-daten-{month}.zip"
    response = requests.get(file_url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Progress bar for downloading
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        file_content = BytesIO()
        for data in response.iter_content(block_size):
            t.update(len(data))
            file_content.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            logging.error("Something went wrong during the download")
            return
        
        # Progress bar for extracting
        with ZipFile(file_content) as zip_file:
            file_list = zip_file.namelist()
            with tqdm(total=len(file_list), desc="Extracting") as pbar:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(zip_file.extract, file, target_folder) for file in file_list]
                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)
        
        logging.info(f"Downloaded and extracted: {file_url}")
    else:
        logging.error(f"Failed to download: {file_url} with status code {response.status_code}")
