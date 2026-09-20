import urllib.request
import zipfile
from pathlib import Path
import os

def download_and_extract():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = models_dir / "indiclid-ftn.zip"
    url = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip"
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(models_dir)
        
    print("Extraction complete.")
    
if __name__ == "__main__":
    download_and_extract()
