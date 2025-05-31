import requests
from pathlib import Path
import shutil
import zipfile
import logging
import sys
logger = logging.getLogger(__name__)
stdout_handler =logging.StreamHandler(stream=sys.stdout)
format_output = logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s') # <-
stdout_handler.setFormatter(format_output)
# logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)

def save_remote_zip_extract(url: str, local_path: Path, filename: str, extract_to: Path | str): 
    """
    Downloads a zip file from a remote URL and extracts it to a local path.
    
    :param url: The URL of the remote zip file.
    :param local_path: The local path where the zip file will be saved and extracted.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        extract_to = Path(extract_to)
        if not extract_to.exists():
            extract_to.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", extract_to)
        # Extract the specified file from the zip
        if not local_path.suffix == '.zip':
            raise ValueError(f"Expected a zip file, but got {local_path.suffix}")
        if not local_path.exists():
            raise FileNotFoundError(f"Zip file {local_path} does not exist.")
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extract(member=filename, path=extract_to)
    else:
        raise ValueError(f"Failed to download zip file: {response.status_code}")

def save_remote_file(url: str, local_path: Path):
    """
    Downloads a file from a remote URL and saves it to a local path.
    
    :param url: The URL of the remote file.
    :param local_path: The local path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    else:
        raise ValueError(f"Failed to download file: {response.status_code}")     

def save_misaki_font(target_dir: Path | str = "font", filename_in_zip: str = "misaki_gothic.png",
    local_zip_path = Path("misaki_png.zip"),
    url = "https://littlelimit.net/arc/misaki/misaki_png_2021-05-05a.zip"):
    """
    Downloads the Misaki Gothic font from a remote URL and extracts it to a local directory.
    """
    extract_to = Path(target_dir)
    try:
        save_remote_zip_extract(url, local_zip_path, filename_in_zip, extract_to)
        logger.info("File '%s' has extracted to '%s'", filename_in_zip, extract_to)
    except Exception as e:
        logger.error("An error occurred: %s", e)     
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)