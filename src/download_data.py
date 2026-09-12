import os
import urllib.request
import zipfile
import sys

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    def reporthook(blocknum, blocksize, totalsize):
        readso = blocknum * blocksize
        if totalsize > 0:
            percent = readso * 1e2 / totalsize
            sys.stdout.write(f"\r{percent:5.1f}% {readso} / {totalsize}")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest, reporthook)
    print("\nDownload finished.")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip finished.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fma_dir = os.path.join(base_dir, 'data', 'raw', 'fma')
    deam_dir = os.path.join(base_dir, 'data', 'raw', 'deam')
    
    os.makedirs(fma_dir, exist_ok=True)
    os.makedirs(deam_dir, exist_ok=True)
    
    # FMA
    fma_meta_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    fma_medium_url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    
    download_file(fma_meta_url, os.path.join(fma_dir, "fma_metadata.zip"))
    unzip_file(os.path.join(fma_dir, "fma_metadata.zip"), fma_dir)
    
    download_file(fma_medium_url, os.path.join(fma_dir, "fma_medium.zip"))
    unzip_file(os.path.join(fma_dir, "fma_medium.zip"), fma_dir)
    
    # DEAM
    deam_audio_url = "https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip"
    deam_ann_url = "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip"
    deam_feat_url = "https://cvml.unige.ch/databases/DEAM/features.zip"
    
    download_file(deam_audio_url, os.path.join(deam_dir, "DEAM_audio.zip"))
    unzip_file(os.path.join(deam_dir, "DEAM_audio.zip"), deam_dir)
    
    download_file(deam_ann_url, os.path.join(deam_dir, "DEAM_Annotations.zip"))
    unzip_file(os.path.join(deam_dir, "DEAM_Annotations.zip"), deam_dir)
    
    download_file(deam_feat_url, os.path.join(deam_dir, "features.zip"))
    unzip_file(os.path.join(deam_dir, "features.zip"), deam_dir)
    
    print("All downloads and extractions complete!")
