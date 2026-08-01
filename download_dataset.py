import os
import requests

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists at {dest_path}. Skipping download.")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.2f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end='\r')
                else:
                    print(f"Downloaded: {downloaded / (1024*1024):.2f} MB", end='\r')
    print("\nDownload complete!")

if __name__ == "__main__":
    url = "https://huggingface.co/datasets/SquareBracket/fraud_detection/resolve/main/fraudTrain.csv"
    dest = os.path.join("dataset", "fraudTrain.csv")
    download_file(url, dest)
