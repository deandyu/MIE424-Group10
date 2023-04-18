from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import os

def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    url = "https://storage.googleapis.com/kaggle-data-sets/568973/1032238/compressed/Data.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230418%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230418T005138Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5b8f0d48277a2ddfc67738cd2762c0c17fd8ce5ea6b1f1969980fc6ee9667850514bf212a2430f614ecc485b3fed540790b4a5049a5d4de18099428eee75d5bcb3a7a325376a6eb4e96dab8282d4c85fa1b46e5c2dab5d0bd109d89cc5afe17e78f52d5cc8f8e6398c91eeba14a590f2988a45c605cdf93181b5470e43df64a3e39a1e2340145d33a702e5eaec695193bd941923608e98c626606ea5e40330d16577ff25450c9d9261a4bffc7e90f6e087faf293f9d6c00075855b695ac6eef1f6886aa7e55aee15f25418ceb73a7d0480dbf3e856a3cd2fc081f0601be745067433ab351d1219c57f76b0a02c874e7dfa82bbe51d6daae71e8ac3ab8db15649"
    data_directory = "data"
    download_and_unzip(url, data_directory)

    # Remove Corrupted .wav file
    os.remove('data/genres_original/jazz/jazz.00054.wav')