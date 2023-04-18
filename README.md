# MIE424 Final Project: Group10
## Comparative Analysis of Stacked Auto Encoders, Extreme Gradient Boosting, and Convolutional Neural Networks for Music Genre Classification

Music genre classification is a critical task in the field of Music Information Retrieval (MIR) that involves automatically categorizing musical pieces into predefined genres. For our MIE424 Final Project, we performed a comparative analysis of three state-of-the-art machine learning models for music genre classification: Stacked Denoising Auto Encoders (SDAs), Extreme Gradient Boosting (XGBoost), and Convolutional Neural Networks (CNNs).

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Make sure you have all the prerequisites installed
* [Git](https://git-scm.com/downloads)
* [Python](https://www.python.org/downloads/)

### Set-up

#### 1. Clone the repository to your local machine
```sh
$ git clone https://github.com/deandyu/MIE424-Group10
```

#### 2. Install dependencies (if you haven't done this before).
```powershell
pip install -r requirements.txt
```

<!-- DOWNLOAD DATASET-->
## Download Dataset
We have provided a script that downloads and extracts the dataset. Download the dataset using the script:

```powershell
python src/data/download_data.py
```

Alternatively you can download the dataset from [kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), and extract the zip file in the root directory of this repository. Please ensure that the resulting directory structure is as follows:

```python
.
├── data
│   └── genres_original
│   └── images_original
```

<!-- FEATURE EXTRACTION -->
## Feature Extraction
Extract numerical features for the SDA and XGBoost models:

```powershell
python src/features/extract_features.py
```

Extract Mel Spectrograms for the CNN model:

```powershell
python src/features/extract_spectrogram.py
```

<!-- MODEL TRAINING -->
## Model Training
Train the SDA model:

```powershell
python src/train_SDA.py
```

Train the XGBoost model:

```powershell
python src/train_XGBoost.py
```

Train the CNN model:

```powershell
python src/train_CNN.py
```