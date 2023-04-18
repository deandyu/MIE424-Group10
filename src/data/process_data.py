
import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from keras.utils import to_categorical

def process_data(df: pd.DataFrame) -> tuple:
    """
    This function preprocesses the input DataFrame by dropping unnecessary columns, creating one-hot encoded columns
    for the categorical 'label' column, and standardizing the data using the standardize_data function. The data is
    then split into training and testing sets.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the data to be preprocessed.

    Returns:
        tuple: A tuple containing the standardized training and testing data and one-hot encoded labels
               (X_train, X_test, y_train, y_test).
    """

    # Drop unnecessary columns
    df = df.drop(['filename', 'slice'], axis=1)

    # Drop the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Create a LabelEncoder object and fit_transform the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

    # Standardize the data
    X_train, X_test = scale_data(X_train, X_test)

    # Convert the DataFrames to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    return X_train, X_test, y_train, y_test

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    This function scales the input data using the MinMaxScaler() method from scikit-learn. It creates a copy of 
    the input DataFrames, fits the scaler to the training data, and applies the same transformation to the test data.
    
    Parameters:
        X_train (pd.DataFrame): A pandas DataFrame containing the training data to be standardized.
        X_test (pd.DataFrame): A pandas DataFrame containing the test data to be standardized.
        
    Returns:
        tuple: A tuple containing the scaled training and testing data (X_train_standardized, X_test_standardized).
    """
    # Create copies of the input DataFrames
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit the scaler to the training data
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Transform both the training and testing data
    X_train_scaled.loc[:] = scaler.transform(X_train)
    X_test_scaled.loc[:] = scaler.transform(X_test)
                        
    return X_train_scaled, X_test_scaled

def process_cnn_data(spectograms: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process spectrograms and their corresponding labels for use in a CNN.

    Parameters:
        spectograms (List[np.ndarray]): A list of spectrograms as 2D NumPy arrays.
        labels (List[int]): A list of labels corresponding to each spectrogram.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the preprocessed training and testing data and labels as NumPy arrays.

    """
    X = np.array(spectograms)
    y = np.array(labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test