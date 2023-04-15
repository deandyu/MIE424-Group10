from data.process_data import process_data
from models.SAE import create_sdae_svc

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

def train_SAE(data):
  # Acquire training, validation, and testing sets
  X_train_sda, X_test_sda, y_train_sda, y_test_sda = process_data(data)
  X_val_sda, X_test_sda, y_val_sda, y_test_sda = train_test_split(X_test_sda, y_test_sda, test_size=0.5, random_state=1)

  stacked_encoder_1, svc_1 = create_sdae_svc(X_train_sda, X_val_sda, y_train_sda, y_val_sda, num_layers=1, layer_sizes=[64], kernel='rbf', gamma=1, C=100, learning_rate=0.001, epochs=50, batch_size=16)

  X_train_features_1 = stacked_encoder_1.predict(X_train_sda)
  X_test_features_1 = stacked_encoder_1.predict(X_test_sda)
  X_val_features_1 = stacked_encoder_1.predict(X_val_sda)

  sda_history_1 = svc_1.fit(X=X_train_features_1, y=to_categorical(y_train_sda))

  y_pred_1 = svc_1.predict(X_test_features_1)
  accuracy_1 = accuracy_score(to_categorical(y_test_sda), y_pred_1)

  print(f'SVC Accuracy: {accuracy_1 * 100:.2f}%')
  
  stacked_encoder_2, svc_2 = create_sdae_svc(X_train_sda, X_val_sda, y_train_sda, y_val_sda, num_layers=2, layer_sizes=[64, 32], kernel='rbf', gamma=1, C=100, learning_rate=0.001, epochs=50, batch_size=16)

if __name__ == "__main__":
  data_path = 'data'
  csv_name = 'GTZAN.csv'

  GTZAN = pd.read_csv(os.path.join(data_path, csv_name))
  train_SAE(GTZAN)