from data.process_data import process_data
from models.SDA import create_sdae_svc
from evaluate import get_precision, get_recall, get_f1

import pandas as pd
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

def train_SDA(X_train_sda, X_val_sda, y_train_sda, y_val_sda):

  stacked_encoder_1, svc_1 = create_sdae_svc(X_train_sda, X_val_sda, y_train_sda, y_val_sda, num_layers=1, layer_sizes=[64], kernel='rbf', gamma=1, C=120, learning_rate=0.01, epochs=50, batch_size=64)

  X_train_features_1 = stacked_encoder_1.predict(X_train_sda)
  X_val_features_1 = stacked_encoder_1.predict(X_val_sda)

  sda_history_1 = svc_1.fit(X=X_train_features_1, y=to_categorical(y_train_sda))

  return stacked_encoder_1, svc_1, sda_history_1


def test_SDA(stacked_encoder_1, svc_1, X_test_sda, y_test_sda):
  X_test_features_1 = stacked_encoder_1.predict(X_test_sda)
  y_pred_1 = svc_1.predict(X_test_features_1)

  sda_accuracy_1 = accuracy_score(to_categorical(y_test_sda), y_pred_1)
  sda_recall_1 = get_recall(tf.cast(to_categorical(y_test_sda), tf.float32), y_pred_1)
  sda_precision_1 = get_precision(tf.cast(to_categorical(y_test_sda), tf.float64), y_pred_1)
  sda_f1_1 = get_f1(tf.cast(to_categorical(y_test_sda), tf.float64), y_pred_1)

  print(f'SVC Accuracy: {sda_accuracy_1 * 100:.2f}%')
  print(f'SVC Recall: {sda_recall_1 * 100:.2f}%')
  print(f'SVC Precision: {sda_precision_1 * 100:.2f}%')
  print(f'SVC F1 Score: {sda_f1_1 * 100:.2f}%')

if __name__ == "__main__":
  data_path = 'data'
  csv_name = 'GTZAN.csv'

  GTZAN = pd.read_csv(os.path.join(data_path, csv_name))

  # Acquire training, validation, and testing sets
  X_train_sda, X_test_sda, y_train_sda, y_test_sda = process_data(GTZAN)
  X_val_sda, X_test_sda, y_val_sda, y_test_sda = train_test_split(X_test_sda, y_test_sda, test_size=0.2, random_state=1)

  stacked_encoder_1, svc_1, sda_history_1 = train_SDA(X_train_sda, X_val_sda, y_train_sda, y_val_sda)

  test_SDA(stacked_encoder_1, svc_1, X_test_sda, y_test_sda)