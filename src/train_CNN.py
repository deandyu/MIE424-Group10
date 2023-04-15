
from data.process_data import process_cnn_data
from models.CNN import create_cnn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

CMAP_LIGHT = sns.light_palette("#98D2AB", as_cmap=True)

def train_CNN(X_train, y_train, X_val_cnn, y_val_cnn):
  input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

  cnn = create_cnn(input_shape=input_shape, num_classes=len(GENRES))

  cnn.summary()

  cnn_history = cnn.fit(x=X_train, 
                      y=y_train, 
                      validation_data=(X_val_cnn, y_val_cnn), 
                      epochs=100,
                      verbose=True)
  
  return cnn, cnn_history

def test_CNN(cnn, X_test_cnn, y_test_cnn):
  y_pred_cnn = cnn.predict(X_test_cnn)
  print(f'CNN Accuracy: {cnn.evaluate(X_test_cnn, y_test_cnn)[1] * 100:.2f}%')

  confusion_matrix_cnn = confusion_matrix(np.argmax(y_test_cnn, axis=1), np.argmax(y_pred_cnn, axis=1))
  plt.figure(figsize = (16, 9))
  sns.heatmap(confusion_matrix_cnn, cmap=CMAP_LIGHT, annot=True, xticklabels = GENRES, yticklabels=GENRES)


def plot(cnn_history):
  plt.figure(figsize=(10, 8))
  plt.plot(cnn_history.epoch, cnn_history.history['loss'], label='Training Loss')
  plt.plot(cnn_history.epoch, cnn_history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epochs', fontsize=16)
  plt.ylabel('Accuracy', fontsize=16)
  plt.title('Training and Validation Loss', fontsize=18)
  plt.legend(fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  data_path = 'data'

  [spectrograms, labels] = joblib.load(os.path.join(data_path, "cnn_raw.joblib"))

  X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = process_cnn_data(spectrograms, labels)
  X_val_cnn, X_test_cnn, y_val_cnn, y_test_cnn = train_test_split(X_test_cnn, y_test_cnn, test_size=0.5, random_state=1)

  cnn, cnn_history = train_CNN(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn)

