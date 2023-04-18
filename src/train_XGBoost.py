
from data.process_data import process_data
from models.XGBoost import evaluate_model
from evaluate import get_precision, get_recall, get_f1

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import joblib

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

CMAP_LIGHT = sns.light_palette("#98D2AB", as_cmap=True)
CMAP_DARK = sns.dark_palette("#98D2AB", as_cmap=True)

def train_XGBoost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb):
  model = XGBClassifier(objective='multi:softmax')
  best_xgb, best_xgb_parameters, best_xgb_accuracy = evaluate_model(model, X_train_xgb, y_train_xgb)

  joblib.dump([best_xgb, best_xgb_parameters, best_xgb_accuracy], "models/best_xgb.joblib")

  best_xgb = XGBClassifier(objective='multi:softmax', **best_xgb_parameters)

  # Train model with full training data
  best_xgb_history = best_xgb.fit(X=X_train_xgb, 
                                  y=y_train_xgb, 
                                  eval_set=[(X_train_xgb, y_train_xgb), (X_val_xgb, y_val_xgb)],
                                  eval_metric=['mlogloss', 'merror'], 
                                  verbose=True)
  
  return best_xgb, best_xgb_history

def test_XGBoost(best_xgb, X_test_xgb, y_test_xgb):
  y_pred_xgb = best_xgb.predict(X_test_xgb)

  # Evaluate classification performance
  xbg_accuracy = accuracy_score(y_test_xgb, y_pred_xgb)
  xbg_recall = get_recall(tf.cast(y_test_xgb, tf.float32), y_pred_xgb)
  xbg_precision = get_precision(tf.cast(y_test_xgb, tf.float32), y_pred_xgb)
  xbg_f1 = get_f1(tf.cast(y_test_xgb, tf.float32), y_pred_xgb)

  print(f'XGBoost Accuracy: {xbg_accuracy * 100:.2f}%')
  print(f'XGBoost Recall: {xbg_recall * 100:.2f}%')
  print(f'XGBoost Precision: {xbg_precision * 100:.2f}%')
  print(f'XGBoost F1 Score: {xbg_f1 * 100:.2f}%')

  # Plot confusion matrix
  confusion_matrix_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
  plt.figure(figsize = (16, 9))
  sns.heatmap(confusion_matrix_xgb, cmap=CMAP_LIGHT, annot=True, xticklabels=GENRES, yticklabels=GENRES)


def plot_feature_importance(best_xgb, GTZAN):
  # Plot feature importance
  importances = best_xgb.feature_importances_
  feature_names = GTZAN.iloc[:, 2:-1].columns.tolist()

  sorted_idx = np.argsort(importances)[::-1]
  sorted_importances = importances[sorted_idx][:20]
  sorted_feature_names = np.array(feature_names)[sorted_idx][:20]  

  n_colors = len(sorted_feature_names)
  colors = CMAP_DARK(np.linspace(0, 1, n_colors))

  fig, ax = plt.subplots(figsize=(10, 8))
  sns.barplot(x=sorted_importances, y=sorted_feature_names, palette=colors, ax=ax)

  ax.set_title("Feature Importance", fontsize=18)
  ax.set_xlabel("Importance", fontsize=16)
  ax.set_ylabel("Features", fontsize=16)
  ax.tick_params(labelsize=14)
  plt.show()

def plot_log_loss(best_xgb):
  # Get the evaluation results
  eval_results = best_xgb.evals_result()

  # Log loss
  train_logloss = eval_results['validation_0']['mlogloss']
  val_logloss = eval_results['validation_1']['mlogloss']

  # Accuracy (1 - merror)
  train_accuracy = [1 - x for x in eval_results['validation_0']['merror']]
  val_accuracy = [1 - x for x in eval_results['validation_1']['merror']]

  epochs = range(1, len(train_logloss) + 1)

  # Plot log loss
  plt.figure(figsize=(10, 8))
  plt.plot(epochs, train_logloss, label='Training Log Loss')
  plt.plot(epochs, val_logloss, label='Validation Log Loss')
  plt.xlabel('Epochs', fontsize=16)
  plt.ylabel('Log Loss', fontsize=16)
  plt.title('Log Loss', fontsize=18)
  plt.legend(fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  data_path = 'data'
  csv_name = 'GTZAN.csv'

  GTZAN = pd.read_csv(os.path.join(data_path, csv_name))
  
  X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = process_data(GTZAN)
  X_val_xgb, X_test_xgb, y_val_xgb, y_test_xgb = train_test_split(X_test_xgb, y_test_xgb, test_size=0.5, random_state=1)

  best_xgb, best_xgb_history = train_XGBoost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)

  test_XGBoost(best_xgb, X_test_xgb, y_test_xgb)

  plot_feature_importance(best_xgb, GTZAN)
  plot_log_loss(best_xgb)