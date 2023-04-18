
import numpy as np

from tqdm import tqdm
import os
import librosa
import joblib
from typing import List, Tuple

from sklearn.preprocessing import normalize

def create_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Create a normalized spectrogram of an audio file.

    Parameters:
        audio (np.ndarray): The audio data as a 1D NumPy array.
        sr (int): The sample rate of the audio file.

    Returns:
        np.ndarray: A normalized spectrogram of the audio file as a 2D NumPy array.

    """
    # Compute the mel spectrogram of the audio data
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert the mel spectrogram to decibels
    spectrogram = librosa.power_to_db(S=melspectrogram, ref=1.0)

    # Normalize the spectrogram to have values between 0 and 1
    normalized_spectrogram = normalize(spectrogram)

    # Return the normalized spectrogram as a 2D NumPy array
    return normalized_spectrogram

def augment_samples(audio: np.ndarray, sr: int, n_slices: int) -> list:
    """
    Augment an audio signal by applying various transformations to its slices.

    Parameters:
        audio (np.ndarray): The audio data as a 1D NumPy array.
        sr (int): The sample rate of the audio file.
        n_slices (int): The number of slices to split the audio data into.

    Returns:
        list: A list of augmented audio samples, each represented as a 1D NumPy array.

    """
    # Split the audio data into slices
    audio_samples = np.array_split(audio, n_slices)

    # Create a list to store the augmented samples
    augmented_samples = []

    # Apply different transformations to each slice
    for audio_sample in audio_samples:

        # Add the original slice to the list of augmented samples
        augmented_samples.append(audio_sample)

        # Apply pitch shifting to the slice with 3 and 5 steps
        for n_steps in [3, 5]:
            augmented_samples.append(librosa.effects.pitch_shift(y=audio_sample, sr=sr, n_steps=n_steps))

        # Apply time stretching to the slice with 0.5 and 1.5 rates
        for rate in [0.5, 1.5]:
            augmented_samples.append(librosa.effects.time_stretch(y=audio_sample, rate=rate))

        # Add white noise to the slice
        white_noise = np.random.randn(len(audio_sample))
        augmented_samples.append(audio_sample + 0.005 * white_noise)

    # Return the list of augmented samples
    return augmented_samples

def get_cnn_data(path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load and preprocess the data for a CNN model.

    Parameters:
        path (str): The path to the GTZAN dataset.

    Returns:
        Tuple[List[np.ndarray], List[int]]: A tuple containing a list of spectrograms and a list of corresponding labels.

    """
    # Define the music genres in GTZAN
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Determine the number of music genres
    n_classes = len(genres)

    # Define the number of audio slices
    n_slices = 3

    spectrograms = []
    labels = []

    for genre in tqdm(genres):

      genre_dir = os.path.join(path, 'genres_original', genre)
      genre_index = genres.index(genre)

      for filename in tqdm(os.listdir(genre_dir)):

          if filename.endswith('.wav'):

              filepath = os.path.join(genre_dir, filename)
              audio, sr = librosa.load(filepath, duration=29)

              input_length = len(audio) // n_slices
              
              augmented_audios = augment_samples(audio, sr, n_slices)

              for aug_audio in augmented_audios:

                  if len(aug_audio) > input_length:
                      aug_audio = aug_audio[:input_length]
                  else:
                      aug_audio = np.pad(aug_audio, (0, max(0, input_length - len(aug_audio))))

                  spectrogram = create_spectrogram(aug_audio, sr)
                  spectrogram = np.expand_dims(spectrogram, axis=-1)
                  spectrograms.append(spectrogram)
                  labels.append(genre_index)

    return spectrograms, labels

if __name__ == "__main__":
  data_path = 'data'
  csv_name = 'GTZAN.csv'

  spectrograms, labels = get_cnn_data(data_path)

  joblib.dump([spectrograms, labels], os.path.join(data_path, "cnn_raw.joblib"))