
import numpy as np

from tqdm import tqdm
import os
import librosa
import joblib

from sklearn.preprocessing import normalize

def create_spectrogram(audio, sr):

    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    spectrogram = librosa.power_to_db(S=melspectrogram, ref=1.0)

    normalized_spectrogram = normalize(spectrogram)

    return normalized_spectrogram

def augment_samples(audio, sr, n_slices):

    audio_samples = np.array_split(audio, n_slices)

    augmented_samples = []

    for audio_sample in audio_samples:

      augmented_samples.append(audio_sample)

      for n_steps in [3, 5]:
          augmented_samples.append(librosa.effects.pitch_shift(y=audio_sample, sr=sr, n_steps=n_steps))

      for rate in [0.5, 1.5]:
          augmented_samples.append(librosa.effects.time_stretch(y=audio_sample, rate=rate))

      white_noise = np.random.randn(len(audio_sample))
      augmented_samples.append(audio_sample + 0.005 * white_noise)

    return augmented_samples

def get_cnn_data(path):

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    n_classes = len(genres)

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