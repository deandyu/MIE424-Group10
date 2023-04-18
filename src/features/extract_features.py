import pandas as pd
import numpy as np

import os
import librosa

from tqdm import tqdm

def load_gtzan_data(path: str) -> pd.DataFrame:
    """
    Load the GTZAN dataset from the given path and extract audio features for each music slice.

    Parameters:
        path (str): The path to the directory containing the GTZAN dataset.

    Returns:
        pd.DataFrame: A Pandas dataframe containing the extracted features and labels for each slice.

    """
    # Different music genres in GTZAN
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    feature_cols = ['filename', 'slice', 'zcr_mean', 'zcr_var', 'rmse_mean', 
                    'rmse_var', 'sc_mean', 'sc_var', 'sbw_mean', 'sbw_var', 
                    'sro_mean', 'sro_var', 'tempo', 'harmony_mean', 
                    'perc_mean', 'harmony_var', 'perc_var']

    for i in range(1, 21):
        feature_cols += [f'mfcc{i}_mean', f'mfcc{i}_var', f'dmfcc{i}_mean', f'dmfcc{i}_var']

    for i in range(1, 13):
        feature_cols += [f'cstft{i}_mean', f'cstft{i}_var']

    feature_cols += ['label']

    data = []

    sr = 22050
    total_samples = 29 * sr
    num_slices = 10
    samples_in_slice = int(total_samples / num_slices)
    
    for genre in tqdm(genres):
      
        genre_dir = os.path.join(path, 'genres_original', genre)

        for filename in tqdm(os.listdir(genre_dir)):

          if filename.endswith('.wav'):
              
            # Load audio file
            filepath = os.path.join(genre_dir, filename)
            audio, sr = librosa.load(filepath, duration=29)

            for s in range(num_slices):

              start_sample = samples_in_slice * s
              end_sample = start_sample + samples_in_slice
              slice_audio = audio[start_sample:end_sample]

              # Extract features and their mean and variance
              zcr = librosa.feature.zero_crossing_rate(y=slice_audio)
              rmse = librosa.feature.rms(y=slice_audio)
              mag = np.abs(librosa.stft(slice_audio))

              f = librosa.fft_frequencies(sr=sr, n_fft=2048)
              sc = librosa.feature.spectral_centroid(S=mag, freq=f)
              sbw = librosa.feature.spectral_bandwidth(S=mag, freq=f, p=2)
              sro = librosa.feature.spectral_rolloff(S=mag, freq=f)

              mfcc = librosa.feature.mfcc(y=slice_audio, sr=sr, n_mfcc=20)
              dmfcc = librosa.feature.delta(mfcc)
              cstft = librosa.feature.chroma_stft(y=slice_audio, sr=sr, n_chroma=12)

              # Extract tempo
              tempo, _ = librosa.beat.beat_track(y=slice_audio, sr=sr)

              # Extract harmony and perceptual features
              S = librosa.feature.melspectrogram(y=slice_audio, sr=sr)
              S_harmonic, S_percussive = librosa.effects.hpss(S)

              # Append features and label
              row = [filename, s] + \
                    [np.mean(feature) for feature in [zcr, rmse, sc, sbw, sro]] + \
                    [np.var(feature) for feature in [zcr, rmse, sc, sbw, sro]] + \
                    [tempo, np.mean(S_harmonic), np.mean(S_percussive), np.var(S_harmonic), np.var(S_percussive)] + \
                    [np.mean(feature) for feature in mfcc] + [np.var(feature) for feature in mfcc] + \
                    [np.mean(feature) for feature in dmfcc] + [np.var(feature) for feature in dmfcc] + \
                    [np.mean(feature) for feature in cstft] + [np.var(feature) for feature in cstft] + \
                    [genre]

              data.append(row)

    # Create dataframe
    df = pd.DataFrame(data, columns=feature_cols)
    df = df.sort_values(['filename', 'slice'], ascending=[True, True]).reset_index(drop=True)

    return df

if __name__ == "__main__":
  data_path = 'data'
  csv_name = 'GTZAN.csv'

  gtzan_df = load_gtzan_data(data_path)

  # Save the prediction dataframe as a CSV file
  gtzan_df.to_csv(os.path.join(data_path, csv_name), index=False)