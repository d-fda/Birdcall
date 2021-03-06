import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import os

import warnings

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

import librosa
import librosa.display

audio_path = r"C:\Users\David D'Amario\Data\Birdcalls\train_audio"
df_train = pd.read_csv(r"C:\Users\David D'Amario\Data\Birdcalls\train.csv")
df_test = pd.read_csv(r"C:\Users\David D'Amario\Data\Birdcalls\test.csv")

audio_samples = []


def get_samples(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave_data, sr = librosa.load(file_path)

    n_clips = int(len(wave_data)/(sr*5))
    clip_index = np.random.choice(n_clips)

    wave_data = wave_data[clip_index:clip_index + sr*5]

    mel_spec = librosa.feature.melspectrogram(wave_data, fmin=300)
    mel_spec = librosa.power_to_db(mel_spec)
    mel_spec = (mel_spec - np.min(mel_spec))/np.ptp(mel_spec)

    return mel_spec

for i in tqdm(range(len(df_train))):
    audio_filename = df_train['filename'][i]
    bird = df_train['ebird_code'][i]
    file_path = os.path.join(audio_path, bird, audio_filename)

    try:
        mel_spec = get_samples(file_path)
    except:
        continue


    new_item = {'mel spec': mel_spec, 'bird': bird, 'id': audio_filename}
    audio_samples.append(new_item)

df_train_audio = pd.DataFrame(audio_samples)

df_train_audio.to_pickle(r"C:\Users\David D'Amario\Data\Birdcalls\train_audio_samples.pkl")

print(len(df_train_audio))
print(df_train_audio.head())
