from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

samples_path = Path('../../Data/Birdcalls/train_audio_samples.pkl')

# sampling rate
sr = 43
# 5s clip length
clip_length = int(sr*5)

df = pd.read_pickle(samples_path)

X = []
y = []
for i in tqdm(range(len(df))):
    mel_spec = df.iloc[i]['mel spec']
    bird = df.iloc[i]['bird']
    n_clips = int(len(mel_spec[0][:])/clip_length)
    for j in range(0, n_clips*clip_length, clip_length):
        clip = mel_spec[:, j:j+clip_length]
        X.append(clip)
        y.append(bird)

X = np.array(X)
y = np.array(y)

print(type(X))
print(len(X))