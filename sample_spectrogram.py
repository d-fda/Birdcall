from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

samples_path = Path('../../Data/Birdcalls/train_audio_samples.pkl')

# sampling rate
sr = 43
# 5s clip length
clip_length = int(sr*5)

# read in spectrogram of mp3s
df = pd.read_pickle(samples_path)

df_clipped = []
for i in tqdm(range(len(df))):
    mel_spec = df.iloc[i]['mel spec']
    bird = df.iloc[i]['bird']
    filename = df.iloc[i]['id']

    # calculate how many 5s clips are in the file
    n_clips = int(len(mel_spec[0][:])/clip_length)

    n = 0
    # j is start index of clip to be sliced
    for j in range(0, n_clips*clip_length, clip_length):
        clip = mel_spec[:, j:j+clip_length]
        # expand to make shape (l,w,d) d = 1
        clip = np.expand_dims(clip, axis=2)
        # populate row of dataframe, fileid marks mp3 filename AND start time (start time=n*clip_length)
        row = {'mel spec': clip,
               'bird': bird,
               'fileid': filename + '_' + str(n)}
        n += 1
        df_clipped.append(row)

# save list to df
df_clipped = pd.DataFrame(df_clipped)

# write to memory
df_clipped.to_pickle(r"C:\Users\David D'Amario\Data\Birdcalls\spectrogram_samples.pkl")

print(len(df_clipped))