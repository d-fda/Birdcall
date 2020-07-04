import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten

import librosa
import librosa.display

samples_path = Path('../../Data/Birdcalls/train_audio_samples.pkl')

# sampling rate
sr = 43
# 5s clip length
clip_length = int(sr*5)

df = pd.read_pickle(samples_path)

X = []
y = []
for i in tqdm(range(len(df[:1000]))):
    mel_spec = df.iloc[i]['mel spec']
    bird = df.iloc[i]['bird']
    n_clips = int(len(mel_spec[0][:])/clip_length)
    for j in range(0, n_clips*clip_length, clip_length):
        clip = mel_spec[:, j:j+clip_length]
        clip = np.expand_dims(clip, axis=2)
        X.append(clip)
        y.append(bird)

X = np.array(X)
y = np.array(y)

lb = LabelBinarizer()
y = lb.fit_transform(y)

print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train[0])


#Define Model
model = Sequential()

model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(128, clip_length, 1)))
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(12, 12), padding='same'))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(len(y[0]), activation='softmax'))

#Compile
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
print(model.summary())

#Train and Test The Model
history = model.fit(X_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(X_test, y_test))

model.save('birdcall_model')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()