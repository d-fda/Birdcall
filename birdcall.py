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
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten
from keras.models import model_from_json

import librosa
import librosa.display

df = pd.read_pickle(r"C:\Users\David D'Amario\Data\Birdcalls\train_audio_samples.pkl")

X = df['mel spec']
y = df['bird'].to_numpy()

X = np.stack(X, axis=0)

X = np.resize(X, (len(X), 128, 216, 1))

lb = LabelBinarizer()
y = lb.fit_transform(y)

print(y.shape)
print(X.shape)

print(X[0].shape)
print(y[0].shape)

# load json and create model
json_file = open('autoencoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)
# load weights into new model
autoencoder.load_weights('autoencoder.h5')

# Compile
autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
print(autoencoder.summary())

for i in range(len(X)):
    X[i] = autoencoder.predict(np.array([X[i]]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define Model
model = Sequential()

model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same', input_shape=X_train[0].shape))
model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(len(y[0]), activation='softmax'))

# Compile
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
print(model.summary())

# Train and Test The Model
history = model.fit(X_train, y_train, batch_size=4, epochs=40, verbose=1, validation_data=(X_test, y_test))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

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
