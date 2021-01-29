from pathlib import Path
import numpy as np
import keras
from keras.models import Sequential
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import holistic

TIMESTEPS = 150 # Should exceed sample-count of a typical video
LANDMARK_COUNT = holistic.HAND_LANDMARK_COUNT * 2
POINT_DIM = 3

def build_model(labels):
    model = Sequential()
    model.add(keras.Input(shape = (TIMESTEPS, LANDMARK_COUNT * POINT_DIM)))
    model.add(layers.LSTM(64, name="lstm1", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm2"))
    model.add(layers.Dense(labels))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def load_data(dirname):
    words, words_test = [], []
    dataset, dataset_test = [], []

    for sign in Path(dirname).iterdir():

        for i, datafile in enumerate(sign.iterdir()):
            data = holistic.read_datafile(datafile)
            assert data.shape[0] <= TIMESTEPS
            # Zero-pad the data array
            zeros = np.zeros( (TIMESTEPS-data.shape[0],) + data.shape[1:] )
            data = np.concatenate((data, zeros), axis=0)
            if i % 3 == 2:
                dataset_test.append(data)
                words_test.append(sign.name)
            else:
                dataset.append(data)
                words.append(sign.name)

    t = Tokenizer(filters="\n\t")
    t.fit_on_texts(words)
    Y = t.texts_to_matrix(words)
    Y_test = t.texts_to_matrix(words_test)

    X = np.array(dataset)
    X_test = np.array(dataset_test)

    return X, X_test, Y, Y_test

def train_model(dirname):
    X, X_test, Y, Y_test = load_data(dirname)
    model = build_model(Y.shape[1])
    # TODO tweak batch size and epochs, add validation data
    history = model.fit(X, Y, epochs=10)
    score, acc = model.evaluate(X_test, Y_test)

    print(history.history)
    print("score: {} accuracy: {}".format(score, acc))

train_model("output")
