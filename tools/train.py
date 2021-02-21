from random import random
from sys import argv
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import holistic

TIMESTEPS = 250 # Should exceed sample-count of a typical video
POINT_DIM = 3

def build_model(labels):
    model = Sequential()
    model.add(keras.Input(shape = (TIMESTEPS, holistic.LANDMARK_COUNT * POINT_DIM)))
    model.add(layers.LSTM(64, name="lstm1", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm2", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm3"))
    model.add(layers.Dense(labels, activation="softmax"))
    adam = Adam(lr = 0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model

def label_to_matrix(labels, label_list):
    table = { l:i for i, l in enumerate(labels) }
    indices = [table[l] for l in label_list]
    return tf.one_hot(indices, len(table))

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
def load_data(dirname):
    labels = []
    words, words_test, words_val = [], [], []
    dataset, dataset_test, dataset_val = [], [], []

    for sign in Path(dirname).iterdir():
        labels.append(sign.name)
        for i, datafile in enumerate(sign.iterdir()):
            data = holistic.read_datafile(datafile)
            assert data.shape[0] <= TIMESTEPS, "TIMESTEP must be above {}".format(data.shape[0])
            # Zero-pad the data array
            zeros = np.zeros( (TIMESTEPS-data.shape[0],) + data.shape[1:] )
            data = np.concatenate((data, zeros), axis=0)

            data_loc = random()
            if data_loc < TEST_SPLIT:
                dataset_ref = dataset_test
                words_ref = words_test
            elif data_loc < TEST_SPLIT + VALIDATION_SPLIT:
                dataset_ref = dataset_val
                words_ref = words_val
            else:
                dataset_ref = dataset
                words_ref = words
            dataset_ref.append(data)
            words_ref.append(sign.name)

    t = Tokenizer(filters="\n\t")
    t.fit_on_texts(words)
    Y = t.texts_to_matrix(words)
    Y_test = t.texts_to_matrix(words_test)
    Y_val = t.texts_to_matrix(words_val)

    X = np.array(dataset)
    X_test = np.array(dataset_test)
    X_val = np.array(dataset_val)

    return X, Y, X_test, Y_test, X_val, Y_val

def plot_data(name, data):
    plt.figure()
    plt.plot(data)
    plt.title(name)

def train_model(dirname, epochs=100, batch_size=32):
    X, Y, X_test, Y_test, X_val, Y_val = load_data(dirname)
    print("Size of training set = {}, test set = {}, validation set = {}".format(X.shape[0], X_test.shape[0], X_val.shape[0]))
    model = build_model(Y.shape[1])
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))
    score, acc = model.evaluate(X, Y, batch_size=batch_size)

    print("loss: {} accuracy: {}".format(score, acc))
    for name, data in history.history.items():
        plot_data(name, data)
    plt.show()
    return model

# Usage: data_dir [model_file]
if __name__ == "__main__":
    data_dir = argv[1]
    model = train_model(data_dir)

    if len(argv) >= 3:
        model_file = argv[2]
        model.save(model_file)
