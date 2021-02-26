from collections import Counter
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

TIMESTEPS = 120
POINT_DIM = 3

def build_model(labels, frame_dim):
    model = Sequential()
    model.add(keras.Input(shape = (TIMESTEPS, frame_dim)))
    model.add(layers.LSTM(64, name="lstm1", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm2"))
    model.add(layers.Dense(labels, activation="softmax"))
    adam = Adam(lr = 0.0002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def load_data(dirname):
    for sign in Path(dirname).iterdir():
        for datafile in sign.iterdir():
            yield (holistic.read_datafile(datafile), sign.name)

def truncate_data(data_iter, timesteps):
    for data, sign in iter(data_iter):
        if data.shape[0] > timesteps:
            data = data[:timesteps]
        yield (data, sign)

def extend_data(data_iter, timesteps):
    for data, sign in iter(data_iter):
        if data.shape[0] < timesteps:
            zeros = np.zeros( (timesteps-data.shape[0],) + data.shape[1:] )
            data = np.concatenate((data, zeros), axis=0)
        yield (data, sign)

TEST_SPLIT = 2
def split_data(data_iter):
    dataset, dataset_test = [], []
    for i, (data, sign) in enumerate(data_iter):
        data_loc = i % 10
        if data_loc < TEST_SPLIT:
            dataset_ref = dataset_test
        else:
            dataset_ref = dataset
        dataset_ref.append((data, sign))
    return (dataset, dataset_test)

def load_and_process_data(dirname):
    data_iter = load_data(dirname)
    data_iter = truncate_data(data_iter, TIMESTEPS)
    data_iter = extend_data(data_iter, TIMESTEPS)
    data, data_test = split_data(data_iter)    

    dataset = [d for d, w in data]
    words = [w for d, w in data]
    dataset_test = [d for d, w in data_test]
    words_test = [w for d, w in data_test]

    t = Tokenizer(filters="\n\t")
    t.fit_on_texts(words)
    Y = t.texts_to_matrix(words)
    Y_test = t.texts_to_matrix(words_test)

    X = np.array(dataset)
    X_test = np.array(dataset_test)

    return X, Y, X_test, Y_test

def plot_data(name, data):
    plt.figure()
    plt.plot(data)
    plt.title(name)

def train_model(dirname, epochs=300, batch_size=64, val_split=0.2):
    X, Y, X_test, Y_test = load_and_process_data(dirname)
    print("Size of training set = {}, test set = {}".format(X.shape[0], X_test.shape[0]))
    model = build_model(Y.shape[1], X.shape[2])
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=val_split)
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print(model.predict(X_test))

    print("score: {} accuracy: {}".format(score, acc))
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
