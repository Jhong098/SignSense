from sys import argv
from pathlib import Path
import numpy as np
import keras
from keras.models import Sequential
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import holistic

TIMESTEPS = 250 # Should exceed sample-count of a typical video
LANDMARK_COUNT = holistic.HAND_LANDMARK_COUNT * 2
POINT_DIM = 3

def build_model(labels):
    model = Sequential()
    model.add(keras.Input(shape = (TIMESTEPS, LANDMARK_COUNT * POINT_DIM)))
    model.add(layers.LSTM(64, name="lstm1", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm2", return_sequences=True))
    model.add(layers.LSTM(32, name="lstm3"))
    model.add(layers.Dense(labels, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

TEST_SPLIT = 3
VALIDATION_SPLIT = 1
def load_data(dirname):
    words, words_test, words_val = [], [], []
    dataset, dataset_test, dataset_val = [], [], []

    for sign in Path(dirname).iterdir():
        for i, datafile in enumerate(sign.iterdir()):
            data = holistic.read_datafile(datafile)
            assert data.shape[0] <= TIMESTEPS, "TIMESTEP must be above {}".format(data.shape[0])
            # Zero-pad the data array
            zeros = np.zeros( (TIMESTEPS-data.shape[0],) + data.shape[1:] )
            data = np.concatenate((data, zeros), axis=0)

            data_loc = i % 10 # Should be generated randomly once we have more data
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

def train_model(dirname):
    X, Y, X_test, Y_test, X_val, Y_val = load_data(dirname)
    print("Size of training set = {}, test set = {}, validation set = {}".format(X.shape[0], X_test.shape[0], X_val.shape[0]))
    model = build_model(Y.shape[1])
    history = model.fit(X, Y, epochs=5, batch_size=10, validation_data=(X_val, Y_val))
    score, acc = model.evaluate(X_test, Y_test)

    print(history.history)
    print("score: {} accuracy: {}".format(score, acc))
    return model

# Usage: data_dir [model_file]
if __name__ == "__main__":
    data_dir = argv[1]
    model = train_model(data_dir)

    if len(argv) >= 3:
        model_file = argv[2]
        model.save(model_file)
