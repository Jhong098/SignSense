import random
from collections import Counter
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

TIMESTEPS = 30
TRAIN_TIMESTEPS = TIMESTEPS * 2
POINT_DIM = 3


# hand_model: (0.0) 89.5% overfit?
# hand_model2: (0.15) 90% overfit?
# hand_model3: (0.3) 86%
def build_model(labels, frame_dim, dropout=0.3):
    model = Sequential()
    model.add(keras.Input(shape = (TIMESTEPS, frame_dim)))
    model.add(layers.LSTM(128, name="lstm1", dropout=dropout, return_sequences=True))
    model.add(layers.LSTM(64, name="lstm2", dropout=dropout))
    model.add(layers.Dense(labels, activation="softmax"))
    adam = Adam(lr = 0.0002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def get_labels(dirname):
    holds = [sign.name for sign in Path(dirname, 'holds_data').iterdir()]
    nonholds = [sign.name for sign in Path(dirname, 'nonholds_data').iterdir()]
    return (sorted(holds + nonholds), set(holds))

def load_data(dirname):
    for sign in Path(dirname, 'holds_data').iterdir():
        for datafile in sign.iterdir():
            yield (holistic.read_datafile(datafile), sign.name)
    for sign in Path(dirname, 'nonholds_data').iterdir():
        for datafile in sign.iterdir():
            yield (holistic.read_datafile(datafile), sign.name)

def count_labels(data_iter, count):
    for data, sign in iter(data_iter):
        count[sign] += 1
        yield (data, sign)
def count_zeros(data_iter, count):
    for data, sign in iter(data_iter):
        if sign == 0:
            count[None] += 1
        yield (data, sign)

def label_signs(data_iter, labels):
    labels = {l:i+1 for i, l in enumerate(labels)}
    for data, sign in iter(data_iter):
        yield (data, labels[sign])

def onehot_labelled_signs(data_iter, num_labels):
    for data, sign in iter(data_iter):
        onehot = np.zeros(num_labels + 1)
        onehot[sign] = 1
        yield (data, onehot)

def generate_windows(data, sign, start, end, stride):
    while start + TRAIN_TIMESTEPS < min(end, data.shape[0]):
        yield (data[start:start+TRAIN_TIMESTEPS], sign)
        start += stride

def truncate_and_divide_data(data_iter, holds, trim):
    for data, sign in iter(data_iter):
        datalen = data.shape[0]
        if datalen > TRAIN_TIMESTEPS:
            if sign in holds:
                yield from generate_windows(data, sign, 30, datalen - 30, 30)
            else:
                # If not even one window can be generated, then just take the back end of the data
                if datalen - trim < TRAIN_TIMESTEPS:
                    yield (data[datalen - TRAIN_TIMESTEPS:], sign)
                else:
                    yield from generate_windows(data, sign, trim, datalen, 2)
        else:
            yield (data, sign)

def extend_data(data_iter):
    for data, sign in iter(data_iter):
        if data.shape[0] < TRAIN_TIMESTEPS:
            #print("{} {}".format(sign, data.shape[0]))
            zeros = np.zeros( (TRAIN_TIMESTEPS-data.shape[0],) + data.shape[1:] )
            data = np.concatenate((data, zeros), axis=0)
        yield (data, sign)

def add_gesture_zero(dataset, num_labels):
    len_per_label = int(len(dataset) // num_labels)
    zeros = [(np.zeros((TRAIN_TIMESTEPS, dataset[0][0].shape[1])), 0)] * len_per_label
    return dataset + zeros

def add_none_data(data_iter, dirname):
    for data, sign in iter(data_iter):
        yield (data, sign)
    for datafile in Path(dirname, 'None').iterdir():
        data = holistic.read_datafile(datafile)
        yield from generate_windows(data, 0, 0, data.shape[0], TRAIN_TIMESTEPS)

def speed_up_data(data_iter):
    for data, sign in iter(data_iter):
        data = data[::2]
        yield (data, sign)


TEST_SPLIT = 3
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
    labels, holds = get_labels(dirname)
    print(labels)
    count = Counter()

    data_iter = load_data(dirname)
    data_iter = extend_data(data_iter)
    data_iter = truncate_and_divide_data(data_iter, holds, 0)
    data_iter = count_labels(data_iter, count)
    data_iter = label_signs(data_iter, labels)
    data_iter = add_gesture_zero(list(data_iter), len(labels))
    data_iter = add_none_data(data_iter, dirname) # Comment out this line to exclude "None" data when training
    data_iter = count_zeros(data_iter, count)
    data_iter = onehot_labelled_signs(data_iter, len(labels))
    data_iter = speed_up_data(data_iter)
    data, data_test = split_data(data_iter)
    random.shuffle(data)

    dataset = [d for d, w in data]
    words = [w for d, w in data]
    dataset_test = [d for d, w in data_test]
    words_test = [w for d, w in data_test]

    X = np.array(dataset)
    X_test = np.array(dataset_test)
    Y = np.array(words)
    Y_test = np.array(words_test)

    print(count)

    return X, Y, X_test, Y_test

def plot_data(history, name1, name2):
    plt.figure()
    plt.plot(history[name1], label=name1)
    plt.plot(history[name2], label=name2)
    plt.legend()

def train_model(dirname, epochs=25, batch_size=128, val_split=0.25):
    X, Y, X_test, Y_test = load_and_process_data(dirname)
    print("Size of training set = {}, test set = {}".format(X.shape[0], X_test.shape[0]))
    model = build_model(Y.shape[1], X.shape[2])
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=val_split)
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)

    print("score: {} accuracy: {}".format(score, acc))
    plot_data(history.history, 'loss', 'val_loss')
    plot_data(history.history, 'accuracy', 'val_accuracy')
    plt.show()
    return model

def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Usage: data_dir [model_file]
if __name__ == "__main__":
    data_dir = argv[1]
    init_gpu()

    model = train_model(data_dir)

    if len(argv) >= 3:
        model_file = argv[2]
        model.save(model_file)
