import keras
from keras.models import Sequential
from keras import layers

TIMESTEPS = 100 # Should exceed sample-count of a typical video
LANDMARK_COUNT = 42
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

build_model(10)
