#build and train the model
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from prepare_data import x_train, y_train


def build_model() -> tf.keras.models.Sequential:
    """
    Builds and compiles an LSTM-based neural network model.
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(units=50, activation = 'relu', return_sequences = True, input_shape =(100, 1)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=60, activation = 'relu', return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(units=80, activation = 'relu', return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.LSTM(units=120, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer="adam", loss='mean_squared_error')

    return model

def training_model(model: tf.keras.models.Sequential, x_train: np.ndarray, y_train: np.ndarray) -> None:

    model.fit(x_train, y_train, epochs = 50)

if __name__ == "__main__":
    model = build_model()

    training_model(model, x_train, y_train)

    model.save('predicting_model.keras')
    print("Model saved")