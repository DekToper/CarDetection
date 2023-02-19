import os

from DataPreprocessing.data_preprocessing import data_preprocessing
from keras import callbacks
from keras.applications.densenet import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D
import tensorflow as tf
from config import load_config
from matplotlib import pyplot as plt

cfg = load_config()


def create_model():
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(256, 256, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model


create_model()


def check_log_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The log directory is created.")


def train(model, train_data, val_data, epochs=20):
    path = cfg['logs_dir']
    check_log_dir(path)
    tensorboard_callback = callbacks.TensorBoard(log_dir=path)
    hist = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

