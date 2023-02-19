import sys
import tensorflow as tf
from DataPreprocessing.clean_dataset import *
import config
from DataPreprocessing.show_images import *
from DataPreprocessing.data_separation import *

cfg = config.load_config()

sys.path.append('/DataPreprocessing')


def data_preprocessing(data):
    clean_dataset(data, cfg['img_exts'])

    data = tf.keras.utils.image_dataset_from_directory(data)
    data_iterator = data.as_numpy_iterator()

    show_images(4, data_iterator, cfg)

    data = data.map(lambda x, y: (x / 255, y))
    train, test, val = data_separation(data)
