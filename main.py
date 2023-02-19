import tensorflow as tf
from DataPreprocessing.data_preprocessing import data_preprocessing

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus[0])
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_preprocessing('DataSet')



