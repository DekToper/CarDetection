import tensorflow as tf
from DataPreprocessing.data_preprocessing import data_preprocessing
from DeepLearning.train import *


train_data, test_data, val_data = data_preprocessing('DataSet')
model = create_model()
train(model, train_data, val_data, epochs=20)


