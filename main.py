from matplotlib.image import imread
from PIL import Image
import tensorflow as tf
from DataPreprocessing.data_preprocessing import data_preprocessing
from DataPreprocessing.show_images import show_predict
from DeepLearning.train import *


train_data, test_data, val_data = data_preprocessing('DataSet')
model = create_model()
model = train(model, train_data, val_data, epochs=1)
yhat = predict(model, Image.open('img.png'))
show_predict(imread('img.png'), yhat)
yhat = predict(model, imread('img_1.png'))
show_predict(imread('img_1.png'), yhat)