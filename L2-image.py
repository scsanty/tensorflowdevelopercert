import tensorflow as tf
from tensorflow import keras

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()