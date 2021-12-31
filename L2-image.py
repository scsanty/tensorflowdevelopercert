import tensorflow as tf
from tensorflow import keras

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()


class myCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if (logs.get('loss') < 0.3):
            print('Loss is effectively less')
            self.model.stop_training = True

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.fit(train_x, train_y, epochs=10, callbacks=[myCallback()])
print("Accuracy: ", model.evaluate(test_x, test_y))
