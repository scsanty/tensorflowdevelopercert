import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
train_x = train_x/255.0
test_x = test_x/255.0


class myCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if (logs.get('loss') < 0.3):
			self.model.stop_training = True
			print('Loss is effectively less')

model = keras.models.Sequential(
	[
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(1024, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)
	]
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, callbacks=[myCallback()])
scores = model.evaluate(test_x, test_y, return_dict=True)
scores
print("Test Accuracy: ", round(scores['accuracy']*100, 2), '%')
