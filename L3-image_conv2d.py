import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
train_x = train_x/255.0
test_x	= test_x/255.0

train_x = train_x.reshape(60000, 28, 28, 1)
test_x	= test_x.reshape(10000, 28, 28, 1)



class myCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if (logs.get('loss') < 0.1):
			self.model.stop_training = True
			print('Loss is effectively less')

model = keras.models.Sequential(
	[
		keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
		keras.layers.MaxPool2D(2, 2),
		keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
		keras.layers.MaxPool2D(2, 2),
		keras.layers.Flatten(),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)
	]
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, callbacks=[myCallback()])
scores = model.evaluate(test_x, test_y, return_dict=True)
print("Test Accuracy: ", round(scores['accuracy']*100, 2), '%')
