import tensorflow as tf
import numpy as np

tf.get_logger().setLevel('INFO')

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
y = np.array([-2, -1, 0, 1, 2, 3, 4, 5])

model.fit(x, y, epochs=500)

print(model.predict([10]))
