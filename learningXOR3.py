import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(2, activation="sigmoid", name="hidden_layer"))
model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))
model.compile(loss = "mse", optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 10))
model.fit(x,y, epochs = 1000)
print(model.predict(x))

# print(model.summary())
# keras.utils.plot_model(model, to_file='model.png', show_shapes = True)
