
# From https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf
# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()
predictions

# The tf.nn.softmax function converts these logits to "probabilities" for each class
tf.nn.softmax(predictions).numpy()

# This untrained model gives probabilities close to random (1/10 for each class)
# so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts the model parameters to minimize the loss: 
model.fit(x_train, y_train, epochs=10)

# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set"
model.evaluate(x_test,  y_test, verbose=2)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
