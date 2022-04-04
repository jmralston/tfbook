# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf

# Class which stops further epochs training once a certain training accuracy is met.
# This saves time from redundant training and also helps combat overfitting to the training set.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
# Uses the Tensorflow 'datasets' library of a pre-existing labelled clothing dataset with pre-split training and testing data.
# Based of the original MNIST dataset, it holds a distribution of 70,000 images of 10 classes, with each image
# holding 784 pixels (28x28) and each pixel holding a value from 0 - 255 for a greyscale image.
# Hosts:
# - 60,000 training images, 60,000 training labels.
# - 10,000 testing images, 10,000 testing labels.
mnist = tf.keras.datasets.fashion_mnist
# Stores the training/testing data/labels into relevant variables.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# Training and test pixel values from 0 - 255 are then batch normalised to values from 0 - 1.
# This configuration ensures the training process proceeds smoothly due to the complexity of the problem.
training_images=training_images/255.0
test_images=test_images/255.0
# Defining the neural network model.
model = tf.keras.models.Sequential([
  # 'Flatten' is a method of formatting the input, rather than the neurons.
  # Converts the input from a 2D array making up an image as a 'square', into a 1D array as pure numeric values as a 'line'
  tf.keras.layers.Flatten(),
  # Fully-connected dense layer. Made up of 128 neurons using the 'rectified linear units' activation function.
  # Neurons - The more neurons, the slower the network. Too many neurons = overfitting. Too little = underfitting.
  #         - Choosing the best neuron count for generalisation comes from understanding the problem complexity and
  #           experience with creating neural networks. No clear 'best' neuron count.
  #         - Selection process known as 'Hyperparameter tuning'
  # Activation Function - ReLU (Rectified Linear Units) used to ensure no negative input. (Negative input defaulted
  #                       to '0'. Used to combat vanishing gradient problem and generally resulting in better performance.
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  # Output layer with a neuron for each potential output. Ten classes so ten neurons. Each neuron will have a value
  # which will be the probability that a given input is a given output.
  # Softmax activation function outputs a vector listing the probability distributions of all potential outcomes.
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Compiles the model with given parameters.
# Optimiser - 'Adam'. Extension of stochastic gradient descent optimised for deep learning in computer vision and
#             natural language processing.
# Loss - 'Spare_categorical_crossentropy'. Used for categorical problems where there is more than one output.
# Metrics - 'Accuracy'. Rather than just the loss, we are interested in how often the model guesses the correct labels.
#         - Performs this by comparing the model selected via the output later to the training labels which hold the
#           true values.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fits the model with our specified parameters to the training data. Compares guesses with training labels.
# Performs a maximum of 50 iterations across the network, but may terminate training early due to our 'callback'
# class which ceases training at above 95% accuracy.
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
