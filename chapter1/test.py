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
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Creating the model:
# Sequential is used to define the layers of your neural network
# Inside Sequential you specify what each layer looks like (l0: layer 0) (using the keras.layers API)
# The layer type is Dense, meaning a set of fully connected nuerons
# units=1 indicates one layer
# the first layer requires a shape of the input data, input_shape=[1] meaning one value for X
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])

# Defining how the model will train:
# optimizer='sgd' uses stochastic gradient descent, a mathematical function that can provide a guess using
# target values, the previous guess value, and the results of the calculated loss on that guess
# In this case the calculated loss will use MSE (Mean Squared Error):
# 1. Subtract the predicated value from the actual value
# 2. Take the absolute value of the error and square it
# 3. Sum all of your error values and find the mean
model.compile(optimizer='sgd', loss='mean_squared_error')

# Using the numpy module you can generate a formatted array to use with TensorFlow
# In this case both arrays have 6 values that are of type float
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Now we train the model using the fit function, passing the source value xs, target value ys, and number of guesses epochs=500
# Basically saying guess the relationship between X and Y and try it 500 times
model.fit(xs, ys, epochs=500)

# Using the trained model you can call the predict function to output the best guess Y value
print(model.predict([10.0]))

# Once the model has been trained, get_weights can be used on each layer to see what the model is using for the relationship
# In this particular example there is only one nueron, so you will only get a single weights value comprising of:
# Y = (Weight)X + (Bias)
# Weight = how much influence the input feature will have on the output
# Bias = refers to the error of making an assumption for data that may be more complicated, in this case if we were trying to 
# use this model to solve a real world problem, there may be additional features that could be included
# The goal for these values: Y = (2)X + (-1)
print("Here is what I learned: {}".format(l0.get_weights()))
