import numpy as np
import nnfs
import cv2
from models import *
from func import *

nnfs.init()

# Read an image
# image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
# image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
# image_data = 255 - image_data

# Reshape and scale pixel data
# image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
# model = Model.load('fashion_mnist.model')

# Predict on the image
# confidences = model.predict(image_data)

# Get prediction instead of confidence levels
# predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
# prediction = fashion_mnist_labels[predictions[0]]

# print(prediction)

x, y, x_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(x.shape[0]))
np.random.shuffle(keys)
x = x[keys]
y = y[keys]
# Scale and reshape samples
x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5
x_test = (x_test.reshape(x_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(x.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

# Set loss and accuracy objects
# We do not set optimizer object this time - there's no need to do it
# as we won't train the model
model.set(loss=LossCategoricalCrossentropy(), accuracy=AccuracyCategorical())
# Finalize the model
model.finalize()
# Set model with parameters instead of training it
model.load_parameters('fashion_mnist.parms')
# Evaluate the model
model.evaluate(x_test, y_test)
