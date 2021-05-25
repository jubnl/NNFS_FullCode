from .models import *
import os
import cv2


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    x = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            x.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(x), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    x, y = load_mnist_dataset('train', path)
    x_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return x, y, x_test, y_test


# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
