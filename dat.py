import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Save as .npz file
np.savez('fashion_mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
