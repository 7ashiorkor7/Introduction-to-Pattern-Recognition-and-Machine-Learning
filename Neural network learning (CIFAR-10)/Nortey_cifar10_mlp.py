""" Data: We will use the CIFAR-10 dataset.
Study the neural network examples in the neural networks Jupyter notebook.
Note that with neural networks we prefer to use so called “one-hot encoding”. That means that the
network does not output a single value between 0-9 but there is a separate output for each class.
For network training, the target value is a vector of 10 elements. For that purpose, you need to convert
the class ids to one-hot vectors:
0 → (1 0 0 0 0 0 0 0 0 0)
1 → (0 1 0 0 0 0 0 0 0 0)
2 → (0 0 1 0 0 0 0 0 0 0)
. . .
Write Python code that 1) makes a full connected neural network that take a 32 × 32 Cifar-10 image as
a 3,072 dimensional input. The network produces 10 outputs each representing one Cifar-10 class. The
hidden layer for a simple MLP can, for example, contain five neurons only, in the beginning.
Then, 2) code trains the network with Cifar-10 trainining data. Set a suitable learning rate and number
of epochs. Plot the training loss curve after training to confirm that the network learns.
Play with the parameters, and after you find good parameters, 3) test the model with Cifar-10 test samples.
Finally, print classification accuracy for the training data and test data.
During the lectures Keras (includes TensorFlow) was used to train neural networks, but also PyTorch can
be used.
In Keras a layer of 5 full-connected neurons can be added as
model.add(Dense(5, input_dim=3072, activation=’sigmoid’))
For the last layer, you need to always add a full-connected (dense) layer of 10 sigmoid units. """


import numpy as np
import pickle
import tensorflow as tf
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Load CIFAR-10 data from individual files
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Load training data
train_data = []
train_labels = []
for i in range(1, 6):
    batch_data = unpickle(f'data_batch_{i}')
    train_data.append(batch_data[b'data'])
    train_labels += batch_data[b'labels']


# Combine training data and labels
x_train = np.concatenate(train_data, axis=0)
y_train = np.array(train_labels)

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes=10)

# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0

# Create a neural network model
model = Sequential()
model.add(Dense(5, input_dim=3072, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
#history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
""" Test Accuracy: 31.56%
1563/1563 [==============================] - 5s 3ms/step - loss: 1.7955 - accuracy: 0.3444
Training Accuracy: 34.44% """
#history = model.fit(x_train, y_train, epochs=30, batch_size=96, validation_split=0.2)
""" Test Accuracy: 31.48%
1563/1563 [==============================] - 4s 2ms/step - loss: 1.7501 - accuracy: 0.3701
Training Accuracy: 37.01% """
#history = model.fit(x_train, y_train, epochs=40, batch_size=128, validation_split=0.2)
""" Test Accuracy: 30.77%
1563/1563 [==============================] - 5s 3ms/step - loss: 1.7401 - accuracy: 0.3715
Training Accuracy: 37.15% """
#history = model.fit(x_train, y_train, epochs=50, batch_size=160, validation_split=0.2)
""" Test Accuracy: 29.32%
1563/1563 [==============================] - 6s 4ms/step - loss: 1.7244 - accuracy: 0.3838
Training Accuracy: 38.38% """
#history = model.fit(x_train, y_train, epochs=60, batch_size=192, validation_split=0.2)
""" Test Accuracy: 28.56%
1563/1563 [==============================] - 8s 5ms/step - loss: 1.7349 - accuracy: 0.3727
Training Accuracy: 37.27% """
#history = model.fit(x_train, y_train, epochs=70, batch_size=224, validation_split=0.2)
""" Test Accuracy: 29.09%
1563/1563 [==============================] - 7s 4ms/step - loss: 1.6980 - accuracy: 0.3891
Training Accuracy: 38.91% """
#history = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_split=0.2)
""" Test Accuracy: 26.22%
1563/1563 [==============================] - 7s 4ms/step - loss: 1.9109 - accuracy: 0.2974
Training Accuracy: 29.74%
 """
#history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
""" Test Accuracy: 30.83%
1563/1563 [==============================] - 5s 3ms/step - loss: 1.7590 - accuracy: 0.3719
Training Accuracy: 37.19% """




# Plot the training loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load test data
test_batch = unpickle('test_batch')
X_test = test_batch[b'data']
y_test = test_batch[b'labels']
y_test_onehot = to_categorical(y_test, num_classes=10)


test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')


