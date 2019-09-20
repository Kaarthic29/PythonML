from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Fashion MNIST Dataset - contains 70,000 grayscale images in 10 categories. images at a low resolution (28 by 28 pixels)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Loading thw dataset will return 4 NumPy arrays - train images, train labels, test images, test labels.

#labels are an array of integers, ranging from 0 to 9 which represent class of clothing.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(class_names)

#Exploring the data - 60000 images each with 28X28 pixels
trainingimages_shape = train_images.shape
print(trainingimages_shape)

#60000 labels in training set
trainingimages_length = len(train_labels)
print(trainingimages_length)

#each label is an integer between 0 and 9
print(train_labels)

#10000 test images
testimages_shape = test_images.shape
print(testimages_shape)

#10000 test image labels
testimages_length = len(test_labels)
print(testimages_length)

#Data Preprocessing

#Inspecting first image in training set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Scaling the pixel values from 0 to 1 by dividing each by 255 before feeding to neural network

train_images = train_images / 255.0

test_images = test_images / 255.0

#Display first 25 images from training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Building the model: Setup layers & Compile the model

#Setting up layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compile the model - to add more settings
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(train_images, train_labels, epochs=5)

#Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#Make Predictions
predictions = model.predict(test_images)
print(predictions[0])

MaxConfidenceValue = np.argmax(predictions[0])
print(MaxConfidenceValue)

print(test_labels[0])

#Putting Graph
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()



