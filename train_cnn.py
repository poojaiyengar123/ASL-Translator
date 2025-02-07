import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import cv2

# dataset credits: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

train_path = r'/Users/poojaraghuram/Documents/VS Code/ASL Translator/train'
test_path = r'/Users/poojaraghuram/Documents/VS Code/ASL Translator/test'

train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = train_path, target_size = (64, 64), class_mode = 'categorical', batch_size = 10, shuffle = True)
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path, target_size = (64, 64), class_mode = 'categorical', batch_size = 10, shuffle = True)

images, labels = next(train_batches)

def plotImages(images_list, title):
    _, axes = plt.subplots(1, len(images_list), figsize = (20, 10))
    axes = axes.flatten()
    for image, axis in zip(images_list, axes):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.axis('off')
    plt.suptitle(title, fontsize = 20)
    plt.tight_layout()
    plt.show()

plotImages(images, title = "Sample Training Images")
print("Image batch shape:", images.shape)
print("Label batch:", labels)

model = Sequential([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (64, 64, 3)),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'valid'),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Flatten(),
    Dense(64, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(29, activation = 'softmax')
])

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 1, min_lr = 0.0001)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 2, verbose = 0, mode = 'auto')

history = model.fit(train_batches, epochs = 10, callbacks = [reduce_lr, early_stop], validation_data = test_batches)

model.save('gestures_model.keras')

images, labels = next(test_batches) 
scores = model.evaluate(images, labels, verbose = 0)
print(f"Loss: {scores[0]:.4f}, Accuracy: {scores[1] * 100:.2f}%")

alphabet_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
alphabet_list.append('del')
alphabet_list.append('nothing')
alphabet_list.append('space')
predictions = model.predict(images, verbose = 0)

print("Predictions on a small set of test data:")
print("")
for index, prediction in enumerate(predictions):
    print(alphabet_list[np.argmax(prediction)], end='   ')

plotImages(images, title = "Predicted Labels on Test Data")

print('\nActual labels: ')
for label in labels:
    print(alphabet_list[np.argmax(label)], end='   ')