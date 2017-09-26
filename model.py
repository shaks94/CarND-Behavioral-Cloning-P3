import csv
import cv2
import numpy as np
import os
import sklearn
import random

from keras.models import Sequential,Model
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Lambda ,Dropout,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del(lines[0])

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imgPaths = []
    imgPaths.extend(center)
    imgPaths.extend(left)
    imgPaths.extend(right)


    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imgPaths, measurements)


def imgageFormation(lines):
    center, left, right , totalMeasurements = [], [], [],[]
    for line in lines:
        centerImagePath = line[0]
        leftImagePath = line[1]
        rightImagePath = line[2]


        centerFileName = centerImagePath.split('/')[-1]
        leftFileName = leftImagePath.split('/')[-1]
        rightFileName = rightImagePath.split('/')[-1]

        center_path = "data/IMG/" + centerFileName
        left_path   = "data/IMG/" + leftFileName
        right_path  = "data/IMG/" + rightFileName


        center.append(center_path)
        left.append(left_path)
        right.append(right_path)
        measurement = float(line[3])
        totalMeasurements.append(measurement)

        return (center,left,right,totalMeasurements)

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def modelFormation():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((45, 15), (0, 0))))
    # model.add(Cropping2D(cropping=((40,15),(0,0))))

    a= model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    print("a here ",a.size)
    # model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation="relu"))
    # model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#
center, left, right, measurement =imgageFormation(lines)
imagePaths, measurements = combineImages(center,left,right,measurement,0.2)

samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32  )


model = modelFormation()

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')

print("history object keys")
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()

