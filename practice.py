import csv
import cv2
import numpy as np
import os
import sklearn


from keras.models import Sequential,Model
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Lambda ,Dropout,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


##
dir = os.listdir('data/cardata/')
inputing=[]
for input in dir[1:]:
    inputing.extend([input])

#--------------------Reading CSV file Data
def csvInputs(path):
    lines = []
    csv_Input_path = "data/carData/" + path + "/driving_log.csv"
    with open(csv_Input_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    del(lines[0])
    print(lines[0])
    return lines
#------------------ end of reading csv

all_images = []
measurements = []
correction = 0.2  # this is a parameter to tune
#-------------------- Reading images and measurement data
def img_measurement(lines,path):
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            # current_path = 'data/IMG/' + filename
            current_path = "data/carData/"+path+"/IMG/" + filename

            image = cv2.imread(current_path)

            all_images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
            measurements.append(measurement+correction)
            measurements.append(measurement-correction)
    print("done")
#------------------- end of images and mesurement reading

aug_images = []
aug_measurements = []


# --- augmentation of data and flipping the content
def augmentdataion():
    for image, measurement in zip(all_images, measurements):
        aug_images.append(image)
        aug_measurements.append(measurement)
        aug_images.append(cv2.flip(image, 1))

        aug_measurements.append(measurement * -1.0)


##-------------end of augmentation


line2=csvInputs(inputing[0])
img_measurement(line2,inputing[0])

line2=csvInputs(inputing[1])
img_measurement(line2,inputing[1])

line3=csvInputs(inputing[2])
img_measurement(line3,inputing[2])

augmentdataion()

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)
# yield sklearn.utils.shuffle(X_train, y_train)
# generators()

#------- Using keras


model = Sequential()
model.add(Lambda(lambda x:x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
# model.add(Cropping2D(cropping=((40,15),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation="relu"))
# model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation="relu"))
# model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Convolution2D(48,5,5,subsample = (2,2),activation="relu"))
# model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64,3,3,activation="relu"))
# model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3,verbose = 1)

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

#
exit()

