import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, RandomZoom, RandomFlip, RandomRotation
from keras.models import load_model, save_model

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",message="2022")

data_dir = 'data' 
imgHeight = 180
imgWidth = 180
            
trainingData = tf.keras.preprocessing.image_dataset_from_directory('data',
                                                           validation_split=0.2,
                                                           seed=14,
                                                           subset="training",
                                                           image_size=(imgWidth,imgHeight),
                                                           batch_size=imgWidth)

validationData =  tf.keras.preprocessing.image_dataset_from_directory('data',
                                                           validation_split=0.2,
                                                           seed=14,
                                                           subset="validation",
                                                           image_size=(imgWidth,imgHeight),
                                                           batch_size=32)
classes = trainingData.class_names
numClasses = len(classes)
dataAugmentation = Sequential()
dataAugmentation.add(RandomFlip("horizontal", input_shape=(imgWidth, imgHeight,3)))
dataAugmentation.add(RandomRotation(0.1))
dataAugmentation.add(RandomZoom(0.1))

model = Sequential()
model.add(dataAugmentation)
model.add(Rescaling(1./255, input_shape=(imgWidth,imgHeight,3)))
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(numClasses, activation='sigmoid'))

model.compile('adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(trainingData,validation_data=validationData, epochs=100)

model.save('model.h5')
# model = load_model('model.h5')

testImage = tf.keras.preprocessing.image.load_img("test6.jpg",
                                                  target_size=(imgWidth,imgHeight))
testImageArray = tf.keras.preprocessing.image.img_to_array(testImage)
testImageArray = tf.expand_dims(testImageArray, 0)
predictionImage = model.predict(testImageArray)
predictionName = classes[np.argmax(predictionImage[0])]
predictionScore = np.max(predictionImage[0])
