# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 01:30:00 2020

@author: AMD E2
"""

# digit recognition project-2

#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Activation,Dropout
from keras.utils import normalize, to_categorical
from keras.optimizers import Adam

#dataset
X_train=pd.read_csv("dataset/digit_train.csv")
X_test=pd.read_csv("dataset/digit_test.csv")
y_train=X_train['target']
y_test=X_test['target']
X_train=X_train.iloc[:,0:-1]
X_test=X_test.iloc[:,0:-1]
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

# initialising the CNN
classifier=Sequential()
# 1st Convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=(2,2), input_shape=(32,32,1),activation='relu'))
#1st maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
# 2nd Convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=(2,2), activation='relu'))
#2nd maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
#final layer
classifier.add(Dense(10, activation='softmax'))
#compile 
opt=Adam(lr=0.001)
classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#   Fitting CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(
        'train',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(32,32),
        batch_size=32,
        class_mode='categorical')
classifier.fit(
        train_set,
        steps_per_epoch=42000,
        epochs=20,
        validation_data=test_set,
        validation_steps=28000)
