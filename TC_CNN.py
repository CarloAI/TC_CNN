# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:37:47 2017

@author: carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.backend import tensorflow_backend as K

#%%
# Load data from hdf5 files
fX = h5py.File('X_92TC.hdf5','r')
X = fX['Input_X'][:-250,:,:,:]
fX.close()

fY = h5py.File('Y_92TC.hdf5','r')
Y = fY['Target_Y'][:-250]
fY.close()

fy = h5py.File('y_92TC.hdf5','r')
y = fy['Target_y'][:-250]
fy.close()

#%% Retrieve dimensions

# Number of input images
m = len(X[:,0,0,0])

# Height of images
h = len(X[0,:,0,0])

# Width of images
w = len(X[0,0,:,0])

# Number of channels
c = len(X[0,0,0,:])

#%%     
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Rescale so that image pixel lies in the interval [0,1] instead of [0,255]
X_train/=255
X_test/=255
 
# One-hot encode the labels
number_of_classes = 7
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
 
#%%
# Use a simple model - a linear stack of layers
model = Sequential()
 
# Add a convolutional layer with 48 filters of size=(6,6)
model.add(Conv2D(48, (5, 5), activation='relu', input_shape=(h,w,1)))
 
# Check shape of output
print(model.output_shape)
 
# Normalize the matrix after a convolution layer so the scale of each dimension
# remains the same (it reduces the training time significantly)
BatchNormalization(axis=-1)
 
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add a dropout layer
model.add(Dropout(0.2))
 
# Add a second convolutional layer with 32 filters of size=(4,4)
model.add(Conv2D(32, (3, 3), activation='relu'))
BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
#%%
# Flatten convolutional layers before passing them as input to the fully 
# connected dense layers
model.add(Flatten())
 
# Add a fully connected layer with 128 neurons
model.add(Dense(256, activation='relu'))
BatchNormalization(axis=-1)
model.add(Dropout(0.2))

# Add a second fully connected layer with 64 neurons
model.add(Dense(64, activation='relu'))
BatchNormalization(axis=-1)
model.add(Dropout(0.2))
 
# Add an output layer with 10 neurons, one for each class. Use a softmax 
# activation function to output probability-like predictions for each class
model.add(Dense(number_of_classes, activation='softmax'))
 
# Print model summary
model.summary()
 
#%%
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
    K.set_session(sess)

    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
     
    ## Apply Data Augmentation to training and test sets
    #gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
    #                         height_shift_range=0.08, zoom_range=0.08)
    #test_gen = ImageDataGenerator()
    #
    ## Generate batches of augmented data
    #train_generator = gen.flow(X_train, Y_train, batch_size=64)
    #test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
     
    # Train the model
    model.fit(X_train, Y_train, batch_size=128, epochs=5)

    # Evaluate the model
    score = model.evaluate(X_test, Y_test)
    print('\nTest accuracy: ', score[1])
    
    # Predict classes of test data
    predictions = model.predict_classes(X_test)
    
    # Number of mislabelled images
    print('\nNumber of mislabelled images: ',np.count_nonzero(y_test-predictions),
          '\nTotal number of images      : ',len(y_test))
    
    # Save actual category and predictions in a csv file
    output_df = pd.DataFrame({'Actual': y_test, 'Prediction': predictions})
    output_df.to_csv('./output_TC_CNN.csv', index=False)
    
    acc = 'Test accuracy: ' + str(score[1])
    # Save accuracy in a text file
    f = open('Accuracy.txt','w')
    f.write(acc)
    f.close()
    
    #%%
    cat = {}
    mislabel = {}
    no_images = {}
    no_mislab = {}
    perc_mislab = {}
    for i in range(number_of_classes):
        cat[i] = predictions[np.where(y_test==i)]
        mislabel[i] = np.nonzero(cat[i]-i)[0]
        no_images[i] = len(cat[i])
        no_mislab[i] = len(mislabel[i])
        perc_mislab[i] = len(mislabel[i]) / len(cat[i])
    
    
    print('\nNumber of images per category:')
    for i in range(number_of_classes):
        print('Category '+str(i)+':',no_images[i],'images, ',no_mislab[i],'mislabelled')
    
    print('\nPercentage of mislabelled images per category:')
    for i in range(number_of_classes):
        print('Category '+str(i)+':',round(perc_mislab[i]*100,1),'%')
    
#%%
#fX = h5py.File('X_92TC.hdf5','r')
#X_val = fX['Input_X'][-250:,:,:,:]
#fX.close()
#
#fY = h5py.File('Y_92TC.hdf5','r')
#Y_val = fY['Target_Y'][-250:]
#fY.close()
#
#fy = h5py.File('y_92TC.hdf5','r')
#y_val = fy['Target_y'][-250:]
#fy.close()
#
#X_val = X_val.astype('float32')
#X_val/=255
# 
## One-hot encode the labels
#number_of_classes = 7
#Y_valhot = np_utils.to_categorical(y_val, number_of_classes)
#
#predictions_val = model.predict_classes(X_val)
#
## Number of mislabelled images
#print('\nNumber of mislabelled images: ',np.count_nonzero(y_val-predictions_val),
#      '\nTotal number of images      : ',len(y_val))    
    
    
    
    