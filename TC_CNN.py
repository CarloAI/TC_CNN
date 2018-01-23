#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:37:47 2017

@author: carlo
"""
#%% IMPORT MODULES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.backend import tensorflow_backend as K

#%% DEFINE KERAS CALLBACKS
logdir = '/scratch/cmc13/git_projects/TC_CNN/logs'
summary_writer = tf.summary.FileWriter(logdir)

callbacks_list = [ EarlyStopping(monitor='accuracy', 
                                 patience=1),
                   ModelCheckpoint(filepath='my_model.h5',
                                   monitor='val_loss',
                                   save_best_only=True),
                   TensorBoard(log_dir=logdir, 
                               histogram_freq=1,
                               write_graph=True, 
                               write_images=True) ]


#%% DEFINE CONSTANTS
load_dir = '/scratch/cmc13/Satellite_images/binary_input_n_target/3ch_1res/'

number_of_classes = 7

fX = h5py.File(load_dir+'X_train.hdf5','r')
X = fX['Input_X'][0,:,:,:]
fX.close()

# Retrieve dimensions (height, width and number of channels)
h,w,c = np.shape(X)


#%% DEFINE FUNCTIONS
def variable_summaries(var):
    '''Attach a lot of summaries to a Tensor (for TensorBoard visualization).'''
    
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#%% DEFINE MODEL
# Use a simple model - a linear stack of layers
model = Sequential()
 
# Add a convolutional layer with 48 filters of size=(5,5)
model.add(Conv2D(48, (5, 5), activation='relu', input_shape=(h,w,c)))
 
# Check shape of output
print(model.output_shape)
 
# Normalize the matrix after a convolution layer so the scale of each dimension
# remains the same (it reduces the training time significantly)
BatchNormalization(axis=-1)
 
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add a dropout layer
model.add(Dropout(0.2))
 
# Add a second convolutional layer with 32 filters of size=(3,3)
model.add(Conv2D(32, (3, 3), activation='relu'))
BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
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
 
# Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
     
#%%
t1 = time.time()

#with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as sess:
#    
#    K.set_session(sess)

# Load test data
fX_test = h5py.File(load_dir+'X_test.hdf5','r')
X_test = fX_test['Input_X'][:,:,:,:]
fX_test.close()

fy_test = h5py.File(load_dir+'y_test.hdf5','r')
y_test = fy_test['Target_y'][:]
fy_test.close()

# Rescale so that image pixel lies in the interval [0,1] instead of [0,255]
X_test = X_test.astype('float32')
X_test/=255

# One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, number_of_classes)

nb_epoch = 1
# Number of image per batch
N = 3072

for e in range(nb_epoch):
    print('epoch %d' % e)
    
    for i in range(1,21):
        
        # Load data
        fX = h5py.File(load_dir+'X_train.hdf5','r')
        X = fX['Input_X'][(i-1)*N:i*N,:,:,:]
        fX.close()

        fY = h5py.File(load_dir+'Y_train.hdf5','r')
        Y = fY['Target_Y'][(i-1)*N:i*N]
        fY.close()
        
        fy = h5py.File(load_dir+'y_train.hdf5','r')
        y = fy['Target_y'][(i-1)*N:i*N]
        fy.close()
        
        # Rescale so that image pixel lies in the interval [0,1] instead of [0,255]
        X_train = X.astype('float32')
        X_train/=255
        
        # One-hot encode the labels
        Y_train = np_utils.to_categorical(y, number_of_classes)
        
        # Train the model
        model.fit(X_train, Y_train, 
                  batch_size=128, 
                  epochs=1, 
                  verbose=1, 
                  callbacks=callbacks_list, 
                  validation_data=(X_test,Y_test))
    
     
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

print(time.time() - t1)


    
