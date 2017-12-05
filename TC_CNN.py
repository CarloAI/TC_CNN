# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:37:47 2017

@author: carlo
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#%%
# Take the closest time
def nearest(ts, intensity_df):
    ''' This function compares the datetime "ts" with the datetimes at which 
        intesity is recorded in the dataframe "intensity_df". 
        The closest time is returned.
    '''
    return min(intensity_df['Synoptic Time'], key=lambda d: abs(d-ts))

 # Assign intensity to each image
def assign_intensity(image_dt, intensity_df):
    ''' This function assigns the closest (in time) intensity taken from the 
        dataframe "intensity_df" to "image_dt".
        Then returns the intensity (in m/s) and the category (1-7).
    '''
    Y = round(intensity_df[intensity_df['Synoptic Time']==
                           nearest(image_dt,intensity_df)]['Intensity'].values[0]\
                                                        * 0.514444).astype(int)
    if Y < 18: 
        y = 1
    elif Y >= 18 and Y < 33:
        y = 2
    elif Y >= 33 and Y < 43:
        y = 3
    elif Y >= 43 and Y < 50:
        y = 4
    elif Y >= 50 and Y < 58:
        y = 5
    elif Y >= 58 and Y < 70:
        y = 6
    elif Y >= 70: 
        y = 7
        
    return Y,y

#%%
image_dir = '/scratch/cmc13/Satellite_images/'
TCs = os.listdir(image_dir)

# Reduced images size
rh = 120
rw = 160

# number of rows to exclude
nr = 15

# Image dimensions
h = rh-2*nr
w = rw-2*nr

X = np.zeros((h,w))
Y = np.zeros((1,), dtype=int)
y = np.zeros((1,), dtype=int)
image_datetime = []
for TC in TCs:
    
    print(TC)
    # Local directories for stored images
    TC_dir = image_dir + TC + '/'

    # URL path
    page_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/storm.asp?storm_identifier='+TC
    
    # Retrieve intensity table and save it in a dataframe
    tables = pd.read_html(page_url,header=0)
    intensity_df = tables[1]
    # Convert time column from float to datetime format
    intensity_df['Synoptic Time'] = pd.to_datetime(intensity_df['Synoptic Time'], 
                                                              format='%Y%m%d%H%M')
  
#    i = 0  
    for filename in glob.glob(os.path.join(TC_dir, '*.GIF')):
        image_datetime = datetime.strptime(os.path.basename(filename)[-16:-4], 
                                                                '%Y%m%d%H%M')
        im = Image.open(filename)
        # reduce image size
        reduc_im = im.resize((rw,rh),Image.ANTIALIAS)
        # save image as a numpy array
        im_array = np.array(reduc_im)[nr:-nr,nr:-nr]
        X = np.dstack((X,im_array))
        # assign intensity to the image
        Y = np.append(Y,assign_intensity(image_datetime, intensity_df)[0])
        y = np.append(y,assign_intensity(image_datetime, intensity_df)[1])
#        i += 1
#        if i == 5: break
        

# Reshape the input to take the shape (batch, height, width, channels)
X = np.swapaxes(np.swapaxes(X,0,1),0,2) 
X = X.reshape(X.shape[0], h, w, 1)

# Delete first image (black image - used to initialize the array) and intensity
X = np.delete(X,[0],axis=0)
Y = np.delete(Y,[0],axis=0)
y = np.delete(y,[0],axis=0)

# Change type of input to float32
X = X.astype('float32')


#%%     
# # Split data into train and test sets
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
 
# Add a convolutional layer with 32 filters of size=(3,3)
model.add(Conv2D(32, (6, 6), activation='relu', input_shape=(h,w,1)))
 
# Check shape of output
print(model.output_shape)
 
# Normalize the matrix after a convolution layer so the scale of each dimension
# remains the same (it reduces the training time significantly)
BatchNormalization(axis=-1)
 
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add a dropout layer
model.add(Dropout(0.2))
 
# Add a second convolutional layer with 16 filters of size=(3,3)
model.add(Conv2D(16, (3, 3), activation='relu'))
BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
#%%
# Flatten convolutional layers before passing them as input to the fully 
# connected dense layers
model.add(Flatten())
 
# Add a fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))
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
model.fit(X_train, Y_train, batch_size=32, epochs=5)

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

