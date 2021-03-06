# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:37:47 2017

@author: carlo
"""

import pandas as pd
import numpy as np
import requests
import os
import glob
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


TC_id = 'AL092017'

save_dir = 'C:/Users/carlo/Desktop/Satellite_images/'+TC_id+'/'

images_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/archive.asp?product=4kmirimg&storm_identifier='+TC_id
page_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/storm.asp?storm_identifier='+TC_id
images_path = 'http://rammb.cira.colostate.edu/products/tc_realtime/'+TC_id+'/'

req = requests.get(images_url)
soup = BeautifulSoup(req.text, "lxml")
title = soup('title')[0].string

f = open(save_dir+'README.txt','w') 
f.write(title)
f.close()

def download_file(link,local_dir):
    
    local_filename = link.split('/')[-1]
    r = requests.get(link, stream=True)
    if str(r) == '<Response [200]>':
        with open(local_dir+local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)
    return

for anchor in soup.findAll('a', href=True):
    im_url = anchor['href']
    if im_url[-4:] == '.GIF':
        download_file(images_path+im_url, save_dir)
        


#%%
        
tables = pd.read_html(page_url,header=0)
intensity_table = tables[1]
# Convert time column from float to datetime format
intensity_table['Synoptic Time'] = pd.to_datetime(intensity_table['Synoptic Time'], 
                                                  format='%Y%m%d%H%M')

#%%
# number of rows to exclude
nr = 60

# Image dimensions
h = 480-2*nr
w = 640-2*nr

image_datetime = []
X = np.zeros((h,w))
#i = 0
for filename in glob.glob(os.path.join(save_dir, '*.GIF')):
    image_datetime.append(datetime.strptime(os.path.basename(filename)[-16:-4], 
                                            '%Y%m%d%H%M'))
    im = Image.open(filename)
    im_array = np.array(im)[nr:-nr,nr:-nr]
    X = np.dstack((X,im_array))
#    i += 1
#    if i == 5: break

# ReReshape the input to take the shape (batch, height, width, channels)
X = np.swapaxes(np.swapaxes(X,0,1),0,2) 
X = X.reshape(X.shape[0], h, w, 1)

# Delete first image (black image - used to initialize the array)
X = np.delete(X,[0],axis=0)

X = X.astype('float32')
    
# Take closest tim
def nearest(ts): 
    return min(intensity_table['Synoptic Time'], key=lambda d: abs(d-ts))


# Assign intensity to each image
Y = np.zeros([len(image_datetime),], dtype=int)
y = np.zeros([len(image_datetime),], dtype=int)
for i in np.arange(len(image_datetime)):
    Y[i] = round(intensity_table[intensity_table['Synoptic Time']==
                                      nearest(image_datetime[i])]['Intensity']\
                                                        * 0.514444).astype(int)
    if Y[i] < 18: 
        y[i] = 1
    elif Y[i] >= 18 and Y[i] < 33:
        y[i] = 2
    elif Y[i] >= 33 and Y[i] < 43:
        y[i] = 3
    elif Y[i] >= 43 and Y[i] < 50:
        y[i] = 4
    elif Y[i] >= 50 and Y[i] < 58:
        y[i] = 5
    elif Y[i] >= 58 and Y[i] < 70:
        y[i] = 6
    elif Y[i] >= 70: 
        y[i] = 7
    
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

# Add a convolutional layer with 32 filters of size=(3,3)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(h,w,1)))

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

