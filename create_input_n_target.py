#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:01:47 2017

@author: cmc13
"""

import pandas as pd
import numpy as np
import os
import glob
import h5py
from datetime import datetime
from PIL import Image


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
        y = 0
    elif Y >= 18 and Y < 33:
        y = 1
    elif Y >= 33 and Y < 43:
        y = 2
    elif Y >= 43 and Y < 50:
        y = 3
    elif Y >= 50 and Y < 58:
        y = 4
    elif Y >= 58 and Y < 70:
        y = 5
    elif Y >= 70: 
        y = 6
        
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
# Save data into hdf5 files
fX = h5py.File('X_92TC.hdf5','w')
fX.create_dataset('Input_X', data=X)
fX.close()

fY = h5py.File('Y_92TC.hdf5','w')
fY.create_dataset('Target_Y', data=Y)
fY.close()

fy = h5py.File('y_92TC.hdf5','w')
fy.create_dataset('Target_y', data=y)
fy.close()








