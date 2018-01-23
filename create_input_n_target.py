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
import time
import random
from datetime import datetime
from PIL import Image
from skimage.measure import block_reduce

#%% USER INPUT
image_dir = '/scratch/cmc13/Satellite_images/'
TCs = os.listdir(image_dir)
TCs.remove('binary_input_n_target')
N = len(TCs)
# Split TCs randomly into training and test set
test_set = random.choices(TCs, k=60)
train_set = [i for i in TCs if i not in test_set]
# Sort test set in same order as TCs
test_set.sort(key=lambda x: TCs.index(x))

# Image size
rh = 480
rw = 640

# Reduce imgae size by a factor of f
f = 1
rh = rh//f
rw = rw//f

# number of rows to exclude
nr = 60

# Image dimensions
h = rh-2*(nr//f)
w = rw-2*(nr//f)

# Number of channels
ch = 3

# Create directory where to save binary data
bin_dir = image_dir + 'binary_input_n_target/'
save_dir = bin_dir + 'temp_' + str(ch) + 'ch_' + str(f) + 'res'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Save text file with names of TCs in the training and test sets
txt_file = open(save_dir+'/training_n_test_sets.txt','w')
txt_file.write('Training set:\n\n')
for TC in train_set:
    txt_file.write(str(train_set.index(TC))+') ')
    txt_file.write(TC)
    txt_file.write('\n')
txt_file.write('\n\nTest set:\n\n')
for TC in test_set:
    txt_file.write(str(test_set.index(TC))+') ')
    txt_file.write(TC)
    txt_file.write('\n')
txt_file.close()


#%% DEFINE FUNCTIONS

def nearest(ts, intensity_df):
    ''' This function compares the datetime "ts" with the datetimes at which 
        intensity is recorded in the dataframe "intensity_df". 
        The closest time is returned.
    '''       
    return min(intensity_df.index, key=lambda d: abs(d-ts))


def assign_intensity(image_dt, intensity_df):
    ''' This function assigns the closest (in time) intensity taken from the 
        dataframe "intensity_df" to "image_dt".
        Then returns the intensity (in m/s) and the category (1-7).
    '''
    try:
        Y = (intensity_df[intensity_df.index==image_dt]['Intensity'][0]* \
                                                          0.514444).astype(int)
    except:
        Y = round(intensity_df[intensity_df.index==nearest(image_dt,intensity_df)]
                                   ['Intensity'].values[0] * 0.514444).astype(int)
        
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


def save_hdf5(TC, dataX, dataY, datay):
    ''' This function saves input and target data for tropical cyclone "TC" 
        into hdf5 files.
    '''
    
    if TC in train_set:
        set_name = 'train'
    else:
        set_name = 'test'
    
    fileX = 'X_' + set_name + '.hdf5'
    fileY = 'Y_' + set_name + '.hdf5'
    filey = 'y_' + set_name + '.hdf5'
    
    if TC==train_set[0] or TC==test_set[0]:
        # Create new hdf5 files for X,Y and y:
        fX = h5py.File(save_dir+'/'+fileX,'w')
        fX.create_dataset('Input_X', data=dataX, maxshape=(None,h,w,ch), chunks=True)
        fX.close()
        
        fY = h5py.File(save_dir+'/'+fileY,'w')
        fY.create_dataset('Target_Y', data=dataY, maxshape=(None,), chunks=True)
        fY.close()
        
        fy = h5py.File(save_dir+'/'+filey,'w')
        fy.create_dataset('Target_y', data=datay, maxshape=(None,), chunks=True)                                             
        fy.close()
        
    else:
        # Append data to the existing hdf5 files:
        with h5py.File(save_dir+'/'+fileX,'a') as fX:
            fX['Input_X'].resize((fX['Input_X'].shape[0] + dataX.shape[0]), axis=0)
            fX['Input_X'][-dataX.shape[0]:] = dataX
        
        with h5py.File(save_dir+'/'+fileY,'a') as fY:
            fY['Target_Y'].resize((fY['Target_Y'].shape[0] + dataY.shape[0]), axis=0)
            fY['Target_Y'][-dataY.shape[0]:] = dataY
            
        with h5py.File(save_dir+'/'+filey,'a') as fy:
            fy['Target_y'].resize((fy['Target_y'].shape[0] + datay.shape[0]), axis=0)
            fy['Target_y'][-datay.shape[0]:] = datay
            
    return


def remove_lat_lon_lines(im_arr):
    ''' This function detects vertical and horizontal lines (solid or dashed) 
        and returns an image array with the lines removed.
    '''
    
    yl, xl , cl = np.shape(im_arr)
    for j in range(len(im_arr[1:-1,0,0])):
        if np.count_nonzero(im_arr[j,:,0]) < 2*xl//3:
            im_arr[j,:,:] = ((im_arr[j-1,:,:].astype('uint16') + \
                                im_arr[j+1,:,:].astype('uint16'))//2).astype('uint8')        
            
    for i in range(len(im_arr[0,1:-1,0])):
        if np.count_nonzero(im_arr[:,i,0]) < 2*yl//3:
            im_arr[:,i,:] = ((im_arr[:,i-1,:].astype('uint16') + \
                                im_arr[:,i+1,:].astype('uint16'))//2).astype('uint8')
    
    return im_arr

#%% MAIN SCRIPT
t1 = time.time()
log_file = open(save_dir+'/processed_TCs.txt','w')
log_file.write('List of processed TCs:\n\n')
for TC in TCs:
    
    print(TC)
    log_file.write(str(TCs.index(TC))+') ')
    log_file.write(TC)
    log_file.write('\n')
    
    # Local directories for stored images
    TC_dir = image_dir + TC + '/'
    
    # URL path
    page_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/storm.asp?storm_identifier='+TC

    # Retrieve intensity table and save it in a dataframe
    try:
        tables = pd.read_html(page_url,header=0)
    except:
        continue

    intensity_df = tables[1]
    # Convert time column from float to datetime format
    intensity_df['Synoptic Time'] = pd.to_datetime(intensity_df['Synoptic Time'], 
                                                              format='%Y%m%d%H%M')
    intensity_df.set_index('Synoptic Time', inplace=True)
    intensity_df = intensity_df.resample('15T').interpolate()

#        i = 0
    X = np.zeros((1,h,w,ch)).astype('uint8')
    Y = np.zeros((1,), dtype=int)
    y = np.zeros((1,), dtype=int)
    for filename in glob.glob(os.path.join(TC_dir, '*.GIF')):
        image_datetime = datetime.strptime(os.path.basename(filename)[-16:-4], 
                                                                '%Y%m%d%H%M')
        im = Image.open(filename)
        rgb_im = im.convert('RGB')
        # Save image as a numpy array and remove lat-lon lines
        im_array = np.array(rgb_im)[nr:-nr,nr:-nr,:]
        remove_lat_lon_lines(im_array)
        # Reduce image size
        if f > 1:
            im_array = block_reduce(im_array, block_size=(f,f,1), func=np.mean)
                                                        
        im_array = np.expand_dims(im_array, axis=0).astype('uint8')
        X = np.concatenate((X,im_array),axis=0)
        # assign intensity to the image
        Y = np.append(Y,assign_intensity(image_datetime, intensity_df)[0])
        y = np.append(y,assign_intensity(image_datetime, intensity_df)[1])                
#            i += 1
#            if i == 10: break

    # Delete first image (black image - used to initialize the array) and intensity    
    X = np.delete(X,[0],axis=0)
    Y = np.delete(Y,[0],axis=0)
    y = np.delete(y,[0],axis=0)
    
    # Save input and target data
    save_hdf5(TC, X, Y, y)

log_file.close()
print(time.time() - t1)

#%% MAIN SCRIPT
#t1 = time.time()    
#
#for TC in TCs:
#    create_input_images(TC)
#
#print(time.time() - t1)








