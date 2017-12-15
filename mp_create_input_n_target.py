#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:14:43 2017

@author: cmc13
"""

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
import time
from multiprocessing import Pool

#%%
# Take the closest time
def nearest(ts, intensity_df):
    ''' This function compares the datetime "ts" with the datetimes at which 
        intesity is recorded in the dataframe "intensity_df". 
        The closest time is returned.
    '''       
    return min(intensity_df.index, key=lambda d: abs(d-ts))

 # Assign intensity to each image
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

#%%
t1 = time.time()    

image_dir = '/scratch/cmc13/Satellite_images/'
TCs = os.listdir(image_dir)

# Image size
rh = 480
rw = 640

# Reduce imgae size by a factor of f
f = 1
rh = int(rh/f)
rw = int(rw/f)

# number of rows to exclude
nr = int(60/f)

# Image dimensions
h = rh-2*nr
w = rw-2*nr

def create_input_images(TC):
    print(TC)
    
    # Local directories for stored images
    TC_dir = image_dir + TC + '/'
    
    # URL path
    page_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/storm.asp?storm_identifier='+TC

    # Retrieve intensity table and save it in a dataframe
    try:
        tables = pd.read_html(page_url,header=0)
    except:
        return None

    intensity_df = tables[1]
    # Convert time column from float to datetime format
    intensity_df['Synoptic Time'] = pd.to_datetime(intensity_df['Synoptic Time'], 
                                                              format='%Y%m%d%H%M')
    intensity_df.set_index('Synoptic Time', inplace=True)
    intensity_df = intensity_df.resample('15T').sum()
    intensity_df = intensity_df.interpolate()

#    i = 0
    X = np.zeros((1,h,w,3)).astype('float32')
    Y = np.zeros((1,), dtype=int)
    y = np.zeros((1,), dtype=int)
    for filename in glob.glob(os.path.join(TC_dir, '*.GIF')):
        image_datetime = datetime.strptime(os.path.basename(filename)[-16:-4], 
                                                                '%Y%m%d%H%M')
        im = Image.open(filename)
        rgb_im = im.convert('RGB')
        # reduce image size
        reduc_im = rgb_im.resize((rw,rh),Image.ANTIALIAS)
        # save image as a numpy array
        im_array = np.array(reduc_im)[nr:-nr,nr:-nr,:]
        im_array = np.expand_dims(im_array, axis=0).astype('float32')
        X = np.concatenate((X,im_array),axis=0)
        # assign intensity to the image
        Y = np.append(Y,assign_intensity(image_datetime, intensity_df)[0])
        y = np.append(y,assign_intensity(image_datetime, intensity_df)[1])
                
#        i += 1
#        if i == 10: break
    
    X = np.delete(X,[0],axis=0)
    Y = np.delete(Y,[0],axis=0)
    y = np.delete(y,[0],axis=0)
    
    return X, Y, y

if __name__ == '__main__':
    pool = Pool(processes=4)
    # 'result' is a list of len(TCs) elements. Each element is a 3-element tuple.
    # First tuple is X, second is Y and third is y.
    output = pool.map(create_input_images, TCs)
    
Xl = [output[i][0] for i in np.arange(len(output)) if output[i] != None]
X = np.concatenate(Xl)
Yl = [output[i][1] for i in np.arange(len(output)) if output[i] != None]
Y = np.concatenate(Yl)
yl = [output[i][2] for i in np.arange(len(output)) if output[i] != None]
y = np.concatenate(yl)

print(time.time() - t1)

#%%
# Save data into hdf5 files
fX = h5py.File('X_186_TC.hdf5','w')
fX.create_dataset('Input_X', data=X)
fX.close()

fY = h5py.File('Y_186_TC.hdf5','w')
fY.create_dataset('Target_Y', data=Y)
fY.close()

fy = h5py.File('y_186_TC.hdf5','w')
fy.create_dataset('Target_y', data=y)
fy.close()








