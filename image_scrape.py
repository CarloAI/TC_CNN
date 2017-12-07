# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:30:39 2017

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

image_dir = '/scratch/cmc13/Satellite_images/'
gif_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/'

def download_file(link,local_dir):
    '''This function downloads the images found at the "link" and saves them in
       the directory "local_dir"
    '''
    
    local_filename = link.strip().split('/')[-1]
    r = requests.get(link, stream=True)
    if str(r) == '<Response [200]>':
        with open(local_dir+local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)
    return

TCs = ['SH0'+str(i)+'2017' if i<10 else 'SH'+str(i)+'2017' for i in np.arange(1,20)]

for TC in TCs:
    TC_id = TC

    # URL paths
    images_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/archive.asp?product=4kmirimg&storm_identifier='+TC_id

    # Create a new folder for each TC. If it already exists, continue with the next TC
    save_dir = image_dir + TC_id + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        continue

    # Write a README file with the name of the TC and the type of satellite image
    req = requests.get(images_url)
    soup = BeautifulSoup(req.text, "lxml")
    title = soup('title')[0].string
    f = open(save_dir+'README.txt','w') 
    f.write(title)
    f.close()

    # Find all objects with extension .GIF and download them
    for anchor in soup.findAll('a', href=True):
       im_url = anchor['href']
       if im_url[-4:] == '.GIF':
           download_file(gif_url+im_url, save_dir)



