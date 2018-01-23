# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:30:39 2017

@author: carlo
"""

import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

image_dir = '/scratch/cmc13/Satellite_images/'
gif_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/'
year = '2016'

TCs = [ b+'0'+str(i)+year if i<10 else b+str(i)+year 
                            for i in np.arange(1,31)
                            for b in ['AL','EP','CP','WP','IO','SH'] ]

TCs = ['AL042015','AL112015','EP012015','EP022015','EP052015','EP102015',
       'EP122015','EP132015','EP152015','EP192015','EP202015','EP222015',
       'CP032015','WP022015','WP042015','WP062015','WP072015','WP092015',
       'WP112015','WP132015','WP162015','WP172015','WP202015','WP212015',
       'WP222015','WP242015','WP252015','WP272015','WP282015']

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


for TC in TCs:
    
    print(TC)
    # URL paths
    images_url = 'http://rammb.cira.colostate.edu/products/tc_realtime/archive.asp?product=4kmirimg&storm_identifier='+TC
  
    req = requests.get(images_url)
    if str(req) == '<Response [200]>':
        
        # Create a new folder for each TC. If it already exists, continue with the next TC
        save_dir = image_dir + TC + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            continue
        
        # Write a README file with the name of the TC and the type of satellite image
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



