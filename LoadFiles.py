'''Module to save recordings as numpy arrays'''

import sys
import os.path
import numpy as np
from numpy import *

# Variables
number_of_channels = 16
sample_rate = 250.4
sample_datatype = 'int16'
display_decimation = 1

# Depending on where the files are, the paths must be changed. 
path_to_folder = '/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/TAINI files/'
path_to_save_folder = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/'


def parse_dat(filename):
    '''Load a .dat file by interpreting it as int16 and then de-interlacing the 16 channels'''
    
    # Load the raw (1-D) data
    dat_raw = np.fromfile(filename, dtype=sample_datatype)
    
    # Reshape the (2-D) per channel data
    step = number_of_channels * display_decimation
    dat_chans = [dat_raw[c::step] for c in range(number_of_channels)]
    
    # Build the time array
    t = np.arange(len(dat_chans[0]), dtype=float) / sample_rate
    
    return dat_chans, t


# The .dat files are quite large so it would be better to transform them into numpy files to save memory space.
def convert_dat_to_npy(filename, path_to_folder, path_to_save_folder, save_as_name):
    
    os.chdir(path_to_folder)
    
    dat_chans, t = parse_dat(filename)
    
    data_to_save = np.array(dat_chans)
    
    os.chdir(path_to_save_folder)
    
    np.save(save_as_name, data_to_save)

    print('File saved for ' + save_as_name)


# Depending on depending on the animals, the loop can be altered.
animals = ["S7098"] # ["S7098", "S7072"] beware that it takes a lot of time!
for animal in animals:
    filenames = [f for f in os.listdir(path_to_folder + animal) if ".dat" in f]

    # To convert all the files:
    for f in filenames:
        print(f)
        convert_dat_to_npy(filename = f, path_to_folder = path_to_folder + animal, path_to_save_folder = path_to_save_folder + animal, save_as_name = f.replace(".dat", ""))

    # To convert a single file: 
    #convert_dat_to_npy(filename = filenames[0], path_to_folder = path_to_folder + animal, path_to_save_folder = path_to_save_folder + animal, save_as_name = filenames[0].replace(".dat", ""))



