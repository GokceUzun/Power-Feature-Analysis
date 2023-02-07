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


'''Save the baseline recording for 3 wildtype and 3 gap animals'''
# S7063, S7064, S7069 = GAP 
#gap_animals = ["S7063", "S7064", "S7069"]  
#for animal in gap_animals:
 #   filenames = [f for f in os.listdir(path_to_folder + animal) if (".dat" in f) & ("Baseline" in f)]
  #  for f in filenames:
   #     convert_dat_to_npy(filename = f, path_to_folder = path_to_folder + animal, path_to_save_folder = path_to_save_folder, save_as_name = animal + "_GAP")

# S7068, S7070, S7071 = WT 
#wt_animals = ["S7068", "S7070", "S7071"]  
#for animal in wt_animals:
 #   filenames = [f for f in os.listdir(path_to_folder + animal) if (".dat" in f) & ("Baseline" in f)]
  #  for f in filenames:
   #     convert_dat_to_npy(filename = f, path_to_folder = path_to_folder + animal, path_to_save_folder = path_to_save_folder, save_as_name = animal + "_WT")

