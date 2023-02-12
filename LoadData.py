'''Module to load the data either in .dat or .npy format'''

import sys
import os.path
import numpy as np
from numpy import *

class LoadData:

    # Variables
    number_of_channels = 16
    sample_rate = 250.4
    sample_datatype = 'int16'
    display_decimation = 1


    def __init__(self, directory, filename):
        self.directory = directory
        self.filename = filename


    # If the data is in .dat format: 
    def parse_dat(self):
        '''Load a .dat file by interpreting it as int16 and then de-interlacing the 16 channels'''
        
        os.chdir(self.directory)
        
        # Load the raw (1-D) data
        dat_raw = np.fromfile(self.filename, dtype=self.sample_datatype)
        
        # Reshape the (2-D) per channel data
        step = self.number_of_channels * self.display_decimation
        dat_chans = [dat_raw[c::step] for c in range(self.number_of_channels)]
        
        # Build the time array
        t = np.arange(len(dat_chans[0]), dtype=float) / self.sample_rate
        
        return np.array(dat_chans), t


    # Get the data directly if it is already in .npy format:
    def get_dat(self):
        os.chdir(self.directory)
        return np.load(self.filename)

    
    # The .dat files are quite large so it would be better to transform them into numpy files to save memory space.
    def convert_dat_to_npy(self, path_to_save_folder, save_as_name):
        
        os.chdir(self.directory)
        
        dat_chans, t = self.parse_dat()
        
        data_to_save = np.array(dat_chans)
        
        os.chdir(path_to_save_folder)
        
        np.save(save_as_name, data_to_save)
        print('File saved for ' + save_as_name)


#Examples:

#directory = '/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/TAINI files/S7072'
#filename = "TAINI_1033_S7072_Drug2-2020_03_23-0000.dat"
#path_to_save_folder = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
#save_as_name = 'S7072xxx'
#data = LoadData(directory = directory, filename=filename)
#data.convert_dat_to_npy(path_to_save_folder=path_to_save_folder, save_as_name=save_as_name)
#unfiltered_data, t = data.parse_dat()

#directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
#filename = 'S7072xxx.npy'
#data = LoadData(directory=directory, filename=filename)
#loaded_data = data.get_dat()

