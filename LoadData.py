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
        
        return np.array(dat_chans)

    # Get the data directly if it is already in .npy format:
    def get_dat(self):
        os.chdir(self.directory)
        return np.load(self.filename)



#Examples:
#directory = '/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/TAINI files/S7072'
#filename = "TAINI_1033_S7072_Drug2-2020_03_23-0000.dat"
#data = LoadData(directory = path_to_folder, filename=filename)
#loaded_data = data.parse_dat() 

#directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
#filename = 'TAINI_1033_S7072_Drug2-2020_03_23-0000.npy'
#data = LoadData(directory=path_to_save_folder, filename=save_as_name)
#loaded_data = data.parse_dat()
