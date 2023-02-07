'''Filters the raw numpy array recording that is loaded.'''
'''It filters out data below 0.2Hz and above 100Hz using a bandpass filter and also filters out epochs above 3000mV.'''

import scipy
from scipy.fft import fft, fftfreq
from scipy import signal
import os 
import numpy as np


class Filter:

    # Variables
    order = 3
    sampling_rate = 250.4
    nyquist = 125.2
    low = 0.2/nyquist
    high = 100/nyquist
    noise_limit = 3000
    epoch_bins = 1252 #5 seconds * sampling rate

    def __init__(self, unfiltered_data):
        self.unfiltered_data = unfiltered_data

    def butter_bandpass(self):
        
        channel_threshold = []
        
        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype='band', analog = False)
        
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
    
        for channel in filtered_data:
            for value in channel:
                if value >= self.noise_limit:
                    channel_threshold.append(value)
                else:
                    pass 
            
        remove_duplicates = sorted(list(set(channel_threshold)))

        channels_without_noise = [i for j, i in enumerate(filtered_data) if j not in remove_duplicates]
        
        return channels_without_noise # Returns the filtered numpy array 


#directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
#os.chdir(directory)
#unfiltered_data = np.load('TAINI_1033_S7072_Baseline-2020_03_16-0000.npy')
#f = Filter(unfiltered_data)
#filtered_data = f.butter_bandpass()
