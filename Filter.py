'''Filters the raw numpy array recording that is loaded.'''
'''It filters out data below 0.2Hz and above 100Hz using a bandpass filter and also filters out epochs above 3000mV.'''

import scipy
from scipy.fft import fft, fftfreq
from scipy import signal
import os 
import numpy as np


class Filter:

    order = 3
    sampling_rate = 250.4 
    nyquist = 125.2
    low = 0.2/nyquist
    high = 100/nyquist
    
    def __init__(self, unfiltered_data):
        self.unfiltered_data = unfiltered_data
        
    def butter_bandpass(self):
        #stripped filter function to apply bandpass filter to entire recording before time and frequency domain calculations

        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype = 'band', analog = False)
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
        return filtered_data

        
        