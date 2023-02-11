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

    def reshape_filtered_data(self, filtered_data):
        #function to reshape data into 5 second epoch bins 
        dataset_length = (len(filtered_data[0])) #calculate total number of data points
        
        number_of_epochs = dataset_length/1252

        reshaped_data = filtered_data.reshape(16, number_of_epochs, -1)

        return reshaped_data

    def packet_loss_indices(self, reshaped_data):
        #function to return list of values where 0 = clean and 6 = packet loss and each value represents the entire epoch
        def packet_loss(epoch):
            mask = epoch.max() < 3000
            return mask

        packet_loss_array = np.apply_along_axis(packet_loss, -1, arr = reshaped_data)
        
        #returns a boolean array, True == where values are below noise_threshold, false is where epochs are above. 

        packet_loss_indices = []
        for idx, epoch in enumerate(packet_loss_array[0]):
            if epoch == False:
                packet_loss_indices.append(6)
            else:
                packet_loss_indices.append(0)
        #returns a list of indices where 0 is clean epochs and 6 is packet loss or non-physiological noise

        return packet_loss_array, packet_loss_indices

fltr_instance = Filter_Basic(unfiltered_data)
filtered_data = fltr_instance.butter_bandpass()
reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(reshaped_data)
