'''Filters the raw numpy array recording that is loaded.'''
'''It filters out data below 0.2Hz and above 100Hz using a bandpass filter and also filters out epochs above 3000mV.'''

import scipy
from scipy.fft import fft, fftfreq
from scipy import signal
import os 
import numpy as np
from LoadData import LoadData


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
        self.noise_index = [] # Index tracker to keep tract of the indices that are discarded
        self.channels = [i for i in range(16)]

    def butter_bandpass(self):
        
        #stripped filter function to apply bandpass filter to entire recording before time and frequency domain calculations
        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype='band', analog = False)
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
        return filtered_data
        
    def reshape_filtered_data(self, filtered_data):
        
        #function to reshape data into 5 second epoch bins 
        dataset_length = (len(filtered_data[0])) #calculate total number of data points
        #print(dataset_length)
        number_of_epochs = int(dataset_length/1252)
        #print(number_of_epochs)
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
    

    # Select channels, default is 16 unless the function is called
    def select_channels(self, reshaped_data, channels):
        self.channels = channels
        return reshaped_data[channels]
    


"""
# TESTING: 
directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project'
filename = 'S7063_GAP.npy'
data = LoadData(directory=directory, filename=filename, start=15324481, end=36959040)
unfiltered_data = data.get_dat()
unfiltered_data = data.slice_data(unfiltered_data)

fltr_instance = Filter(unfiltered_data)
filtered_data = fltr_instance.butter_bandpass()
reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(reshaped_data)

print(len(reshaped_data))
print(len(reshaped_data[0]))
print(reshaped_data[0])
print(len(reshaped_data[0][0]))
print(reshaped_data[0][0])
"""

