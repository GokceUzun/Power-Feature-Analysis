import os 
import pandas as pd
import numpy as np
import  scipy
from scipy.integrate import simps

from LoadData import LoadData
from FilterBasic import Filter
from CentreOfMass import CentreofMass

class FrequencyDomainFeatures:

    def __new__(cls, reshaped_data):
        reshaped_data = reshaped_data

        df = pd.DataFrame.from_records(reshaped_data).T
        channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
        df.columns = channels
        
        # PSD 
        df_PSD = df.applymap(lambda x: scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[1])
        df_PSD.columns = ['psd_ch1', 'psd_ch2', 'psd_ch3', 'psd_ch4', 'psd_ch5', 'psd_ch6', 'psd_ch7', 'psd_ch8', 'psd_ch9', 'psd_ch10', 'psd_ch11', 'psd_ch12', 'psd_ch13', 'psd_ch14', 'psd_ch15', 'psd_ch16']
    
        # Total power
        def total_power(x):
            epoch_psd = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[1]
            freqs = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[0]
            freq_res = freqs[1] - freqs[0] # 0.2
            total_power = simps(epoch_psd, dx=freq_res)
            return total_power
        
        df_totalP = df.applymap(lambda x: total_power(x))
        df_totalP.columns = ['totalP_ch1', 'totalP_ch2', 'totalP_ch3', 'totalP_ch4', 'totalP_ch5', 'totalP_ch6', 'totalP_ch7', 'totalP_ch8', 'totalP_ch9', 'totalP_ch10', 'totalP_ch11', 'totalP_ch12', 'totalP_ch13', 'totalP_ch14', 'totalP_ch15', 'totalP_ch16']
       
       
       # Band power
        def band_power(x, low, high):
    
            epoch_psd = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[1]
            freqs = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[0]

            # Find intersecting values in frequency vector
            idx_band = np.logical_and(freqs >= low, freqs <= high)
    
            # Frequency resolution
            freq_res = freqs[1] - freqs[0]

            # Compute the absolute power by approximating the area under the curve
            band_power = simps(epoch_psd[idx_band], dx=freq_res)
    
            return band_power
        
        # 5-9 Hz
        df_band1 = df.applymap(lambda x: band_power(x, 5, 9))
        df_band1.columns = ['5-9Hz_ch1', '5-9Hz_ch2', '5-9Hz_ch3', '5-9Hz_ch4', '5-9Hz_ch5', '5-9Hz_ch6', '5-9Hz_ch7', '5-9Hz_ch8', '5-9Hz_ch9', '5-9Hz_ch10', '5-9Hz_ch11', '5-9Hz_ch12', '5-9Hz_ch13', '5-9Hz_ch14', '5-9Hz_ch15', '5-9Hz_ch16']
        
        # 1-20 Hz
        df_band2 = df.applymap(lambda x: band_power(x, 1, 20))
        df_band2.columns = ['1-20Hz_ch1', '1-20Hz_ch2', '1-20Hz_ch3', '1-20Hz_ch4', '1-20Hz_ch5', '1-20Hz_ch6', '1-20Hz_ch7', '1-20Hz_ch8', '1-20Hz_ch9', '1-20Hz_ch10', '1-20Hz_ch11', '1-20Hz_ch12', '1-20Hz_ch13', '1-20Hz_ch14', '1-20Hz_ch15', '1-20Hz_ch16']
        
        # 60-90 Hz
        df_band3 = df.applymap(lambda x: band_power(x, 60, 90))
        df_band3.columns = ['60-90Hz_ch1', '60-90Hz_ch2', '60-90Hz_ch3', '60-90Hz_ch4', '60-90Hz_ch5', '60-90Hz_ch6', '60-90Hz_ch7', '60-90Hz_ch8', '60-90Hz_ch9', '60-90Hz_ch10', '60-90Hz_ch11', '60-90Hz_ch12', '60-90Hz_ch13', '60-90Hz_ch14', '60-90Hz_ch15', '60-90Hz_ch16']
        
        # Centre of Mass
        def calculate_centre_of_mass(x):
            psd = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[1]
            freqs = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[0]
    
            df_pow = pd.DataFrame(psd, columns=['Power'])
            df_pow['Frequency'] = freqs
    
            com = CentreofMass(df_pow, 0, 'ch1', 'S7063')
            pow_arr, freq_arr = com.frequency_slice()
    
            return com.calculate_cent_mass(pow_arr, freq_arr)
    
        df_com = df.applymap(lambda x: calculate_centre_of_mass(x))
        df_com.columns = ['com_ch1', 'com_ch2', 'com_ch3', 'com_ch4', 'com_ch5', 'com_ch6', 'com_ch7', 'com_ch8', 'com_ch9', 'com_ch10', 'com_ch11', 'com_ch12', 'com_ch13', 'com_ch14', 'com_ch15', 'com_ch16']
        
        # Peaks
        def find_peak(x):
            psd = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[1]
            freqs = scipy.signal.welch(x, 250.4, window = 'hann', nperseg = 1252)[0]
            maxIndex = np.argmax(psd)
            return psd[maxIndex]
        
        df_peak = df.applymap(lambda x: find_peak(x))
        df_peak.columns = ['peak_ch1', 'peak_ch2', 'peak_ch3', 'peak_ch4', 'peak_ch5', 'peak_ch6', 'peak_ch7', 'peak_ch8', 'peak_ch9', 'peak_ch10', 'peak_ch11', 'peak_ch12', 'peak_ch13', 'peak_ch14', 'peak_ch15', 'peak_ch16']
        

        # FEATURE SELECTION
        df_features = pd.concat([df_PSD, df_com, df_totalP, df_band1, df_band2, df_band3, df_peak], axis=1)
        return len(df_features.columns)

       
    


# TESTING: 
directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project'
filename = 'S7063_GAP.npy'
data = LoadData(directory=directory, filename=filename, start=15324481, end=36959040)
unfiltered_data = data.get_dat()
unfiltered_data = data.slice_data(unfiltered_data)

fltr_instance = Filter(unfiltered_data)
filtered_data = fltr_instance.butter_bandpass()
reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
#packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(reshaped_data)

dataframe = FrequencyDomainFeatures(reshaped_data)
print(dataframe)
