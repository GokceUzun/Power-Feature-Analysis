'''Module that plots raw sleep stage and calculates the correlation matrix'''

import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sn

import mne
import matplotlib.pyplot

from LoadData import LoadData
from Filter import Filter

import random
import scipy
import seaborn as sns

baseline_1 = ['brain_states_1_S7063.pkl', 'brain_states_1_S7064.pkl', 'brain_states_1_S7069.pkl','brain_states_1_S7068.pkl', 'brain_states_1_S7070.pkl', 'brain_states_1_S7071.pkl']
baseline_2 = ['brain_states_2_S7063.pkl', 'brain_states_2_S7064.pkl', 'brain_states_2_S7069.pkl', 'brain_states_2_S7070.pkl']
recordings = ['S7063_GAP.npy', 'S7069_GAP.npy', 'S7064_GAP.npy', 'S7071_WT.npy', 'S7070_WT.npy', 'S7068_WT.npy']

ch_names = ['S1Tr_RIGHT', 'EMG_RIGHT', 'M2_FrA_RIGHT','M2_ant_RIGHT','M1_ant_RIGHT', 'V2ML_RIGHT',
            'V1M_RIGHT', 'S1HL_S1FL_RIGHT', 'V1M_LEFT', 'V2ML_LEFT', 'S1HL_S1FL_LEFT',
            'M1_ant_LEFT','M2_ant_LEFT','M2_FrA_LEFT', 'EMG_LEFT', 'S1Tr_LEFT']

ch_types = ['eeg', 'emg', 'eeg', 'eeg', 'eeg', 'eeg',
           'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
           'eeg', 'eeg', 'eeg', 'emg', 'eeg']


# Function to calculate the psd of the channels and return as dataframe
# Estimate power spectral density using Welch’s method and Hanning window. 
def calculate_psd(data, average):
    
    welch_channel = []
    sampling_rate = 250.4
    nperseg = 1252
    
    for data_array in data:
        f, psd = scipy.signal.welch(data_array, sampling_rate, window = 'hann', nperseg = nperseg) #nperseg?
        # f might be needed for plotting?
        welch_channel.append(psd)
        
    df_psd = pd.DataFrame(welch_channel)
        
    if average == True:
        mean_values = df_psd.mean(axis = 0)
        mean_psd = mean_values.to_numpy()
        return mean_psd
    else:
        return df_psd



class animal():

    def __init__(self, no, baseline, start, end):
        self.no = no
        self.start = start
        self.end = end
        self.baseline = baseline

        filename = [f for f in recordings if self.no in f][0]

        if self.baseline == 1:
            picklefile = [f for f in baseline_1 if self.no in f][0]
        elif self.baseline == 2:
            picklefile = [f for f in baseline_2 if self.no in f][0]

        # Load the recording, slice and filter it:
        directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/'
        data = LoadData(directory, filename)
        unfiltered_data = data.get_dat()[:, self.start:self.end]
        fltr = Filter(unfiltered_data)
        self.filtered_data = np.array(fltr.butter_bandpass())

        #Load the brain state file:
        os.chdir('/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/SYNGAP_brain_states/baseline_brain_states')
        self.brain_state = pd.read_pickle(picklefile)


    def corrMatrix(self, sleep_stage):
        
        # 1) Extract epochs for each sleep stage  
        epochs = self.brain_state.loc[self.brain_state.brainstate == sleep_stage]
        n = random.randint(0, len(epochs)-5)
        epochs = epochs[n:n+5] # Taking 5 consecutive epochs
       
        time_start = int(epochs.start_epoch.iloc[0] * 250.4)
        time_end = int(epochs.end_epoch.iloc[len(epochs)-1] * 250.4)
        e = self.filtered_data[:, time_start:time_end] 
        
        # 2) Plot the raw recordings of those epochs
        raw_info = mne.create_info(ch_names, sfreq = 250.4, ch_types=ch_types)
        raw = mne.io.RawArray(e, raw_info)
        raw.plot(scalings="auto", title=str(self.no) + "-bl" +  str(self.baseline) + "-" + str(sleep_stage))
        #plt.show()
        
        # 3) Calculate the correlation matrices for those to decide on the channels 
        psd = calculate_psd(e, False).T
        psd.set_axis(['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16'], axis="columns", inplace=True)
        
        f = plt.figure(figsize=(12, 8))
        corrMatrix = psd.corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.title(str(self.no) + "-bl" +  str(self.baseline) + "-" + str(sleep_stage))
        plt.show()


# Calculate the correlation matrix for each animal, twice for each stage 
# One for bl1 one for bl2
# S7068 and S7071 doesnt have bl2, run them twice for bl1

# wake = 0
# nonrem = 1
# rem = 2

# GAP = S7063, S7064, S7069
animal_S7063_bl1 = animal("S7063", 1, 15324481, 36959040)
animal_S7063_bl1.corrMatrix(0)
animal_S7063_bl1.corrMatrix(1)
animal_S7063_bl1.corrMatrix(2)

animal_S7063_bl2 = animal("S7063", 2, 36959041, 58593600)
animal_S7063_bl2.corrMatrix(0)
animal_S7063_bl2.corrMatrix(1)
animal_S7063_bl2.corrMatrix(2)

# WT = S7068, S7070, S7071 
