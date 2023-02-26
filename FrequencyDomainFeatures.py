"""Computes the frequency domain features in a dataframe."""

import os
import pandas as pd
import numpy as np
import scipy
from scipy.integrate import simps

from LoadData import LoadData
from FilterBasic import Filter
from CentreOfMass import CentreofMass


class FrequencyDomainFeatures:
    def __new__(cls, data, channels):
        df = pd.DataFrame.from_records(data).T
        df.columns = ["ch" + str(i) for i in channels]

        # Frequency domain features:
        #power_spectrum = []
        total_power = []
        band1_power = [] # 5-9Hz
        band2_power = [] # 1-20Hz
        band3_power = [] # 60-90Hz
        band4_power = [] # 1-5Hz
        centre_of_mass = []
        peaks = []

        for i, ch in df.iterrows():
            #ps = []
            tp = []
            bp1 = []
            bp2 = []
            bp3 = []
            bp4 = []
            com = []
            p = []

            for c in ch.index:
                # Calculate power spectral density and the frequency array
                freqs, psd = scipy.signal.welch(ch[c], 250.4, window="hann", nperseg=1252)

                freq_res = freqs[1] - freqs[0]  # frequency resolution (0.2)

                #ps.append(psd)

                tp.append(
                    simps(psd, dx=freq_res)
                )  # compute power by approximating the area under the psd curve

                # find intersecting values in frequency vector: idx_band = np.logical_and(freqs >= low, freqs <= high)
                # for each band compute power by approximating the area under the  curve
                if c == 'ch1' or c == 'ch14': # EMG channel only calculate 60-90Hz
                    bp1.append(None)  # 5-9Hz
                    bp2.append(None)  # 1-20Hz
                    bp3.append(
                        simps(psd[np.logical_and(freqs >= 60, freqs <= 90)], dx=freq_res)
                    )  # 60-90Hz
                    bp4.append(None)  # 1-5Hz

                else: # EEG channels calculate bands except 60-90Hz
                    bp1.append(
                        simps(psd[np.logical_and(freqs >= 5, freqs <= 9)], dx=freq_res)
                    )  # 5-9Hz
                    bp2.append(
                        simps(psd[np.logical_and(freqs >= 1, freqs <= 20)], dx=freq_res)
                    )  # 1-20Hz
                    bp3.append(None)  # 60-90Hz
                    bp4.append(
                        simps(psd[np.logical_and(freqs >= 1, freqs < 5)], dx=freq_res)
                    )  # 1-5Hz


                # centre of mass calculation
                df_pow = pd.DataFrame(psd, columns=["Power"])
                df_pow["Frequency"] = freqs
                com_instance = CentreofMass(df_pow)
                pow_arr, freq_arr = com_instance.frequency_slice()
                com.append(com_instance.calculate_cent_mass(pow_arr, freq_arr))

                p.append(psd[np.argmax(psd)])

            #power_spectrum.append(ps)
            total_power.append(tp)
            band1_power.append(bp1)
            band2_power.append(bp2)
            band3_power.append(bp3)
            band4_power.append(bp4)
            centre_of_mass.append(com)
            peaks.append(p)

        # feature arrays to dataframe
        #df_psd = pd.DataFrame(
         #   power_spectrum, columns=["psd_ch" + str(i) for i in channels]
        #) 

        df_tp = pd.DataFrame(
            total_power, columns=["totalP_ch" + str(i) for i in channels]
        )

        df_bp1 = pd.DataFrame(
            band1_power, columns=["5-9Hz_ch" + str(i) for i in channels]
        )

        df_bp2 = pd.DataFrame(
            band2_power, columns=["1-20Hz_ch" + str(i) for i in channels]
        )

        df_bp3 = pd.DataFrame(
            band3_power, columns=["60-90Hz_ch" + str(i) for i in channels]
        )

        df_bp4 = pd.DataFrame(
            band4_power, columns=["1-5Hz_ch" + str(i) for i in channels]
        )

        df_com = pd.DataFrame(
            centre_of_mass, columns=["com_ch" + str(i) for i in channels]
        )

        df_peaks = pd.DataFrame(
            peaks, columns=["peak_ch" + str(i) for i in channels]
        )

        # concat all the features in one dataframe
        freq_dom_f = pd.concat(
            [df_tp, df_bp1, df_bp2, df_bp3, df_bp4, df_com, df_peaks], axis=1
        ) # NOT adding the PSD to the feature matrix because the type is different

        return freq_dom_f.dropna(axis=1)

"""
# TESTING:
directory = "/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project"
filename = "S7063_GAP.npy"
data = LoadData(directory=directory, filename=filename, start=15324481, end=36959040)
unfiltered_data = data.get_dat()
unfiltered_data = data.slice_data(unfiltered_data)

fltr_instance = Filter(unfiltered_data, channels=[1, 3, 6, 7, 14])
filtered_data = fltr_instance.butter_bandpass()
reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
# packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(reshaped_data)


dataframe = FrequencyDomainFeatures(reshaped_data, fltr_instance.channels)
print(dataframe)
"""

