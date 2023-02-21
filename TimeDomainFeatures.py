"""Computes the time domain features in a dataframe."""

import os
import pandas as pd
import numpy as np
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis

from LoadData import LoadData
from FilterBasic import Filter


class TimeDomainFeatures:
    def __new__(cls, data, channels):
        df = pd.DataFrame.from_records(data).T
        df.columns = ["ch" + str(i) for i in channels]

        # Time domain features:
        mean = []
        median = []
        variance = []
        skewness = []
        kurtosiS = []
        minimum = []
        maximum = []
        standard_deviation = []

        for i, ch in df.iterrows():
            m = []
            med = []
            var = []
            skw = []
            krt = []
            mN = []
            mX = []
            stdev = []

            for c in ch:
                m.append(np.mean(c))
                med.append(np.median(c))
                var.append(np.var(c))
                skw.append(skew(c))
                krt.append(kurtosis(c))
                mN.append(np.min(c))
                mX.append(np.max(c))
                stdev.append(np.std(c))

            mean.append(m)
            median.append(med)
            variance.append(var)
            skewness.append(skw)
            kurtosiS.append(krt)
            minimum.append(mN)
            maximum.append(mX)
            standard_deviation.append(stdev)

        # feature arrays to dataframe
        df_mean = pd.DataFrame(
            mean, columns=["mean_ch" + str(i) for i in channels]
        )
        df_median = pd.DataFrame(
            median, columns=["median_ch" + str(i) for i in channels]
        )
        df_var = pd.DataFrame(
            variance, columns=["var_ch" + str(i) for i in channels]
        )
        df_skw = pd.DataFrame(
            skewness, columns=["skw_ch" + str(i) for i in channels]
        )
        df_kur = pd.DataFrame(
            kurtosiS, columns=["kurt_ch" + str(i) for i in channels]
        )
        df_min = pd.DataFrame(
            minimum, columns=["min_ch" + str(i) for i in channels]
        )
        df_max = pd.DataFrame(
            maximum, columns=["max_ch" + str(i) for i in channels]
        )
        df_stdev = pd.DataFrame(
            standard_deviation,
            columns=["stdev_ch" + str(i) for i in channels],
        )

        # concat all the features in one dataframe
        time_dom_f = pd.concat(
            [df_mean, df_median, df_var, df_skw, df_kur, df_min, df_max, df_stdev],
            axis=1,
        )

        return time_dom_f


"""# TESTING:
directory = "/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project"
filename = "S7063_GAP.npy"
data = LoadData(directory=directory, filename=filename, start=15324481, end=36959040)
unfiltered_data = data.get_dat()
unfiltered_data = data.slice_data(unfiltered_data)

fltr_instance = Filter(unfiltered_data)
filtered_data = fltr_instance.butter_bandpass()
reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
# packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(reshaped_data)

selected_channels = fltr_instance.select_channels(
    reshaped_data=reshaped_data, channels=[5, 8]
)

dataframe = TimeDomainFeatures(selected_channels, fltr_instance.channels)
print(dataframe)
"""

