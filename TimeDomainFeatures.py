import os 
import pandas as pd
import numpy as np
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis

from LoadData import LoadData
from FilterBasic import Filter

class TimeDomainFeatures:

    def __new__(cls, reshaped_data):
        reshaped_data = reshaped_data

        df = pd.DataFrame.from_records(reshaped_data).T
        channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
        df.columns = channels

        # Mean
        df_mean = df.applymap(lambda x: np.mean(x))
        df_mean.columns = ['mean_ch1', 'mean_ch2', 'mean_ch3', 'mean_ch4', 'mean_ch5', 'mean_ch6', 'mean_ch7', 'mean_ch8', 'mean_ch9', 'mean_ch10', 'mean_ch11', 'mean_ch12', 'mean_ch13', 'mean_ch14', 'mean_ch15', 'mean_ch16']

        # Median
        df_median = df.applymap(lambda x: np.median(x))
        df_median.columns = ['median_ch1', 'median_ch2', 'median_ch3', 'median_ch4', 'median_ch5', 'median_ch6', 'median_ch7', 'median_ch8', 'median_ch9', 'median_ch10', 'median_ch11', 'median_ch12', 'median_ch13', 'median_ch14', 'median_ch15', 'median_ch16']
        
        # Variance
        df_variance = df.applymap(lambda x: np.var(x))
        df_variance.columns = ['var_ch1', 'var_ch2', 'var_ch3', 'var_ch4', 'var_ch5', 'var_ch6', 'var_ch7', 'var_ch8', 'var_ch9', 'var_ch10', 'var_ch11', 'var_ch12', 'var_ch13', 'var_ch14', 'var_ch15', 'var_ch16']
        
        # Skewness
        df_skewness = df.applymap(lambda x: skew(x))
        df_skewness.columns = ['skew_ch1', 'skew_ch2', 'skew_ch3', 'skew_ch4', 'skew_ch5', 'skew_ch6', 'skew_ch7', 'skew_ch8', 'skew_ch9', 'skew_ch10', 'skew_ch11', 'skew_ch12', 'skew_ch13', 'skew_ch14', 'skew_ch15', 'skew_ch16']

        # Kurtosis
        df_kurtosis = df.applymap(lambda x: kurtosis(x))
        df_kurtosis.columns = ['kur_ch1', 'kur_ch2', 'kur_ch3', 'kur_ch4', 'kur_ch5', 'kur_ch6', 'kur_ch7', 'kur_ch8', 'kur_ch9', 'kur_ch10', 'kur_ch11', 'kur_ch12', 'kur_ch13', 'kur_ch14', 'kur_ch15', 'kur_ch16']
        

        # Min
        df_min = df.applymap(lambda x: np.min(x))
        df_min.columns = ['min_ch1', 'min_ch2', 'min_ch3', 'min_ch4', 'min_ch5', 'min_ch6', 'min_ch7', 'min_ch8', 'min_ch9', 'min_ch10', 'min_ch11', 'min_ch12', 'min_ch13', 'min_ch14', 'min_ch15', 'min_ch16']

        # Max
        df_max = df.applymap(lambda x: np.max(x))
        df_max.columns = ['max_ch1', 'max_ch2', 'max_ch3', 'max_ch4', 'max_ch5', 'max_ch6', 'max_ch7', 'max_ch8', 'max_ch9', 'max_ch10', 'max_ch11', 'max_ch12', 'max_ch13', 'max_ch14', 'max_ch15', 'max_ch16']

        # Standard Deviation
        df_stdev = df.applymap(lambda x: np.std(x))
        df_stdev.columns = ['stdev_ch1', 'stdev_ch2', 'stdev_ch3', 'stdev_ch4', 'stdev_ch5', 'stdev_ch6', 'stdev_ch7', 'stdev_ch8', 'stdev_ch9', 'stdev_ch10', 'stdev_ch11', 'stdev_ch12', 'stdev_ch13', 'stdev_ch14', 'stdev_ch15', 'stdev_ch16']

    
        # FEATURE SELECTION
        df_features = pd.concat([df_mean, df_median, df_skewness, df_kurtosis, df_variance, df_min, df_max, df_stdev], axis=1)
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

dataframe = TimeDomainFeatures(reshaped_data=reshaped_data)
print(dataframe)


