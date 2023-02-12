'''Module that combines both preprocessing steps loading and filtering'''

import numpy as np
from LoadData import LoadData
from Filter import Filter

directory = '/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/TAINI files/S7072'
filename = "TAINI_1033_S7072_Drug2-2020_03_23-0000.dat"
#path_to_save_folder = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
#save_as_name = 'S7072xxx'
data = LoadData(directory = directory, filename=filename)
#data.convert_dat_to_npy(path_to_save_folder=path_to_save_folder, save_as_name=save_as_name)
unfiltered_data, t = data.parse_dat()
print(unfiltered_data)

directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
filename = 'S7072xxx.npy'
data = LoadData(directory, filename)
unfiltered_data = data.get_dat()
print(unfiltered_data)

#fltr = Filter(unfiltered_data)
#filtered_data = np.array(fltr.butter_bandpass())
#noise = fltr.noise_index

#print(data)
#print(unfiltered_data)

#print(fltr)
#print(len(filtered_data))
#print(len(noise))



