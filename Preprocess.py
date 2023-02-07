'''Module that combines both preprocessing steps loading and filtering'''

from LoadData import LoadData
from Filter import Filter

#directory = '/Volumes/Gonzalez-Sulser/SYNGAP SLEEP 24hr PAPER/Circadian-ETX/TAINI files/S7072'
#filename = "TAINI_1033_S7072_Drug2-2020_03_23-0000.dat"
#unfiltered_data = LoadData(filename, directory).parse_dat() 

directory = '/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project/S7072'
filename = 'TAINI_1033_S7072_Drug2-2020_03_23-0000.npy'

data = LoadData(directory, filename)
unfiltered_data = data.get_dat()

fltr = Filter(unfiltered_data)
filtered_data = fltr.butter_bandpass()

#print(data)
#print(unfiltered_data)

#print(fltr)
#print(filtered_data)



