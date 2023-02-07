import LoadFile
import Filter

#COMBINE BOTH CLASSES HERE!

# Loads the file from the directory into a numpy array 
# Input filename and directory, output unfiltered recording as numpy array
unfiltered_data = LoadData(filename, directory).parse_dat() 

# Input the unfiltered recording numpy array, output filtered out numpy array
preprocessed_data = Filter(unfiltered_data).butter_bandpass() 