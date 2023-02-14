import os 
import numpy as np 
import pandas as pd 

class CentreofMass():

    def __init__(self, df, brainstate_number, channel, animal_id):
        self.df = df 
        self.brainstate_number = brainstate_number
        self.channel = channel 
        self.animal_id = animal_id

    
    def frequency_slice(self):
        #input power dataframe
        freq_df = self.df[(self.df['Frequency'] >= 1) & (self.df['Frequency'] <= 49)]

        #select power and frequency columns 
        power_array = freq_df['Power'].to_numpy()
        freq_array = freq_df['Frequency'].to_numpy()

        return power_array, freq_array

    def calculate_cent_mass(self, power_array, freq_array):
        #calculate sum of power values 
        sum_power = sum(power_array)

        #normalise array 
        summed_freq_power = []
        for power_value, freq_value in zip(power_array, freq_array):
            summed_freq_power.append(power_value*freq_value)
        
        #calculate centre of mass
        centre_of_mass = sum(summed_freq_power)/sum_power

        return centre_of_mass
