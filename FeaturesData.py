"""Outputs the features to be fed into the model."""
"""Optional: feature engineering relative power, noise removal, dimensionality reduction."""

import os
import pandas as pd
import numpy as np
import scipy
from scipy.integrate import simps

from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns

from LoadData import LoadData
from FilterBasic import Filter
from FrequencyDomainFeatures import FrequencyDomainFeatures
from TimeDomainFeatures import TimeDomainFeatures


class FeaturesData:
    def __new__(
        cls, fltr_instance, relativePower, removeNoise, applyPCA, brainstatefile
    ):
        # Do the filtering, reshaping, selecting channels:
        filtered_data = fltr_instance.butter_bandpass()
        reshaped_data = fltr_instance.reshape_filtered_data(filtered_data)
        packet_loss_array, packet_loss_idx = fltr_instance.packet_loss_indices(
            reshaped_data
        )

        # Calculate frequency and time domain features:
        freq_dom_features = FrequencyDomainFeatures(
            reshaped_data, fltr_instance.channels
        )  # 7 frequency domain features (psd is expluded)
        time_dom_features = TimeDomainFeatures(
            reshaped_data, fltr_instance.channels
        )  # 8 frequency domain features

        features = pd.concat([time_dom_features, freq_dom_features], axis=1)
        
        f_cols = list(time_dom_features.columns.values) + list(
            freq_dom_features.columns.values
        )

        # Calculate relative power of the bands as a feature and remove the redundant band and power columns:
        if relativePower == True:
            b = []
            p = []
            for ch in fltr_instance.channels:
                pow_col = "totalP_ch" + str(ch)
                for bands in ["5-9Hz_", "1-20Hz_", "60-90Hz_", "1-5Hz_"]:
                    new_col = "relPow_" + bands + "ch" + str(ch)
                    band_col = bands + "ch" + str(ch)
                    if band_col in features.columns.to_list():
                        b.append(band_col)
                        features[new_col] = features[band_col] / features[pow_col]

                p.append(pow_col)

            cols_to_remove = b + p
            features.drop(cols_to_remove, axis=1, inplace=True)
            f_cols = list(features.columns.values)


        if applyPCA == True:
            # Step1 - standardise data
            scalar = StandardScaler()
            scaled_data = pd.DataFrame(scalar.fit_transform(features))

            # Step2 - applying principle component analysis
            n = 20
            cols = ["PC" + str(i + 1) for i in range(n)]
            pca = PCA(n_components=n)  # Go for components 10-20
            pca.fit(scaled_data)
            data_pca = pca.transform(scaled_data)
            feature_matrix = pd.DataFrame(data_pca, columns=cols)
        else:
            feature_matrix = features


        # Get the brain states from the pickle file:
        os.chdir(
            "/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project" # Can change depending on where the pickle files are 
        )

        brainstatelabels = pd.read_pickle(brainstatefile)

        labeledFeatures = pd.concat([feature_matrix, brainstatelabels.brainstate], axis=1)
        labeledFeatures["packet_loss"] = packet_loss_idx


        # Remove the noisy epochs or keep them:
        if removeNoise == True:
            return labeledFeatures.loc[labeledFeatures["packet_loss"] == 0].drop(columns='packet_loss')
            feature_matrix = labeledFeatures[f_cols].loc[
                labeledFeatures["packet_loss"] == 0
            ]
            labels = labeledFeatures["brainstate"].loc[
                labeledFeatures["packet_loss"] == 0
            ]
        else:
            return labeledFeatures.drop(columns='packet_loss')
            feature_matrix = labeledFeatures[f_cols]
            labels = labeledFeatures["brainstate"]

        return pd.concat([finalFeatures, labels], axis=1, ignore_index=True)


"""
# TESTING:
directory = "/Volumes/Macintosh HD/Users/gokceuzun/Desktop/4. SENE/Honors Project"
filename = "S7063_GAP.npy"
data = LoadData(directory=directory, filename=filename, start=15324481, end=36959040)
unfiltered_data = data.get_dat()
unfiltered_data = data.slice_data(unfiltered_data)

fltr_instance = Filter(unfiltered_data, channels=[3,12,5,9,4,11,1])

features = FeaturesData(
    fltr_instance,
    relativePower=False,
    removeNoise=True,
    applyPCA=True,
    brainstatefile="S7063_BL1.pkl")

print(features)
"""



