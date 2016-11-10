
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
from joblib import Memory
import sys
sys.path.append(os.getcwd()+'/utils/')
import eegml_data_load


# In[18]:

cache_directory = './cache'
mem = Memory(cachedir=cache_directory, verbose=0)

@mem.cache
def normalize_and_bin_power_spectra(file, bin_size):

    # Load power data.
    freq, power = eegml_data_load.load_power_data(file['path'])

    # Handle corrupt file.
    if power.size == []: 
        return []

    def bin_power(x,bin_size):
        return np.sum(x.reshape(-1, bin_size), axis=1)

    # Normalize power spectra.
    norm_power = (power.T/power.sum(axis=1)).T

    # Bin normalized power spectra. 
    num_channels = power.shape[0]
    bin_norm_power = np.array([bin_power(norm_power[i,:], bin_size) for i in np.arange(num_channels)])

    return bin_norm_power


# In[ ]:



