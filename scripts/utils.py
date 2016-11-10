
# coding: utf-8

# In[1]:

import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from joblib import Memory
import re


# In[46]:

BASE_DATA_DIR = '../raw_data'
def create_files_df():
    def pathToSeries(path):
        m = re.search('/(.*)_(.*)_(.*).mat',path)
        if 'train' in path: 
            train = 1
            patient = int(m.group(1))
            preictal = int(m.group(3))
        else: 
            train = 0
            patient = int(m.group(2))
            preictal = -1

        m = re.search('(.*)/(.*)',path)
        dir_name = m.group(1)
        file_name = m.group(2)
        return pd.Series({'path':path,'train':train,'preictal':preictal, 'patient':patient,
                         'dir_name': dir_name,'file_name':file_name})

    # Get Metadata for all files in raw data directory. 
    data_dirs = [name for name in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, name))]
    paths = [[os.path.join(data_dir, name) for name in os.listdir(os.path.join(BASE_DATA_DIR,data_dir))] 
             for data_dir in data_dirs] 
    paths = [j for i in paths for j in i] # flatten to 1d
    paths = [i for i in paths if '.DS_Store' not in i]
    files_df = pd.DataFrame([pathToSeries(p) for p in paths])

    # Filter training files by safe labels. 
    safe_labels = pd.read_csv(os.path.join(BASE_DATA_DIR, 'train_and_test_data_labels_safe.csv'))
    filt_df = files_df
    filt_df = filt_df.merge(safe_labels,how='left',left_on='file_name', right_on ='image')
    filt_df = filt_df[(filt_df['train']==0) | ((filt_df['train']==1) & (filt_df['safe']==1.0))]
    filt_df = filt_df.drop(['image','class','safe'],axis=1)

    # Final processing. 
    files_df = filt_df
    files_df = files_df.reset_index(drop=True)
    files_df['file_id'] = np.arange(0,files_df.shape[0])

    return files_df


# In[ ]:

SAMPLING_RATE = 400
cache_directory = './cache'
mem = Memory(cachedir=cache_directory, verbose=0)

def load_temporal_data(path):
    try: 
        mat = loadmat(os.path.join(BASE_DATA_DIR,path))
        data = mat['dataStruct']['data'][0][0].transpose()
        time = np.arange(0,data.shape[1],1/SAMPLING_RATE)
        return time, data
    except:
        return np.empty(0), np.empty(0)

@mem.cache
def load_power_data(path):
    temporal_data = load_temporal_data(path)
    if temporal_data.size == 0:
        return np.empty(0), np.empty(0)
    n = temporal_data.shape[1]
    freq = np.array(np.fft.fftfreq(n)*SAMPLING_RATE)[:n//2]
    power = np.abs(np.fft.fft(temporal_data))[:,:n//2]
    return freq, power

