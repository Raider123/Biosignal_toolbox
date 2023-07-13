# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.utils import shuffle
from scipy import signal as sig

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

class TimeseriesData:

    def __init__(self, file_str, f_samp):

        """
        Initializing EMGData class

        Attributes:
        file_str = file name
        f_samp = sampling rate
        """

        self.file_str = file_str
        self.f_samp = f_samp

    def loadBrainproductsData(dataset_list): 

        """
        This function can be used for loading one or more datasets in brainproducts format.
        Arguments:
            dataset_list: A list of strings with filenames of the datasets to be loaded.

        Returns:
            raw: An mne object with the loaded (concatenated) dataset(s). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
        """
        
        if (len(dataset_list) > 1): 
            raw_list = []
            for dataset in dataset_list: 
                raw1 = mne.io.read_raw_brainvision(dataset, preload = True, verbose = False)
                raw_list.append(raw1)
            raw = mne.concatenate_raws(raw_list)
        else: 
            raw = mne.io.read_raw_brainvision(dataset_list[0], preload = True, verbose = False)

        return raw
    
    def loadCometaEMGData(self, file_str):

        """
        This function loads the EMG data recorded from the Cometa EMG system (as txt file). 
        Arguments:
            file_str: The file to load given as String. 

        Returns: 
            emg_data_channel: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
            emg_time_axis: The time axis of the EMG data as numpy array (shape: (n_sampels,)). 
            channel_names: The EMG channel names/muscles (names specified in the recording software) as numpy array (shape: (n_channels,)). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 31.01.2023 (by Niklas Kueper)
        """

        #seperate between data, meta and channel names 
        emg_data = np.loadtxt(file_str, dtype = float, delimiter=None, skiprows=5)

        # extract EMG channel names 
        channel_names = np.loadtxt(file_str, dtype = str, delimiter=':', max_rows=1, skiprows=4)
        channel_names = channel_names[1:-1] # cut off last and first values since they are not EMG channel names 

        emg_data_channel = emg_data[:, 1:] # channel dimensions 
        emg_time_axis = emg_data[:, 0] # time axis 

        return emg_data_channel, emg_time_axis, channel_names

    
    def loadMiniANTEMGData(file_str, f_samp): 

        """ TODO: No sampling rate given in the data 
        This function loads the EMG data recorded from the ANT EMG system (as txt file, recorded via SDK). 
        Arguments:
            file_str: The file to load given as String. 

        Returns: 
            emg_data_channel: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
            emg_time_axis: The time axis of the EMG data as numpy array (shape: (n_sampels,)). 
            channel_names: The EMG channel names/muscles (names specified in the recording software) as numpy array (shape: (n_channels,)). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 31.01.2023 (by Niklas Kueper)
        """

        #seperate between data, meta and channel names 
        emg_data_raw = np.loadtxt(file_str) # at least th
        emg_data = emg_data_raw[:-1, :-2]
        time_axis = np.arange(0, (len(emg_data)/f_samp), step = 1/f_samp)


        return emg_data, time_axis