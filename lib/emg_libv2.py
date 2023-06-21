import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

class EMGData:

    def __init__(self, file_str=None, f_samp=None, emg_data=None, emg_time_axis=None, channel_names=None):
        if file_str is not None:
            self.emg_data, self.emg_time_axis, self.channel_names = self.loadCometaEMGData(file_str)
        elif emg_data is not None and emg_time_axis is not None and channel_names is not None:
            self.emg_data = emg_data
            self.emg_time_axis = emg_time_axis
            self.channel_names = channel_names
        elif file_str is None and f_samp is not None and emg_data is not None:
            self.emg_data, self.emg_time_axis = self.loadMiniANTEMGData(file_str, f_samp)
            self.channel_names = channel_names

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


    def loadMiniANTEMGData(self, file_str, f_samp): 

        """ TODO: No sampling rate given in the data 
        This function loads the EMG data recorded from the ANT EMG system (as txt file, recorded via SDK). 
        Arguments:
            file_str: The file to load given as String. 
            f_samp:  The sampling rate in Hz. 

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
