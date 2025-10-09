# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import warnings 
#from datetime import datetime
import os
import mne
from pathlib import Path
from biosignal_toolbox.time_series_lib import OnlineTimeseriesStreaming
from biosignal_toolbox.time_series_lib import Timeseries
from biosignal_toolbox.utils import getAbsolutePath


# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************
class EMGData(Timeseries):
    """
    This class includes useful methods for the processing and visualization of EMG data.

    Parameters
    ----------
    Timeseries : class
        The base timeseries class that includes most of the data processing methods for biosignals (e.g. filters for EMG and EEG etc.)
    """

    def __init__(self, data_arr = None, format="ANTmini",data_path = None, filenames = None, f_samp = 500, channel_names = None):

        """
        The constructor of the EMG class. 
            
        Parameters
        ----------
        data_arr: array, expects data in the format of function loadMiniANTEMGData
        format : str, optional
            The format in which the data is loaded", by default "ANTmini"
        data_path : str, optional
            The path where the data is stored, by default None
        filenames : list of str, optional
            A list of filenames to loaded, currently only one set can be loaded at a time (single element in the list), by default None
        f_samp : int, optional
            The sampling rate of the EMG system in Hz, by default None
        channel_names : list of str, optional
            A list of strings with the channel names/muscles, by default None 

        Attributes
        ----------
        raw_obj : mne raw object
            The mne raw object that is used to create the object. Only required for format type "RawObj".
        f_samp : float
            The sampling rate of the EMG system in Hz
        channel_names : list
            A list of channel names as strings, if not known from the data format.
        data : numpy ndarray
             The channel wise (raw) data as numpy array (shape: n_channel, n_sampels). 
        epochs : numpy ndarray
            The epoched data as numpy ndarray with shape (n_trials, n_channels, n_sampels).
        windows : numpy ndarray
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows).
        events : numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 18.04.2024 (by Niklas Kueper)
        """  
        
        # parameter 
        self.raw_obj = None
        self.f_samp = f_samp
        self.channel_names = channel_names
        # data structures 
        self.data = None
        self.epochs = None
        self.windows = None
        self.events = None # not provided by loaded data yet 
        # absolute data path
        self.data_path = getAbsolutePath(input_path=data_path)
        
        if(filenames):
            if(isinstance(filenames, list)): # check if list or string class 
                raw_list = []
                events_list = []
                # filename = filenames[0] # use only one 
                for file in filenames:
                    if(format == "ANTmini"):
                        raw_data, self.time_axis,  = self.loadMiniANTEMGData(file, self.f_samp)

                        self.data = raw_data # store data in numpy array 
                        self.createMNERaw()
                        # Adding an extra event channel at the end for qualisys markers
                        column_of_no_markers = -1 * np.ones((1,self.data.shape[1]))
                        self.data = np.vstack((self.data,column_of_no_markers))
                        # Making the 10th and last samples of EMG as the boundaries for syncing
                        # This number is selected taking into account 20 ms start delay in qualisys
                        offset_idx = 1*(20*f_samp/1000)
                        self.data[-1,int(offset_idx)] = 1
                        self.data[-1,-1] = 2
                        print(f"Data: {self.data.shape}")

                        # set annotation events (markers)
                        event_channel = self.data[-1, :]
                        marker_indices = np.where(event_channel > 0)[0]
                        marker_numbers = event_channel[marker_indices]
                        events = np.zeros((len(marker_indices), 3))
                        events[:, 0] = marker_indices
                        events[:, 2] = marker_numbers
                        self.events = events.astype(int)

                    else: 
                        raw_data, self.time_axis, self.channel_names = self.loadCometaEMGData(file)

                        self.data = raw_data # store data in numpy array 
                        self.createMNERaw()
                    
                    raw_list.append(self.raw_obj)
                    events_list.append(self.events)
                
                #? Check if the channel names match in all raws

                for i, raw in enumerate(raw_list,start=1):
                    assert raw.ch_names == raw_list[0].ch_names, f"Channel mismatch in raw {i}"
                self.raw_obj, self.events = mne.concatenate_raws(raws=raw_list, events_list=events_list)
                self.data = self.raw_obj.get_data()

        if (data_arr):
            raw_list = []
            events_list = []

            if (format == "ANTmini"):
                raw_data, self.time_axis, = data_arr

                self.data = raw_data  # store data in numpy array
                self.createMNERaw()
                # Adding an extra event channel at the end for qualisys markers
                column_of_no_markers = -1 * np.ones((1, self.data.shape[1]))
                self.data = np.vstack((self.data, column_of_no_markers))
                # Making the 10th and last samples of EMG as the boundaries for syncing
                # This number is selected taking into account 20 ms start delay in qualisys
                offset_idx = 1 * (20 * f_samp / 1000)
                self.data[-1, int(offset_idx)] = 1
                self.data[-1, -1] = 2
                print(f"Data: {self.data.shape}")

                # set annotation events (markers)
                event_channel = self.data[-1, :]
                marker_indices = np.where(event_channel > 0)[0]
                marker_numbers = event_channel[marker_indices]
                events = np.zeros((len(marker_indices), 3))
                events[:, 0] = marker_indices
                events[:, 2] = marker_numbers
                self.events = events.astype(int)

                raw_list.append(self.raw_obj)
                events_list.append(self.events)

                # ? Check if the channel names match in all raws

                for i, raw in enumerate(raw_list, start=1):
                    assert raw.ch_names == raw_list[0].ch_names, f"Channel mismatch in raw {i}"
                self.raw_obj, self.events = mne.concatenate_raws(raws=raw_list, events_list=events_list)
                self.data = self.raw_obj.get_data()

        # print("data shape:", self.data.shape)
        super().__init__(f_samp = self.f_samp, channel_names = self.channel_names, raw_obj=self.raw_obj, events=self.events, data = self.data, epochs = self.epochs, windows = self.windows)


    def createMNERaw(self):
        """
        This function is used (internally) to create mne raw objects. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.04.2024 (by Niklas Kueper)
        """
        
        # create mne object 
        sfreq = self.f_samp  # Sampling frequency
        data = self.data # (channel, sampels)
        #times = np.arange(0, data.shape[1], 1/sfreq)  # 
        ch_types = ['emg'] * len(self.channel_names) # 
        info = mne.create_info(ch_names=self.channel_names, sfreq=sfreq, ch_types=ch_types)
        #scalings = {'emg': 1}
        raw = mne.io.RawArray(data[0:len(self.channel_names), :], info) # only pass the actual EMG channel 
        self.raw_obj = raw
        self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels)


    def loadCometaEMGData(self, file_str):
        """
        This funcion loads the EMG data recorded from the Cometa EMG system (as txt file)

        Parameters
        ----------
        data_path : str
            The path where the data is stored
        file_str : str
            The file to load

        Returns
        -------
        tuple
            emg_data_channel : numpy array
                The EMG data with shape (n_samples, n_channel)
            emg_time_axis : numpy array
                The time axis of the EMG data with shape (n_samples, )
            channel_names : numpy array
                The EMG channel names/muscles (names specified in the recording software) with shape (n_channels, )

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 31.01.2023 (by Niklas Kueper)
        """

        filename = self.data_path / Path(file_str)
        emg_data = np.loadtxt(filename, dtype = float, delimiter=None, skiprows=5)

        # extract EMG channel names 
        channel_names = np.loadtxt(filename, dtype = str, delimiter=':', max_rows=1, skiprows=4)
        channel_names = list(channel_names[1:-1]) # cut off last and first values since they are not EMG channel names 

        emg_data_channel = emg_data[:, 1:] # channel dimensions 
        emg_time_axis = emg_data[:, 0] # time axis 

        # calc sampling freq from data 
        dt = emg_time_axis[1] - emg_time_axis[0]
        self.f_samp = float(1/dt)

        emg_data_channel = emg_data_channel.T # transpose for same data format (channels, sampels)

        return emg_data_channel, emg_time_axis, channel_names

    def loadMiniANTEMGData(self, file_str, f_samp):
        """
        TODO: No sampling rate given in the data
        This function loads the EMG data recorded from the ANT EMG system (as .txt file, recorded via SDK)

        Parameters
        ----------
        data_path : str
            The path where the data is stored
        file_str : str
            The file to load
        f_samp : int
            The sampling frequency in Hz

        Returns
        -------
        tuple
            emg_data : numpy array
                The EMG data with shape (n_samples, n_channel)
            time_axis : numpy array
                The time axis of the EMG data with shape (n_samples,)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 31.01.2023 (by Niklas Kueper)
        """

        #seperate between data, meta and channel names 
        emg_data_raw = np.loadtxt(self.data_path / Path(file_str)) # at least th
        emg_data = emg_data_raw[:-1, :-2]
        time_axis = np.arange(0, (len(emg_data)/f_samp), step = 1/f_samp)

        emg_data = emg_data.T # transpose for timeseries data format 

        return emg_data, time_axis 
    
    
    def showEMGData(self): # deprecated, will be removed in future 
    
        """
        This function is plotting the EMG data

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
        warnings.warn("This method is deprecated, will be removed soon! Use mne methods for visualization for now ")

        print(self.data.shape)

        if (self.data.ndim > 1): 
            n_channels = self.data.shape[0]

            for n_channel in range(0, n_channels): 
                plt.figure()
                plt.plot(self.time_axis, self.data[n_channel, :])
                plt.title(self.channel_names[n_channel])
                plt.xlabel("Time in seconds")
                plt.ylabel("Voltage in uV")
        else:
            plt.figure()
            plt.plot(self.time_axis, self.data)
            plt.title(self.channel_names)
            plt.xlabel("Time in seconds")
            plt.ylabel("Voltage in uV")

        plt.show()


    def calcSampleLossFromSameSamples(self, num_allowed_samples = 5):
        """
        A helping method for calculating how much samples are lost in a recording for the cometa system (wireless connection loss). 

        Returns
        -------
        numpy ndarray
            A numpy array with shape: (channels, sampels)) containing the numbers of samples that are lost for each recording channel and index. Only the total amount of lost samples are shown for each index. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 02.02.2024 (by Niklas Kueper)

        """

        loss_count_local = 0 
        loss_array = np.zeros(self.data.shape) 

        if (True): 
            for channel_idx in range(0, self.data.shape[0]): 
                is_this_samp_loss = False
                is_last_samp_loss = False
                print("loss channel: " +str(channel_idx))

                for sample_idx in range (1, self.data.shape[1]): 
                    is_last_samp_loss = is_this_samp_loss  # update 

                    if(self.data[channel_idx, sample_idx] == self.data[channel_idx, sample_idx-1]): # same value as before 
                        loss_count_local = loss_count_local+1
                        is_this_samp_loss = True 
                    else: 
                        is_this_samp_loss = False
                    
                    
                    if(is_last_samp_loss and not is_this_samp_loss): # the loss is over at this point 
                        loss_array[channel_idx, sample_idx] = loss_count_local
                        loss_count_local = 0 


                loss_count = np.sum(loss_array[channel_idx, :] > num_allowed_samples) # calc actual number of losses above threshold  

                print("number of data losses: ", loss_count)

            return loss_array

        else: 
            warnings.warn("not implemented for other data stages, terminating ... ")
            return None

class OnlineEMG(OnlineTimeseriesStreaming, EMGData): 
    """
    This class provides useful methods for doing old_online EMG processing and classification. It inherits processing methods from the EMGData class and methods for data streaming from OnlineTimeseriesStreaming .

    Parameters
    ----------
    OnlineTimeseriesStreaming : class
        The OnlineTimeseriesStreaming includes 
    EMGData : _type_
        _description_
    """

    def __init__(self, stream_type = "data", channel_names = ["1", "2", "3"], n_samples= 500, dt_process_data = 0.05, f_samp = 1000.0): 
        """
        The constructor of the OnlineEMG class. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        
        OnlineTimeseriesStreaming.__init__(self, stream_type = stream_type, channel_names = channel_names, n_samples= n_samples, dt_process_data = dt_process_data, f_samp = f_samp)
        EMGData.__init__(self, format = "Live", f_samp = f_samp, channel_names = channel_names)

        


    

  
