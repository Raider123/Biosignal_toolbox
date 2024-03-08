# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal as sig
import os 
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import to_categorical
import mne_features.univariate as mne_feat
from mne.preprocessing import ICA
import copy 
from mne.preprocessing import Xdawn
import requests
from pybv import write_brainvision
import zmq 
import warnings


# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************


# not required anymore, will be removed fully soon 
# class OnlineEEGUtils: # leave this for backward compability for now --> deprecated 

#     def __init__(self, n_channels=34, n_samples= 500, dt_process_data = 0.05):
#         """_summary_ TODO: add description
#         Parameters
#         ----------
#         n_channels : int, optional
#             _description_, by default 34
#         n_samples : int, optional
#             _description_, by default 500
#         dt_process_data : float, optional
#             _description_, by default 0.05
#         """

#         self.buffersize = n_samples
#         self.dt_process_data = dt_process_data
#         self.data_buffer = np.zeros((1, n_channels, self.buffersize, 1)) # data buffer has shape (trials, n_channels, sampels, windows)
#         warnings.warn("This class is deprecated, use OnlineEEG now!")
        

#     def sendDetectedEventToAPI(self, timestamp_buffer_vals, local_clock_time, team_name = "example_team", secret_id = 5, url = 'http://10.250.223.221:5000/results'):
#         """
#         This function gathers all the relevant results and sends it to the host
#         This function should be called eveytime an error is detected

#         Parameters
#         ----------
#         timestamp_buffer_vals : Numpy array
#             subset of the timestamp_buffer array at the instant when you have predicted an error and want to send the current result. Basically the i-th element of the timestamp_buffer array
#         local_clock_time : float
#             current LSL local clock time when you have run your classifier and predicted an error. This can be determined with the help of "local_clock()" call.
#         team_name : str, optional
#             Each team will be assigned a team name, by default "example_team"
#         secret_id : int, optional
#             Each team will be provided with a secret code, by default 5
#         url : str, optional
#             The URL where the resulst are stored, by default 'http://10.250.223.221:5000/results'

#         Author
#         ------
#         Author : Niklas Kueper \n
#         Last changed: 05.02.2024 (by Niklas Kueper)
#         """        

#         # calculate the final values for the timings 
#         comm_delay = timestamp_buffer_vals[1] -timestamp_buffer_vals[0] -timestamp_buffer_vals[2]
#         computation_time = local_clock_time - timestamp_buffer_vals[1]

#         # connection to API for sending the results online 
        
#         myobj = {'team': team_name,
#                 'secret': secret_id,
#                 'host_timestamp': timestamp_buffer_vals[0], 
#                 'comp_time': computation_time, 
#                 'comm_delay': comm_delay}

#         x = requests.post(url, json = myobj)


#     def printStreamMetadata(self, stream_info_obj):
#         """
#         This function prints some basic meta data of the stream

#         Parameters
#         ----------
#         stream_info_obj : StreamInlet object
#             A pylsl StreamInlet object which contains alle the information of the stream

#         Author
#         ------
#         Author : Niklas Kueper \n
#         Last changed: 05.02.2024 (by Niklas Kueper)
#         """ 

#         print("") 
#         print("Meta data")
#         print("Name:", stream_info_obj.name())
#         print("Type:", stream_info_obj.type())
#         print("Number of channels:", stream_info_obj.channel_count())
#         print("Nominal sampling rate:", stream_info_obj.nominal_srate())
#         print("Channel format:",stream_info_obj.channel_format())
#         print("Source_id:",stream_info_obj.source_id())
#         print("Version:",stream_info_obj.version())
#         print("")


#     def updateBuffer(self, chunk, channel_indices = None, num_non_data_channels = 3):  #current_local_time, timestamp_offset, 
#         """
#         This function provides the most recent data samples and timestamps in a buffer (fist val is oldest, last the newest)

#         Parameters
#         ----------
#         chunk : list
#             Current data chunk with shape (samples, channels)
#         channel_indices : list, optional
#             If only selected channel indices should be extraced, by default None
#         check_sample_loss : bool, optional
#             If True the function is checkinf for sample losses, by default True

#         Author
#         ------
#         Author : Niklas Kueper \n
#         Last changed: 05.02.2024 (by Niklas Kueper)
#         """

#         #data 
        
#         current_chunk = (np.array(chunk).T) # chunk is sampels, channels, after transpose then channels, sampels !
        
#         #print("chunk shape", current_chunk.shape) # should be channels, sampels 

#         if(channel_indices): 
#             current_chunk = current_chunk[channel_indices, :]

#         # #print(current_chunk.shape)
#         # current_chunk = current_chunk[0:n_channels, :] # use first n channels

#         n_samples = current_chunk.shape[1] 
#         n_channels = current_chunk.shape[0] 

#         if (n_samples > self.data_buffer.shape[2]): # print error message 
#             print("Buffer overflow")

        
#         self.data_buffer = np.roll(self.data_buffer, shift = int(-1*n_samples), axis = 2) # shift array by n samples  data_buffer: shape (trials, channel, sampels, windows)
#         self.data_buffer[0, :, int(-1*n_samples):, 0] = current_chunk # channels, sampels shape , update latest values in buffer  --> is this correct 


#         # check for sample loss 
#         sample_indices = self.data_buffer[0, -3, :, 0].astype(int) # sample indice channel

#         for i in range(0, len(sample_indices) -1): 
#             if sample_indices[i] + 1 != sample_indices[i+1]:
#                 warnings.warn(f"Sample loss at {i}: {sample_indices[i:i+2]}")

#         data_windows = self.data_buffer[:, 0:n_channels-num_non_data_channels, :, :] # assuming last num_non_data_channels are appended at the end (as done by LiveAmp connector)

#         return data_windows# , timestamp_buffer

#     def getDataBuffer(self): 
#         """
#         This function returns the data_buffer

#         Returns
#         -------
#         Numpy array
#             data_butter : float

#         Author
#         ------
#         Author : Niklas Kueper \n
#         Last changed: 05.02.2024 (by Niklas Kueper)
#         """  

#         return self.data_buffer
    

#     def startZMQServer(self, port_name):
#         """
#         This function creats a socket connection as a pubisher to send commands

#         Parameters
#         ----------
#         port_name : str
#            The address string. This has the form "tcp://interface:port"

#         Returns
#         -------
#         Instance of class zmq socket
#             The instance (handle) of the created socket connection. 

#         Author
#         ------
#         Author : Niklas Kueper \n
#         Last changed: 05.02.2024 (by Niklas Kueper)
#         """ 
           
#         # create a socket connection as a publisher to send commands 
#         my_context = zmq.Context()
#         my_socket = my_context.socket(zmq.PUB)
#         my_socket.bind("tcp://*:"+port_name)
#         print("Publisher ready")
#         return my_socket

class EEGData:    
    """
    This class includes useful methods and paramters for the (pre)processing and viusalization of EEG data. It is mainly dependend on numpy, mne, scipy and additional utils (e.g. keras preprocessing)). 

    Parameters
    ----------
    format : str
        The data format of the dataset to be loaded, can be "Brainvision" (default), "NumpyEpochs", "RawObj", "Live" or "Recorded_LSL_stream".
    filenames : list of str
        A list of strings containing the filenames of the dataset(s) to be loaded.
    data_path : str
        The datapath as a string, please pass the path to the "data" directory (see readme of the library). 
    epochs : numpy ndarray, optional 
        The epochs to be set as numpy ndarray, only required when format type "NumpyEpochs" is selected.
    raw_obj :  mne raw object, optional
        The mne raw object that is used to create the object. Only required for format type "RawObj".
    f_samp: float
        The sampling rate of the data, not required for type "Brainvision". 
    channel_names : list of str
        A list of channel names as strings, if not known from the data format. Not required for format "Brainvision".
    windows : numpy ndarray, optional
        A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows), only required for format "Live". 
    data : numpy ndarray, optional 
        The channel wise (raw) data as numpy array (shape: n_channel, n_sampels), currently fully optional (not used by any format). 

    Attributes
    ----------
    raw_obj : instance of class Raw (mne) 
        The mne raw object to be used for full mne support, please see the mne wiki for further information. 
    __ch_names : list of str
        A list of channel names (strings) for each data channel. 
    __montage : mne montage object 
        A created montage object of an mne montage. 
    __fsamp : float 
        The sampling rate of the data in Hz. 
    time_axis_epochs: 1D numpy array 
        The time axis of the data after the epoching step. 
    units : str 
        The units of the data (i.e. "uV" or "V")
    epochs : numpy ndarray 
        The epoched data as numpy ndarray with shape: (n_trials, n_channels, n_sampels). 
    epoch_obj : instance of class Epochs (mne)
        The mne epochs object to be used for full mne support, see the mne wiki for more information. 
    obj_filtered : instance of class Raw (mne)
        A processed version of the mne raw object (deprecated, to be removed in the future)
    average_epochs : numpy ndarray 
        The averaged epochs as numpy ndarray with shape: (n_channels, n_sampels), (deprecated, to be removed in future). 
    events : numpy ndarray 
        A numpy ndarray containing events in the data as provided by mne.raw.get_events() method (has shape: (indizes, 0, event_number)). 
    windows : numpy ndarray 
        The windowed data as numpy array with shape: (n_trials, n_channels, n_sampels, n_windows). 
    window_names : list of str
        The names (indentifier) of each window as a list of strings (has same size as n_windows). 
    feature_vec : numpy ndarray 
        A feature vector as numpy array, can have different shapes depending on the later used ML model (see ML class). 
    calib_means : float 
        A calibration value containing the mean values of the data instance (e.g. of the training data). 
    calib_stds : float 
        A calibration value containing the standard deviation values of the data instance (e.g. of the training data). 

    Methods
    -------
    method1(arg1, arg2):
        A brief description of what the method does.
    method2(arg1, arg2):
        A brief description of what the method does.

    Author
    ------
    Author : Niklas Kueper \n
    Last changed: 09.01.2024 (by Niklas Kueper)
    """

    def __init__(self, format = "Brainvision", filenames = None, data_path = None, epochs = None, raw_obj = None, f_samp = None, channel_names = None, windows = None, data = None):
        """
        The constructor of the EEGData class

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 09.01.2024 (by Niklas Kueper)
        """        
       
        self.raw_obj = None
        # parameter 
        #basic params 

        self.__ch_names = channel_names
        self.__montage = None
        self.__fsamp = None
        self.time_axis_epochs = None
        self.units = None
        # epoching 
        self.epochs = None 
        self.epoch_obj =None
        self.obj_filtered = None
        self.average_epochs = None
        #events
        self.events = None
        # windowing 
        self.windows = None 
        self.window_names = None 
        # features 
        self.feature_vec = None
        self.calib_means = None
        self.calib_stds = None

        
        if(filenames and format == "Brainvision"): 
            #create numpy array with file names 
            data_str_arr = []
            for files_str in filenames: 
                data_str_arr.append(os.path.join(data_path, files_str)) 
            data_str_arr = np.array(data_str_arr)

            self.raw_obj = self.loadBrainproductsData(data_str_arr)

            # update parameter 
            #basic params 
            self.__ch_names = self.raw_obj.ch_names
            self.__fsamp = self.raw_obj.info['sfreq']
            self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 
            #events epochs_filter
            self.events, self.event_ids = mne.events_from_annotations(self.raw_obj)


        elif(format == "NumpyEpochs"): 
            self.epochs = epochs
            self.__fsamp = f_samp

        elif(format == "RawObj"): 
            self.raw_obj = raw_obj
            self.__fsamp = f_samp

        elif(format == "Live"): 
            self.windows = windows
            self.__fsamp = f_samp
            self.__ch_names = channel_names

        elif(format == "Recorded_LSL_stream"): 
            

            if(filenames): # implement running over all files and appending data to each other 

                if(len(filenames) > 1): 
                    concat_list = []
                    for filename in filenames: 
                        concat_list.append(np.load(data_path +filename+".npy"))
                    
                    data = np.concatenate(concat_list)
                else: 
                    data = np.load(data_path +filenames[0]+".npy")

            
            self.__fsamp = f_samp
            self.__ch_names = channel_names

            # create mne object 
            sfreq = self.__fsamp  # Sampling frequency
            data = data.T # in form (channel, sampels)
            #print("markers", np.where(data[-1, :] == 64)[0])
            times = np.arange(0, data.shape[1], 1/sfreq)  # 
            ch_types = ['eeg'] * len(self.__ch_names) # only EEG for now 
            ch_names = self.__ch_names
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            #scalings = {'eeg': 1}
            raw = mne.io.RawArray(data[0:len(ch_names), :], info) # only pass the actual EEG channel 
            self.raw_obj = raw

            # set annotation events (markers)
            event_channel = data[-1, :]
            marker_indices = np.where(event_channel > 0)[0]
            marker_numbers = event_channel[marker_indices]
            events = np.zeros((len(marker_indices), 3))
            events[:, 0] = marker_indices
            events[:, 2] = marker_numbers
            self.events = events.astype(int)

            annotations = mne.annotations_from_events(events = events, sfreq = self.__fsamp, event_desc=None, first_samp=0, orig_time=None, verbose=None)
            self.raw_obj.set_annotations(annotations = annotations)

            # print(type(self.events)) # (events, 3)
            # print("events", self.events)

        else: 
            print("No dataset specified ...")

    def mneRawToBrainvision(self, folder, filename, meas_date = None, resolution = 0.1, unit = "µV"):

        """
        This method can be used to store the data from an mne raw object in the brainvision format. 

        Parameters
        ----------
        folder : str
            The target folder where the data should be stored in the brainvision format, should be the path to the created data folder of the toolbox. 
        filename : str
            The filename of the data to be stored. 
        meas_date : datetime object, str, optional
            The measurement date on on which the data was recorded, by default None
        resolution : float, optional
            The resolution of the data in the measurement unit, by default 0.1
        unit : str, optional
            The units of the data to be stored as string, can be "uV" or "V", by default "µV"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """

        events=copy.deepcopy(self.events) 
        events_bv = events[:, [0, 2]]
        write_brainvision(data=self.raw_obj.get_data(), sfreq=self.__fsamp, ch_names=self.__ch_names, fname_base=filename, folder_out=folder, events=events_bv, meas_date = meas_date, resolution = resolution, unit = unit)
        print("stored data in brainvision format")

    def updatefromRawObject(self): 
        """
        This method can and should be used to update the class internal parameters and variables. Please make sure that this method is called whenever external methods (like mne raw methods) are executed to update the data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 12.11.2023 (by Niklas Kueper)
        
        """

        # update parameter 
        #basic params 
        self.__ch_names = self.raw_obj.ch_names
        self.__fsamp = self.raw_obj.info['sfreq']
        self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 
        #events epochs_filter
        self.events, self.event_ids = mne.events_from_annotations(self.raw_obj)

    def getRawObject(self):
        """
        This method can be used to get the current mne raw object. 

        Returns
        -------
        mne raw object
            The current mne raw object.  

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 12.11.2023 (by Niklas Kueper)
        """

        return self.raw_obj
    
    def getEpochs(self):
        """
        This method returns the epoched an possibly processed data as numpy array. 

        Returns
        -------
        time_axis_epochs : 1D numpy array 
            The time axis of the epochs as numpy array. 
        epochs : numpy ndarray 
            The numpy ndarray including the epoched data with shape: (n_trials, n_channels, n_sampels). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 12.11.2023 (by Niklas Kueper)
        """

        return self.time_axis_epochs, self.epochs
    
    def getEvents(self): 
        """
        This method returns the events (i.e. markers) as numpy array. The events can mark experimental events or stimuli or are used to help with evaluating the data. 

        Returns
        -------
        numpy ndarray
            The events as numpy ndarray with shape: (indices, 0, eventnumbers). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.events
        
    def getSamplingRate(self): 
        """
        This method returns the sampling rate of the data in Hz. 

        Returns
        -------
        float
            The sampling rate of the data as float value in Hz. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.__fsamp

    def getCalibStats(self): 
        """
        This function returns calibration parameters that where calculated for the dataset (see calcCalibStats for more information). 

        Returns
        -------
        tuple
            A tuple containing:
            calib_means : float
                The mean values for each channel as 1D numpy array. 
            calib_stds : float 
                The standard deviations for each channel as 1D numpy array. 
            calib_mins : float 
                The minimum values for each channel as 1D numpy array. 
            calib_maxs : float 
                The maximum values for each channel as 1D numpy array. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.calib_means, self.calib_stds, self.calib_mins, self.calib_maxs
    
    def setCalibStats(self, means, stds, mins, maxs): 
        """ 
        This method can be used to set some basic statistical parameters for the dataset (e.g. mean and std) for each data channel. Can be used for example to calibrate or normalize a train data set afte setting the parameters. 

        Parameters
        ----------
        means : numpy array
            A 1D numpy array containing the mean values of each channel. 
        stds : numpy array 
            A 1D numpy array containing the standard deviation values of each channel. 
        mins : numpy array 
            A 1D numpy array containing the minimum values of each channel. 
        maxs : numpy array 
            A 1D numpy array containing the maximum values of each channel. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        self.calib_means = means
        self.calib_stds = stds
        self.calib_mins = mins
        self.calib_maxs = maxs

    def setChannelNames(self, ch_names): 
        """
        This method can be used to set the channel names of the data if they were not specified before.

        Parameters
        ----------
        ch_names : list of str
            A list of channels names. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        self.__ch_names = list(ch_names)
        
        self.raw_obj.ch_names = list(ch_names)

    def updateChannelNamesRaw(self, new_channel_names): 

        
        def lookUpTableMethod(): 

            new_channel_names

        

    def getChannelNames(self): 
        """
        This method returns the channel names of the data. 

        Returns
        -------
        list of str
            A list of channel names. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.__ch_names

    def calcCalibStats(self, feature_times = None): 
        """
        This method can be used to calculate statistical parameters for the windowed data (training data) like mean and standard deviations for each channel. 

        Parameters
        ----------
        feature_times : 1D numpy array or list, optional
            A numpy array containing 2 values, which define the start and endpoint (times) of the data in the window to be selected for the calculation procedure, by default None
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 01.10.2023 (by Niklas Kueper)
        """

        # shape: trials, channels, sampels, windows
        calib_means = np.zeros(self.windows.shape[1]) # channel wise 
        calib_stds = np.zeros(self.windows.shape[1])
        calib_maxs = np.zeros(self.windows.shape[1])
        calib_mins = np.zeros(self.windows.shape[1])

        if(feature_times): 
            start_point_index = int((feature_times[0]/1000)*self.__fsamp)
            stop_point_index = int((feature_times[1]/1000)*self.__fsamp)

        for channel_idx in range(0, self.windows.shape[1]):   # channel wise 
            if(feature_times):
                channel_data = self.windows[:, channel_idx, start_point_index:stop_point_index, :].flatten() # use only the features time points for window norm 
            else: 
                channel_data = self.windows[:, channel_idx, :, :].flatten() # use hole window 

            # calc descriptive values 
            mean = (np.max(channel_data) +np.min(channel_data))/2 
            std = np.std(channel_data)
            max = np.max(channel_data)
            min = np.min(channel_data)
            # calib values 
            calib_means[channel_idx] = mean
            calib_stds[channel_idx] = std
            calib_maxs[channel_idx] = max 
            calib_mins[channel_idx] = min
        # set the matrices for calibration 
        self.calib_means = calib_means
        self.calib_stds = calib_stds
        self.calib_mins = calib_mins
        self.calib_maxs = calib_maxs

    
    # def convertEpochsToMicrovolts(self): 
        
    #     self.epochs = self.epochs*1000000.0 # not in uV

    # def icaEOGArtifactRemoval(self, raw, eeg_epochs, n_components = 20, drop_epochs = False, threshhold = 0.05, ch_names = ["FP1", "FP2"], plot_steps = False, baseline = (None, -0.2)):

    #     """
    #     This funcion automatically detects EOG artifacts in the given rereferenced eeg-signal with an ICA. The decisive components are marked and removed from the signal

    #     Arguments
    #         raw: The created mne object
    #         eeg_epochs: The rereferenced epoched mne object
    #         n_components: Number of principal components that are passed to the ICA algorithm during fitting:
    #             var1: Give an int which must be greater than 1 and less than or equal to the number of channels.
    #             var2: Give a float between 0 and 1, this will select the smallest number of components required to explain the cumulative variance of the data greater than n_components
    #         drop_epochs: If TRUE a thrshhold determins the dropping of bad epochs based on peak to peak value
    #         ch_names: The channels that are showing EOG artifacts
    #         plot_steps: If TRUE the steps of the ICA will be plotted

    #     Returns
    #         eog_removed: The processed eeg data as an instance of mne object


    #     Meta information: 
    #     Author: Patrick Bings
    #     Last changed: 07.09.2023

    #     """

    #     # dropping bad epochs based on peak to peak value if True
    #     if(drop_epochs):
    #         self.epochs.drop_bad(reject = {'eeg': threshhold})

    #     # epoched_eeg_rereferenced.plot(block=True)

    #     if(plot_steps == False):
    #         # creating epochs around the EOG-artifacts
    #         eog_evoked = create_eog_epochs(self.raw_obj, ch_name=ch_names).average()
    #         eog_evoked.apply_baseline(baseline=baseline)

    #         # creating the ICA and fitting it to the epoched raw data: 
    #         ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
    #         ica.fit(epoched_eeg_rereferenced)

    #         # exclude the right ICs
    #         ica.exclude = []

    #         # find which ICs match the EOG pattern
    #         eog_indices, eog_scores = ica.find_bads_eog(epoched_eeg_rereferenced, ch_name=ch_names)

    #         ica.exclude.extend(eog_indices)

    #         eog_removed = ica.apply(epoched_eeg_rereferenced)
    #     else:
    #         # create an acticap montage  --> why ? 
    #         # acticap_montage = createActicapMontage(plot_montage, rename_channels)
    #         # epoched_eeg_rereferenced.set_montage(acticap_montage) # set created montage 

    #         # creating epochs around the EOG-artifacts
    #         eog_evoked = create_eog_epochs(raw, ch_name=ch_names).average()
    #         eog_evoked.apply_baseline(baseline=(None, -0.2))
    #         #eog_evoked.plot_joint()

    #         # creating the ICA and fitting it to the epoched raw data: 
    #         ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
    #         ica.fit(self.epoch_obj) 
    #         ica.plot_components()

    #         # exclude the right ICs
    #         ica.exclude = []

    #         # find which ICs match the EOG pattern
    #         eog_indices, eog_scores = ica.find_bads_eog(epoched_eeg_rereferenced, ch_name=ch_names)

    #         ica.plot_overlay(eog_evoked, exclude=eog_indices, show=False)

    #         ica.exclude.extend(eog_indices)

    #         print(ica.exclude)

    #         # barplot of ICA component "EOG match" scores
    #         ica.plot_scores(eog_scores)

    #         # plot diagnostics
    #         ica.plot_properties(epoched_eeg_rereferenced, picks=eog_indices)

    #         #plot ICs applied to raw data, with EOG matches highlighted
    #         ica.plot_sources(epoched_eeg_rereferenced, show_scrollbars=False)

    #         # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    #         ica.plot_sources(eog_evoked)

    #         eog_removed = ica.apply(epoched_eeg_rereferenced)

    #     return eog_removed

    def simpleICAFiltering(self, n_components = 20, exclude_components = [0, 1]): 
        """
        This method can be used to fit and apply a ICA on epoched data (on mne epochs object). 

        Parameters
        ----------
        n_components : int, optional
            The number of components of the ICA, by default 20
        exclude_components : list of int, optional
            The components of the ICA to be excluded from the data. Most probably the first channels referring to eyeblinks or saccades, by default [0, 1]
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 01.10.2023 (by Niklas Kueper)
        """

        ica = ICA(n_components=n_components) 
        ica.fit(self.epoch_obj)
        ica.apply(self.epoch_obj, exclude = exclude_components)
        self.epochs = self.epoch_obj.get_data()
        

    def loadBrainproductsData(self, dataset_list): 
        """
        This method is used to load a dataset in the Brainvision format (.eeg, .vhdr, .vmrk). 

        Parameters
        ----------
        dataset_list : list of str
            A list of filenames to be loaded. If the list contains more than one filename, the datasets are concatenated. 

        Returns
        -------
        mne raw object
            The loaded data as mne raw object. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 06.07.2022 (by Niklas Kueper)
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

    # not required anymore 
    # def filterRaw(self, f_highpass, f_lowpass, picks=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=None, method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=None): 
    #     self.raw_obj.filter(f_highpass, f_lowpass, picks=picks, filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs, method=method, iir_params=iir_params, phase=phase, fir_window=fir_window, fir_design=fir_design, skip_by_annotation=skip_by_annotation, pad=pad, verbose= verbose)

    # def show(self, scalings_dict = None): 
    #     self.raw_obj.plot(scalings = scalings_dict)

    def mneRawMethod(self, method_name = None, **kwargs): 
        """
        Apply a mne raw objects method and update the parameters of the current object. This method should be used instead of writing new methods that just execute an existing method of an mne raw object (like raw.show()). 
        Please the the wiki of mne for further informations about the methods. 

        Parameters
        ----------
        method_name : str, optional
            The name of the method to be executed passed as string, by default None
        **kwargs : optional
            The arguments passed to the method (e.g. plot = True, array = [0, 1, 2])
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.12.2023 (by Niklas Kueper)
        """

        if hasattr(self.raw_obj, method_name) and callable(getattr(self.raw_obj, method_name)):
            method = getattr(self.raw_obj, method_name)
            method(**kwargs) # call method with params 
        else:
            print("mne raw object does not have the expected method.")
        
        self.updatefromRawObject() # update class internal params from mne raw object 
        
    def designFilter(self, f_low = None, f_high= None, order = 2, filter_type = "scipy_butter", Q = 30, show_response = False, alpha = 0.98, return_type = "ba"): 
        """
        Design a digital fir or iir filter (notch, bandpass, highpass or lowpass). 

        Parameters
        ----------
        f_low : float, optional
            The lowpass frequency of the filter (the high frequency boundary), by default None
        f_high : float, optional
            The highpass frequency of the filter (the low frequency boundary), by default None
        order : int, optional
            The order of the filter, by default 2
        filter_type : str, optional
            The type of the filter as string (common biosignal filters), can be "scipy_butter", "scipy_bessel", "dc_notch", "dc_removal" or "allpass", by default "scipy_butter"
        Q : int, optional
            The sharpness factor of a notch filter (only required for "dc_notch"), see more information about scipys iirnotch method for further details, by default 30
        show_response : bool, optional
            If True, the filter response is shown (frequency and phase response), by default False
        alpha : float, optional
            The alpha value (sharpness) of the dc removal filter (only required for "dc_removal" filter), by default 0.98
        return_type : str, optional
            The type of the filter coefficients to be returned, can be "ba" or "sos". "sos should be preferred to avoid instabilites, but not all filtering methods support the "sos" type, by default "ba"

        Returns
        -------
        tuple of numpy arrays or numpy array 
            The returned filter parameters (ba or sos). 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.12.2023 (by Niklas Kueper)
        """
        
        # calc individual coeffs 
        if(filter_type == "dc_notch"):
            b, a = sig.iirnotch(f_high, Q, fs=self.__fsamp)

        if(filter_type == "scipy_butter"): # prefer this one 
            if(f_high and f_low): 
                b, a = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='butter', output='ba', fs=self.__fsamp)
                sos = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='butter', output='sos', fs=self.__fsamp)
            elif(f_high):
                sos = sig.iirfilter(order, f_high, btype='highpass', ftype='butter', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
            elif(f_low): 
                sos = sig.iirfilter(order, f_low, btype='lowpass', ftype='butter', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)

        if(filter_type == "scipy_bessel"): # prefer this one 
            if(f_high and f_low): 
                sos = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='bessel', output='sos', fs=self.__fsamp)

            elif(f_high):
                sos= sig.iirfilter(order, f_high, btype='highpass', ftype='bessel', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
            elif(f_low): 
                sos = sig.iirfilter(order, f_low, btype='lowpass', ftype='bessel', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)

        if(filter_type == "dc_removal"): 
            a = [1, -1 * alpha]
            b = [1, -1]
        
        if(filter_type == "allpass"):
            scale = 0.8
            b = [ 1 *scale, -alpha *scale] 
            a = [-alpha*scale, 1*scale]

        if(show_response): 
            w, h = sig.freqz(b, a, worN=2024)
            plt.subplot(2, 1, 1)

            x = (w/np.pi)*(self.__fsamp/2)

            db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
            plt.plot(x, db)
            plt.ylim(-75, 5)
            plt.grid(True)
            plt.yticks([0, -20, -40, -60])
            plt.ylabel('Gain [dB]')
            plt.title('Frequency Response')
            plt.subplot(2, 1, 2)
            plt.plot(x, (np.angle(h)))
            plt.grid(True)
            plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            plt.ylabel('Phase [rad]')
            plt.xlabel('Frequency[Hz]')
            plt.show()

            # group delay 
            w, gd = sig.group_delay((b, a), fs = self.__fsamp)
            plt.title('Digital filter group delay')
            plt.plot(w, gd)
            plt.ylabel('Group delay [samples]')
            plt.xlabel('Frequency [Hz]')
            plt.show()

        # return filter coeffs 
        if(return_type == "ba"): 
            return b, a
        
        elif(return_type == "sos"): 
            return sos 

    def filterWindows(self, b = None, a = [1], sos = None, apply_method = "zero_phase_sos", mne_filter_type = None, f_high = None, f_low = None, order = None, fir_design = None): # under change 
        """
        Apply a designed digital filter to the windowed data (window wise for each channel). Please be careful in selection appropriately designed filters, especially because they are applied on small data chunks (windows)!
        Therefore, consider that artifacts might occur depending on the selected method and parameters. 

        Parameters
        ----------
        b : 1D numpy array, optional
            The filter coefficients (numerator), by default None
        a : 1D numpy array, optional
            The filter coefficients (denominator), by default [1] (fir)
        sos : array_like, optional
            The filter coefficients in sos format, by default None
        apply_method : str, optional
            The filtering method to be applied, can be "zero_phase_sos", "gustav", "zero_phase_ba" or "forward_filter". Please have a look at the documentation of the filter implemetations for more information (i.e. scipy and mne docu), by default "zero_phase_sos"
        mne_filter_type : str, optional
            The type of the mne filter to be applied, can be "mne_fir", "mne_iir", only required when using an mne filter , by default None
        f_high : float, optional
            The highpass filter frequency when applying the mne filter, by default None
        f_low : float, optional
            The lowpass filter frequency when applying the mne filter, by default None, by default None
        order : int, optional
            The order of the mne filter to be applied, only required for mne filter, by default None
        fir_design : str, optional
            The fir design method as string for mne filters (e.g. "hamming", see mne documentation for more information), by default None
                
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.12.2023 (by Niklas Kueper)
        """
        

        # shape: trials, channels, sampels, windows
        
        for trial_idx in range(0, self.windows.shape[0]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    
                    current_wind = self.windows[trial_idx, :, :, window_idx]

                    # currently the best 
                    if(mne_filter_type == "mne_fir" or mne_filter_type == "mne_iir"): # TODO: remove this filter here and add a new filtering method maybe ? Paramters do not fit 
                        if(mne_filter_type == "mne_iir"): 
                            method = "iir"
                        else: 
                            method = "fir"

                        for channel_idx in range(0, self.windows.shape[1]): # TODO: replace the function call by the parser method for mne objects 
                            #print(current_wind.shape)
                            filtered_window= mne.filter.filter_data(current_wind[channel_idx, :], sfreq = self.__fsamp, l_freq =f_high , h_freq = f_low, filter_length=order, method = method, fir_design = fir_design, verbose = "CRITICAL") # pad = "symmetric"
                            self.windows[trial_idx, :, :, window_idx] = filtered_window

                    else: 
                        for channel_idx in range(0, self.windows.shape[1]):
                            
                            # perform zero phase forward backward filtering with gustafson method to reduce artifacts  
                            if(apply_method == "gustav"): 
                                filtered_window = sig.filtfilt(b, a, current_wind[channel_idx, :].copy(), method ="gust") # forward backward filtering with gustafson method
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window

                            #filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy())#, method ="gust") # forward backward filtering with gustafson method 
                            elif(apply_method == "zero_phase_sos"): 
                                filtered_window = sig.sosfiltfilt(sos, current_wind[channel_idx, :], padlen = len(current_wind[channel_idx, :])-1, padtype ="even") # normal filtering with padding 
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window

                            elif(apply_method == "zero_phase_ba"): 
                                filtered_window = sig.filtfilt(b, a, current_wind[channel_idx, :], padlen = len(current_wind[channel_idx, :])-1, padtype ="even") # normal filtering with padding
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window

                            elif(apply_method == "forward_filter"): 
                                filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy())
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window
   

                    # elif(filter_type == "fft_bandpass"):  # fft bandpass implementation from pySPACE, check this again  --> not usable anymore 

                    #     filtered_window = np.zeros(current_wind.shape)
                    #     for channel_idx in range(0, self.windows.shape[1]): 

                    #         n = len(current_wind[channel_idx, :])

                    #         res = 0.1 # fixed resolution to 0.1 Hz 
                    #         fourier_transformed = scipy.fftpack.fft(current_wind[channel_idx, :], n = int(self.__fsamp/res)) # increase resolution to 0.1 Hz 

                    #         #Compute the pass band indices
                    #         lower_bound = int(round(float(f_high) / (self.__fsamp) * len(fourier_transformed)))
                    #         upper_bound = int(round(float(f_low) / (self.__fsamp) * len(fourier_transformed)))
                            

                    #         #Setting frequencies outside the pass band to 0
                    #         for i in range(0, lower_bound):
                    #             fourier_transformed[i] = 0
                    #             fourier_transformed[-i-1] = 0

                    #         for i in range(upper_bound,len(fourier_transformed)//2):
                    #             fourier_transformed[i] = 0
                    #             fourier_transformed[-i-1] = 0
                            
                    #         inverse = scipy.fftpack.ifft(fourier_transformed, n = int(self.__fsamp)) # go back to normal samp rate size 

                    #         #Inverse Fourier transform and project to real component
                    #         self.windows[trial_idx, channel_idx, :, window_idx] = inverse


    def minMaxNormWindows(self): 
        
        normed_wind = np.zeros(self.windows.shape)
        for channel in range(0, self.windows.shape[0]): 
            wind_channel = self.windows[channel, :] + (-1 *np.min(self.windows[channel, :]))
            wind_channel = wind_channel / np.max(wind_channel)
            normed_wind[channel, :] = wind_channel


        for trial_idx in range(0, self.windows.shape[0]): 
            for window_idx in range(0, self.windows.shape[3]): 
                for channel_idx in range(0, self.windows.shape[1]): 
                            
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])
                    current_wind = current_wind + (-1 *np.min(current_wind))
                    current_wind = current_wind / np.max(current_wind)
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind

    def cutWindows(self, n_samples_start = 25, n_samples_end = 25): 
        """
        This function can be used to cut the length of the windowed data (sample dimension).

        Parameters
        ----------
        n_samples_start : int
            The number of samples to cut at the start of the windows.
        n_samples_end : int
            The number of samples to cut at the end of the windows.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 24.11.2023 (by Niklas Kueper)
        """

        if(self.windows.shape[2] < n_samples_start+n_samples_end):   # trials, channels, sampels, windows 
            raise Exception("number of cutted samples exceeds window length, not performing the cutting ... ")
        else: 
            self.windows = self.windows[:, :, int(n_samples_start):int(-1*n_samples_end), :] 


    def createActicapMontage(self, plot_montage = False, rename_channels=None, set_montage = True): 

        """
        This function can be used to create an acticap montage (used by e.g. LiveAmp64). The montage was created based on the acticap manual and an easycap template provided by mne.

        Parameters
        ----------
        plot_montage : bool
            A boolean flag if the montage info should be shown or not. if set to True, the montage will be shown.

        Returns
        -------
        montage : mne.channels.Montage
            An mne montage object, that was created for the acticap layout.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2023 (by Niklas Kueper)
        """
        
        montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095) 
        #show easy cap Montage  

        if (rename_channels):
            montage.rename_channels({'Cz' : 'CZ','Pz' : 'PZ','Fz' : 'FZ','CPz' : 'CPZ', 'Fp1': 'FP1','Fp2': 'FP2','Oz': 'OZ','POz': 'POZ'}, allow_duplicates=False) # maybe change this ? 
        
        exclude_list = np.array(['O10','Fpz', 'Iz', 'F9', 'F10', 'P9', 'P10', 'O9', 'FCz', 'AFz']) # may change this ? 
        
        # get params of structure 
        easy_cap_ch_names = montage.ch_names
        easy_cap_dig = montage.dig
        #dev_head = montage.dev_head_t

        # find indizes to remove channels not in ActiCap 
        indizes_to_removing_channel = []
        for index in range(0, len(exclude_list)):
            for index1 in range(0, len(easy_cap_ch_names)):
                a = exclude_list[index]
                b = easy_cap_ch_names[index1]

                if(a == b):
                    indizes_to_removing_channel.append(index1)

        # seperate Standard digs and EEG digs 
        easy_cap_dig_standard = easy_cap_dig[0:3]
        easy_cap_dig_eeg = easy_cap_dig[3:]

        #delete Channels not there for acticap
        indizes_to_removing_channel_sorted = sorted(indizes_to_removing_channel, reverse= True)
        for indizes in indizes_to_removing_channel_sorted:
            #print('Indizes', indizes-count)
            del easy_cap_ch_names[indizes]
            del easy_cap_dig_eeg[indizes]

        # Adapted channel names and digitazation for acticap montage 
        easy_cap_dig_adapted = easy_cap_dig_standard.copy() # only head digits etc. 
        easy_cap_dig_adapted.extend(easy_cap_dig_eeg) # append selected channel digits 
        easy_cap_ch_names_adapted = easy_cap_ch_names.copy()

        #create Montage 
        #PlotMontage = True
        acti_cap_montage = mne.channels.DigMontage(dig=easy_cap_dig_adapted, ch_names=easy_cap_ch_names_adapted)
        if(plot_montage == True): 
            acti_cap_montage.plot()
            plt.show()
        
        self.__montage = acti_cap_montage

        if(set_montage): 
            self.obj_filtered.set_montage(self.__montage)


    def topoplot(self, times, title_str = "Topoplot at selected times", min_val = -6e-06, max_val = 6e-06):
                 
        """
        This method creates a topoplot at different times in relation to an specific event. 
    
        Parameters
        ----------
        times : list of int
            The times for which the topoplot should be created (in ms). 
        title_str : str, optional
            The title of the topoplot, by default "Topoplot at selected times"
        min_val : float, optional
            The minimum value of the colorbar in the units of the data, by default -6e-06
        max_val : float, optional
            The maximum value of the colorbar in the units of the data, by default 6e-06
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2023 (by Niklas Kueper)
        """
        
        mean_epochs = np.mean(self.epochs, axis = 0)

        #topoplot at different times 
        n,m = mean_epochs.shape
        #start_idx = (start_time/1000)*f_samp_eeg
        #step_idx = (step_time/1000)* f_samp_eeg
        time_idx = (np.array(times)).astype(int)
        time_axis_eeg_epoch_ms = (self.time_axis_epochs *1000).astype(int) # make time axis in ms for the analysis 


        #indices_of_topoplot = np.arange(start_idx, m, step = step_idx).astype(int) # 22 er steps 
        mean_epochs.astype(float)
        
        count = 0
        fig, ax = plt.subplots(nrows=len(time_idx), figsize=(8, 20), gridspec_kw=dict(top=0.9),sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.5)
        for t_index in time_idx: 
            index = np.array(np.where(time_axis_eeg_epoch_ms == t_index))
            if (index.size == 0): # index not found 
                index = np.array(np.where(time_axis_eeg_epoch_ms == t_index+1))
            index = index[0][0] # numpy array to int value 
            cmap = 'bwr'
            im, cn = mne.viz.plot_topomap(mean_epochs[:,index], self.obj_filtered.info, cmap = cmap, axes = ax[count], show = False, image_interp = 'cubic',extrapolate='local',vlim = [min_val, max_val])
            str_time = str(t_index)+" ms"
            ax[count].set_title(title_str+str_time, color='black', fontsize=12)
            cbar =fig.colorbar(im, ax = ax[count], orientation="vertical", pad = 0.15)
            cbar.set_label("in uV")
            count = count+1
        plt.show()

    def getDataFromChannels(self, channel_names, average = True, windowed_data = False, epoched_data = False): 
        
        """
        This function returns data from the channels

        Parameters
        ----------
        channel_names : str
            The channel names (list) of the data tha should be returned
        average : bool, optional
            If True the data should be averaged after windowing of epoching(deprecated, averaged should be implemented seperately), by default True
        windowed_data : bool, optional
            If True the windowed data of the specified channels is returned, by default False
        epoched_data : bool, optional
            If True the epoched data of the specified channels is returned, by default False

        Returns
        -------
        2D-numpy array
            data_channels
                Numpy array with the data from the selected channels. Can be either in the format of windowd data (if windowed_data = True), epoched data (if epoched_data = True or raw data in format (channels, samples)) 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)
        """                
        
        if(len(channel_names) > 1): # for more then one channel
            channel_idxs = []
            for channel_name in channel_names: 
                channel_idx = self.__ch_names.index(channel_name)
                channel_idxs.append(channel_idx)
        else: 
            channel_idxs = self.__ch_names.index(channel_names[0])


        if (windowed_data): 
            # shape: trials, channel, sampels, windows 

            if(average): # if average should be returned 
                
                return np.mean(self.windows[:, channel_idxs, :, :], axis = 0)
            else: 
                return self.windows[:, channel_idxs, :, :]

        elif(epoched_data): # data is epoched  

            if(average):
                data_channel = self.average_epochs[channel_idxs, :] # data channel with shape: (channels, sampels) for average 
            else: 
                data_channel = self.epochs[:, channel_idxs, :] # data channel with shape: (trials, channels, sampels)  

            return data_channel
        
        else: # return data from raw object 
            return self.raw_obj.get_data(picks=channel_names) # format is channels, sampels (numpy array shape)


    def getKerasPredictionResultsLRP(self, model, epochs, n_samp_features):  #TODO: Move to ml_lib or remove duplicate 
        """
        Get the prediction results of the classifier after predicting on the test data (scores and predicted labels). 

        Parameters
        ----------
        model : The machine learning model instance, can be for example a keras or sklearn model. 
            The keras model object
        epochs : Numpy array
            The EEG-epochs as numpy array with shape (n_epochs, n_ channels, n_samples)
        n_samp_features : int
            Number of samples that are used as features for both classes, currentsl the last n_samp_features datapoints are considered as erp class and dht remaining ar from noerp class

        Returns
        -------
        tuple
            predicted_lables : Numpy array
                The predicted labels of the classifier as float values (0.0 noerp or 1.0 erp)
            true_labels : Numpy array
                The true labels in respect to the number of n_samp_features as erp labels (-n_samp_features to time 0 as erp labeled points)
            trial_prediction : Numpy array
                The prediction scores of each classified datapoint

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)

        """

        true_labels = []
        predicted_labels = []
        prediction_scores = []

        for trials in epochs: 
            # true labels single trial 
            erp_labels = np.zeros((trials.shape[1]))
            erp_labels[-n_samp_features:] = 1.0
            true_labels.append(erp_labels[:,])
            # single trial predictions 
            trial_prediction = model.predict(trials.T)
            trial_label = [0 if score <0.5 else 1 for score in trial_prediction]
            trial_label = np.array(trial_label)
            predicted_labels.append(trial_label)
            prediction_scores.append(trial_prediction)
            
        #flatten the trial labels
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        prediction_scores = np.array(prediction_scores)

        return predicted_labels, true_labels, prediction_scores


    def calcMovingAveragePredictionScores(self, trial_prediction_test, n_samp = 20): # 
        """
        This function calculates the moving average prediction scores

        Parameters
        ----------
        trial_prediction_test : numpy Ndarray 
            The numpy array with shape (trials, channel, sampels). 
        n_samp : int,  by default 20
            The length of the moving average window (samples over with the average is calculated). 

        Returns
        -------
        Numpy array
            processed_trial_predictions : float
                A numpy ndarray containing the postprocessed prediction scores (same shape as trial_prediction_test (input)). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """   

        processed_trial_predictions = np.zeros(trial_prediction_test.shape)

        trial_idx = 0
        for trial in trial_prediction_test: 
            for index in range(0, len(trial)): 
                if(index < n_samp): 
                    processed_trial_predictions[trial_idx, index] = 0 # what to to when buffer not full ? 
                else: 
                    processed_trial_predictions[trial_idx, index] = np.mean(trial[index-n_samp:index])

            trial_idx = trial_idx +1

        return processed_trial_predictions
    

    def rereferencingEpoching(self, marker_number, error_number, channel_list, inverse_keep_channel, t1, t2, reref_channels = [], apply_filter=False, f_highpass = None, f_lowpass= None, apply_baseline_correction = False,  t0_baseline = None, t1_baseline= None, apply_ica = False, n_ica_comp = 20, exclude_ica_comp = [0, 1]): 
        """
        Apply referencing and epoching with given filters to a raw_obj mne instance

        Parameters
        ----------
        marker_number
            The markernumber of the used event for creating the epochs
        error_number
            The markernumber of the trials with an error which are excluded from the evaluation
        channel_list
            As list of EEG-channels that are either kept or dropped from evaluation depending on the "inverse_keep_channel" flag
        inverse_keep_channel
            If False, all channel in "channel_list" are kept, otherwise the specified channels are dropped
        t1
            Start time of the epocs in ms
        t2
            End time of the epochs in ms
        reref_channels, optional
            A list of channels that are used for rereferencing. If the list is empty, the original ref-channel is used, if ["average"] an average an average reference is applied , by default []
        apply_filter, optional
            Boolean flag that should be True if a filter should be applied, by default False
        f_highpass, optional
            The highpass cutoff frequency in Hz (only used when apply_filter is True), by default None
        f_lowpass, optional
            The lowpass cutoff frequency in Hz (only used when apply_filter is True), by default None
        apply_baseline_correction, optional
            Boolean flag that is set to True if baseline correction should be applied (mean value between t0_baseline and t1_baseline is used as correction), by default False
        t0_baseline, optional
            Specified start time for the baseline correction, by default None
        t1_baseline, optional
            Specified end time for the baseline correction, by default None
        apply_ica, optional
            Boolean flag that should be True if a simple ICA removal should be applied, by default False
        n_ica_comp, optional
            Number of components the ICA should calculate, by default 20
        exclude_ica_comp, optional
            The components that are excluded if the ICA flag is True, by default [0, 1]
            
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)  

        """        
        
        # rereferencing 
        rereferenced_eeg_raw_obj = self.raw_obj.copy()
        event_id_used = marker_number
        
        if not (reref_channels):
            print("no reref channels specified, using original ref")
            print("")
        else: 
            if (reref_channels[0] == "average"):
                print("using average reference over all electrodes")
                print("")
                rereferenced_eeg_raw_obj, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw_obj, ref_channels='average' ,copy=True)
            else:
                print("using custom electrodes for rereferencing")
                rereferenced_eeg_raw_obj, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw_obj, ref_channels=reref_channels ,copy=True)
        
        raw_obj_eeg_rereferenced = rereferenced_eeg_raw_obj.copy()
        #drop channel

        #apply filter 
        if (apply_filter): 
            filtered_eeg_rereferenced = raw_obj_eeg_rereferenced.filter(f_highpass,f_lowpass)
        else: 
            filtered_eeg_rereferenced = raw_obj_eeg_rereferenced

        # if ica should be used 
        if(apply_ica): 
            ica = ICA(n_components=n_ica_comp) 
            ica.fit(rereferenced_eeg_raw_obj)
            ica.apply(rereferenced_eeg_raw_obj, exclude = exclude_ica_comp)

        
        #extract events
        include_events = self.events
        plot_onset_indices = np.where(include_events[:,2] == marker_number)[0] # S100 marker is leaving plate 
        exclude_indices = np.where(include_events[:,2] == error_number)[0] # S3 marker should be excluded 
        
        #plot_onset_indices = plot_onset_indices[0:-1] # cut of last movement , might be after experiment --> not used anymore, leads to confusion ! 
        include_mask = np.ones(plot_onset_indices.shape)

        #search for correct indizes without S3 errors 
        for i in range(0, len(plot_onset_indices)): 
            for j in range(0, len(exclude_indices)):
                if ((exclude_indices[j]-1) == plot_onset_indices[i] or (exclude_indices[j]-2) == plot_onset_indices[i]):
                    include_mask[i] = 0

        #Epochs with an S3 error are excluded 
        plot_onset_indices_correct = plot_onset_indices[include_mask.astype(bool)]

        #create new event matrix 
        used_include_events = np.zeros((len(plot_onset_indices_correct),3))
        used_include_events = include_events[plot_onset_indices_correct,:]

        #eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events, event_id = event_id_used,tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
        if not(channel_list): 
            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,event_id = marker_number, tmin=t1, baseline=(t0_baseline, t1_baseline), tmax=t2, preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events, event_id = marker_number, tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
                
        else: #drop specified channels if False 

            # keep or drop specified channels 
            if(inverse_keep_channel == True): 
                filtered_eeg_rereferenced.drop_channels(channel_list)
            else: 
                filtered_eeg_rereferenced.pick_channels(channel_list)

            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,tmin=t1,event_id = marker_number, tmax=t2, baseline=(t0_baseline, t1_baseline), preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,tmin=t1, event_id = marker_number, tmax=t2, baseline=None, preload=True, reject_by_annotation = True)
        
        # Get remaining channel names  
        self.__ch_names = filtered_eeg_rereferenced.ch_names
        self.obj_filtered = filtered_eeg_rereferenced.copy()
        
        self.epoch_obj = eeg_epochs.copy() # the object of epochs from mne 
        
        #get data out as numpy array for further processing 
        self.epochs = eeg_epochs.get_data()#units = "uV") 
        self.average_epochs = np.mean(self.epochs, axis = 0)
        self.event_id = event_id_used 
        
        #generate a time axis for the epochs 
        self.time_axis_epochs = np.arange(t1,t2+1/self.__fsamp, step = 1/self.__fsamp) #build time axis (epoch)


    def getTimeAxisEpochs(self): 
        """
        This method returns the calculated time axis for the epoched data as 1D numpy array. 
        
        Returns
        -------
        numpy array (1D)
            The time axis as 1D numpy array containing the time axis values (in seconds) in relation to the event used for epoching. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 09.01.2024 (by Niklas Kueper)
        """

        if (self.epochs is None): 
            warnings.warn("Data not epoched yet, please do this before using this method, terminating ...")
        else: # if epochs exist already  
            return self.time_axis_epochs
        

    def splitTrainTestEpochs(self, n_test_epochs = 5):
        """
        Split the epochs (data type epochs) into two seperate epoched data instances. This can be used for example to split the epochs into training and testing epochs as suggested by the name. 

        Parameters
        ----------
        n_test_epochs : int, optional
            The number of epochs to be splitted (e.g. for testing a classifier), by default 5

        Returns
        -------
        Tuple of data objects of class EEGData (train and test)
            The returned data objects of class EEGData, each including the specified number of trials. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        epochs_train = self.epochs[0:-n_test_epochs, :, :] 
        epochs_test = self.epochs[-n_test_epochs:, :, :]

        print("train shape", epochs_train.shape)
        print("test shape", epochs_test.shape)

        # copy objects 
        data_obj_train = copy.deepcopy(self)
        data_obj_test = copy.deepcopy(self)

        # epochs returning 
        data_obj_train.epochs = epochs_train
        data_obj_test.epochs = epochs_test

        return data_obj_train, data_obj_test


    def timeShiftingLinearSpatialFilter(self, erp_value_type = "min", replace_epochs = False, max_sample_diff = 200): 
        """
        A linear spatial filter with one remaining channel that additionally applies a timeshift to each individual channel according to the a minimum or maximum ERP magnitude. 

        Parameters
        ----------
        erp_value_type : str, optional
            The erp value type for calculating the reference channel for the time shift. Can be "min" or "max" in order to find the channel with the maximum or minimum value for further calculations, by default "min"
        replace_epochs : bool, optional
            A boolean flag indicating if the current epochs for the calculation should be replaced, by default False
        max_sample_diff : int, optional
            The maximum sample difference between channels that is still allowed for applying the time shift. Possible time shifts above this value will not be considered by setting it to the maximum value, by default 200

        Returns
        -------
        Numpy ndarray 
            A numpy array with the shape of epochs (trials, channels, sampels) after applying the spatial filter. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
        
        if(erp_value_type == "min"): 
    
            min_vals = np.min(self.average_epochs, axis = 1)
            min_indices = np.argmin(self.average_epochs, axis = 1)

            source_index = min_indices[np.argmin(min_vals)]
            
            indices_diffs = min_indices - source_index

            coefficients = min_vals *(-1)
            norm_coefficients =  (coefficients/np.sum(coefficients))
            weighted_epochs = np.zeros(self.epochs.shape)

            resulting_channel_epochs = 0
            
            for trial_idx in range(0, self.epochs.shape[0]): 
                for channel_idx in range(0, self.epochs.shape[1]): 
                    weighted_epochs[trial_idx, channel_idx, :] = self.epochs[trial_idx, channel_idx, :] * norm_coefficients[channel_idx]

            # calc actual resulting channel
            resulting_channel_epochs = np.zeros((weighted_epochs.shape[0], weighted_epochs.shape[2]))
            for trial_idx in range(0, weighted_epochs.shape[0]): 
                weighted_trial = weighted_epochs[trial_idx, :, :] # channels, sampels
                for channel_idx in range(0, self.epochs.shape[1]): 
                    
                    weighted_trial_channel = weighted_trial[channel_idx, :] # sampels
                    
                    if ((indices_diffs[channel_idx] > 0) and (max_sample_diff > indices_diffs[channel_idx])): # if there is a delay after the source channel 

                        weighted_trial_channel_temp = copy.deepcopy(weighted_trial_channel)
                        weighted_trial_channel_temp = np.pad(weighted_trial_channel_temp, pad_width = (0, np.abs(indices_diffs[channel_idx])), mode='edge') # append array at end 
                        weighted_trial_channel_temp = weighted_trial_channel_temp[indices_diffs[channel_idx]:]
                        
                    elif ((indices_diffs[channel_idx] < 0) and (max_sample_diff*(-1) < indices_diffs[channel_idx])):
                        
                        weighted_trial_channel_temp = copy.deepcopy(weighted_trial_channel)
                        weighted_trial_channel_temp = np.pad(weighted_trial_channel_temp, pad_width = (np.abs(indices_diffs[channel_idx]), 0), mode='edge') # append array at end 
                        weighted_trial_channel_temp = weighted_trial_channel_temp[0: len(weighted_trial_channel_temp)-np.abs(indices_diffs[channel_idx])]
                    
                    else: 
                        weighted_trial_channel_temp = weighted_trial_channel

                    weighted_epochs[trial_idx, channel_idx, :] = weighted_trial_channel_temp

                resulting_channel_epochs[trial_idx, :] = np.sum(weighted_epochs[trial_idx, :, :], axis = 0) 

            if (replace_epochs): 
                self.epochs = weighted_epochs*self.epochs.shape[1] # calc weighted epochs with same scaling as before 

                self.average_epochs = np.mean(self.epochs, axis = 0)
            
            resulting_channel_epochs  = np.expand_dims(resulting_channel_epochs, axis=1)
                
            return resulting_channel_epochs


    def reshapeWindowsForCNNnets(self): 
        """
        This function reshapes the windows from multiple trials to be fitted for the CNN networks like EEGNet.
        This method is only required if windows are processed for more than one trial and windwo (do not use for single window processing)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)  
        """        

        reshaped_EEG_windows= np.zeros((self.windows.shape[0]*self.windows.shape[3], self.windows.shape[1], self.windows.shape[2], 1)) # (n_trials * n_windows, n_channels, n_sampels, 1). 
        

        # get the features in one dim for all trials and windows 
        for channel_idx in range(0, self.windows.shape[1]):
            for sample_idx in range(0, self.windows.shape[2]): 

                reshaped_EEG_windows[:, channel_idx, sample_idx, 0] = self.windows[:, channel_idx, sample_idx, :].flatten()

        self.windows = reshaped_EEG_windows

    def labelsToCategorical(self, num_classes = 2): 
        """
        This method converts the class labels into the one hot encoding style. 

        Parameters
        ----------
        num_classes : int, optional
            Total number of classes. If None, this would be inferred as max(y) + 1, by default 2

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        y = to_categorical(self.labels, num_classes)

        self.labels = y

        
    def getWindows(self): 
        """
        This function returns the windowed data as a numpy array

        Returns
        -------
        Numpy array
            windows
                The windowed data as numpy array with shape: (n_trials, n_channels, n_sampels, n_windows).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.windows
    
    def getWindowNames(self): 
        """
        This funcion returns the identifier of each window

        Returns
        -------
        window_names : str
            The names (indentifier) of each window as a list of strings (has same size as n_windows).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """    

        return self.window_names
    
    def getFeatures(self): 
        """
        This functions returns the featuer vectors als floats

        Returns
        -------
        feature_vec : float
            A feature vector as numpy array, can have different shapes depending on the later used ML model (see ML class).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.feature_vec

    def getLabels(self): 
        """
        This functions returns the labels as a numpy array with floats

        Returns
        -------
        labels : float
            Numpy array of the labels
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.labels

    # def getTrainLabels(self):  --> not used anymore, use getLabels instead 
    #     """
    #     This functions returns the labels as a numpy array with floats

    #     Returns
    #     -------
    #     labels : 1D numpy array  
    #         A 1D numpy array containing the
        
    #     Author
    #     ------
    #     Author : Niklas Kueper \n
    #     Last changed: Missing (by Niklas Kueper)
    #     """ 

    #     return self.labels


    def onlineLRPWindowPredictionPostprocessing(self, window_wise_predicts, high_tresh, low_tresh, short_samp, long_samp): 
        """
        Apply an online capable postprocessing for the detection of LRP, where a linear function decides for the LRP class over which time a defined probability has to be reached for the detection of the positive class. 

        Parameters
        ----------
        window_wise_predicts : Numpy ndarray
            A numpy array with the predictions made on each window. Has the shape (n_trials, n_windows). 
        high_tresh : float 
            The higher treshold where the linear decision function ends. 
        low_tresh : float
            The lower treshold where the linear decision function starts. 
        short_samp : int
            The short sampels reflect how short the latest part or smallest chunk of data of interest is (i.e. only the late part or motor potential of an LRP).
        long_samp : int
            The long sampels reflecting how long the maximum ERP to be detected possibly is.  

        Returns
        -------
        Numpy ndarray 
            A numpy ndarray with shape (n_trials, n_windows) including the classification decision in the form of class labels (e.g. 1.0 or 0.0 for binary classification)
            
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 05.02.2024 (by Niklas Kueper)
        """

        classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

        for trial_idx in range(0, window_wise_predicts.shape[0]): 
            for window_idx in range(0, window_wise_predicts.shape[2]): 
                
                # get prediction scores of current trial and window 
                current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 

                tested_sampel_range = np.arange(short_samp, long_samp, step = -1)
                #print("sampel range", tested_sampel_range)
                tresh_step = -1*(high_tresh-low_tresh)/len(tested_sampel_range) # from high to low tresh (short sampels to long sampels)
                tested_tresh_range = np.arange(high_tresh, low_tresh, step = tresh_step)
                #print("Tresh range", tested_tresh_range)

                for index in range(0, len(tested_sampel_range)): 
                    mean_val = np.mean(current_predicts[tested_sampel_range[index]:]) 

                    if (mean_val > tested_tresh_range[index]): 
                        classified_windows[trial_idx, window_idx] = 1.0
                        break

        return classified_windows


    def windowEEGEpochs(self, window_size = 1000, window_step = 50, no_channel_dim = False):
        """
        This function cuts (overlapping) windows from continues EEG-signals (currently only for postprocessing without channel deimension)

        Parameters
        ----------
        window_size : int, optional
            The size of the windows in ms to be cutout, by default 1000
        window_step : int, optional
            The stepsize of the sliding window (slinding step) in ms, by default 50
        no_channel_dim : bool, optional
            Set to True if the EEG data (epochs) have no channel dimension, by default False
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """ 

        #if (with_channel_dim == False): 
        window_size_samp = int((window_size/1000) * self.__fsamp) 
        window_step_samp = int((window_step/1000) * self.__fsamp) 

        if(no_channel_dim): # support old way to postprocess
            epochs_arr_cut = self.epochs[:, 1:]
            num_of_windows =  int((epochs_arr_cut.shape[1]-window_size_samp)/window_step_samp)+1
        else: 
            epochs_arr_cut = self.epochs[:, :, 1:] # trials, channels, sampels 
            num_of_windows =  int((epochs_arr_cut.shape[2]-window_size_samp)/window_step_samp)+1


        #init window arrays with shape (trials, channels, sampel of window, windownumber)
        if(no_channel_dim): 
            wind_arr = np.zeros((epochs_arr_cut.shape[0], window_size_samp, num_of_windows))
        else:
            wind_arr = np.zeros((epochs_arr_cut.shape[0],epochs_arr_cut.shape[1], window_size_samp, num_of_windows))

        wind_names = []

        for win_nr in range(0, num_of_windows): 
            wind_start_idx = win_nr*window_step_samp
            wind_end_idx = window_size_samp+wind_start_idx

            if(no_channel_dim): 
                wind_name = "bis"+str(int((((epochs_arr_cut.shape[1]-wind_end_idx)*-1)/self.__fsamp) *1000))
            else: 
                wind_name = "bis"+str(int((((epochs_arr_cut.shape[2]-wind_end_idx)*-1)/self.__fsamp) *1000))
            
            wind_names.append(wind_name) # a list of all window names
            #create arrays for windows 

            if(no_channel_dim): 
                wind_arr[:, :, win_nr] = win_nr
                wind_arr[:, :, win_nr] = epochs_arr_cut[:, wind_start_idx:wind_end_idx]
            else:
                wind_arr[:, :, :, win_nr] = win_nr
                wind_arr[:, :, :, win_nr] = epochs_arr_cut[:, :, wind_start_idx:wind_end_idx]


        self.windows = wind_arr 
        #self.num_windows = num_of_windows
        self.window_names = wind_names




    def windowSelection(self, selected_windows): 
        """
        This function selects windwos and extract them from all windows segmented by specifying the window names

        Parameters
        ----------
        selected_windows : str
            The names of the windows (given after windowing) which are selected for further processing
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
       
        # interate over specified window names and extract the windows 
        indices_selected_winds = []
        for current_window_name in selected_windows: 
            indices_selected_winds.append(self.window_names.index(current_window_name))

        selected_windows_arr = self.windows[:, :, :, indices_selected_winds] 

        self.windows = selected_windows_arr
        self.window_names = selected_windows

    def windowStandardization(self, norm = False, use_min_max_norm = False):
        """
        This method can be used to standardize windowed time series data. 

        Parameters
        ----------
        norm : bool, optional
            Boolean flag wheather to normalize the window after standadizing the data, by default False
        use_min_max_norm : bool, optional
            If True, a minimum maximum normalization is applied to each window. Otherwise (default) the z-transform is applied, by default False

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
        
        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = self.windows[trial_idx, channel_idx, :, window_idx]
                    
                    if(use_min_max_norm): 
                        
                        current_wind_norm = current_wind - self.calib_mins[channel_idx]
                        current_wind_norm = current_wind_norm/((self.calib_mins[channel_idx]*-1)+self.calib_maxs[channel_idx])


                    else: 
                        # apply z-transform 
                        #print(current_wind.shape)
                        current_wind_norm = current_wind - np.mean(current_wind, axis = 0) #self.calib_means[channel_idx] 
                        #print("current window ", current_wind)
                        #print("mean val", np.mean(current_wind))
                        current_wind_norm_out = current_wind_norm/np.std(current_wind_norm)  #self.calib_stds[channel_idx]  
                        
                        if(norm): 
                        #current_wind_norm = current_wind+(-1*min)-1 # -1 is min 
                            current_wind_norm_out = current_wind_norm/np.max(current_wind_norm)


                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_norm_out # 1 is max 

                    # print("")
                    # print(np.min(current_wind_norm))
                    # print(np.mean(current_wind_norm))
                    # print(np.std(current_wind_norm))
                    # print(np.max(current_wind_norm))
                    # print("")

    def WindowMedianCorrection(self, ratio_len = 0.1):
        """
        Apply a median correction to each windowed timeseries data. 

        Parameters
        ----------
        ratio_len : float, optional
            The percentage of the window which is used for the median correction starting from the first point of each window (i.e. 0.1 reflects to the first 10% of the window used to calculated the median value to be corrected for), by default 0.1

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """         

        end_idx = int(self.windows.shape[2]*ratio_len)

        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])

                    median_val = np.median(current_wind[0:end_idx])

                    current_wind_med_corr = current_wind -median_val
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_med_corr # 1 is max 


    def WindowMeanCorrection(self): 
        """
        Apply a mean correction on windowed time series data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])

                    mean_val = np.mean(current_wind)

                    current_wind_mean_corr = current_wind -mean_val
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_mean_corr # 1 is max 

    
    def calcTestAccAndRates(self, prediction_labels, true_labels):
        """
        This function returns the metrics from classification output of the test data. Currently the accuracy, balaned accuracy, tnr and tpr are calculated

        Parameters
        ----------
        prediction_labels : numpy array
            The prediced labels as 1D numpy array (flatten the array if it has more dimensions)
        true_labels : numpy array
            The true labels as 1D numpy array (flatten the array if it has more dimensions)

        Returns
        -------
        tuple
            tnr : float
                True negative rate
            tpr : float
                True positive rate
            acc : float
                Accuracy
            ba : float
                Balanced accuracy

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """

        prediction_labels = prediction_labels.astype(int)
        true_labels = true_labels.astype(int)
        
        negatives = np.sum(np.abs(true_labels-1))
        positives = np.sum(true_labels)
        
        n = len(true_labels)
        tps= 0 
        tns = 0
        for index in range(0, n): 
            if (prediction_labels[index] == 0 and true_labels[index] == 0):
                tns = tns+1
            elif (prediction_labels[index] == 1 and true_labels[index] == 1): 
                tps = tps+1
        tnr = tns/negatives
        tpr = tps/positives
        acc = ((tns+tps)/n) #in percent
        ba = (tnr+tpr)/2

        return tnr, tpr, acc, ba


    # TODO: write this proper to be used 
    # def onlineWindowPredictionPostprocessing_v1(self, window_wise_predicts, short_tresh, mid_tresh, long_tresh, short_sampels, mid_sampels, long_sampels):
    #     """
    #     Missing

    #     Parameters
    #     ----------
    #     window_wise_predicts : Missing
    #         Missing
    #     short_tresh : Missing
    #         Missing
    #     mid_tresh : Missing
    #         Missing
    #     long_tresh : Missing
    #         Missing
    #     short_sampels : Missing
    #         Missing
    #     mid_sampels : Missing
    #         Missing
    #     long_sampels : Missing
    #         Missing

    #     Returns
    #     -------
    #     Numpy array
    #         classified_windows : float
    #             Missing
            
    #     Author
    #     ------
    #     Author : Niklas Kueper \n
    #     Last changed: Missing (by Niklas Kueper)
    #     """          

    #     classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

    #     for trial_idx in range(0, window_wise_predicts.shape[0]): 
    #         for window_idx in range(0, window_wise_predicts.shape[2]): 
                
    #             # get prediction scores of current trial and window 
    #             current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 
    #             mean_long_time_detections = np.mean(current_predicts[long_sampels:]) 
    #             mean_mid_time_detections = np.mean(current_predicts[mid_sampels:])
    #             mean_short_time_detections = np.mean(current_predicts[short_sampels:])

    #             # if one of both criteriums (short or long detection) is fulfilled the window gets the positive class label  
    #             if((mean_long_time_detections > long_tresh) or (mean_short_time_detections > short_tresh) or (mean_mid_time_detections > mid_tresh)): 
    #                 classified_windows[trial_idx, window_idx] = 1.0 
    #             else: 
    #                 classified_windows[trial_idx, window_idx] = 0.0
                
    #     return classified_windows
    
    def xDAWNSpatialfilter(self, n_components = 2, processing_type="fit_apply", return_filter = True, xd = None): 
        """
        This function implements an Xdawn filter algorithm. You can choose between only fit, only apply or both.

        Parameters
        ----------
        n_components : int, optional
            The number of components to decompose the signals, by default 2
        processing_type : str, optional
            Determans how you would like to use the function. If "fit" is parsed the Xdawn filter will be only fitted, if "apply" is parsed the Xdawn filter will be only applyed, "filter_apply" does both, by default "fit_apply"
        return_filter : bool, optional
            Returns the, by default True
        xd : instance of Xdawn, optional
            Parsing an instance of Xdawn for "apply" only purpose, by default None

        Returns
        -------
        xd : instance of Xdawn 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        if (processing_type == "fit_apply"): # assuming this is only for training data or the epochs it should be fitted on 

            # Xdawn instance
            xd = Xdawn(n_components=n_components)
            
            # Fit xdawn
            xd.fit(self.epoch_obj)
            # apply 
            epochs_denoised = xd.apply(self.epoch_obj)
            self.epochs = epochs_denoised[self.event_ids].get_data()
            

            if(return_filter): 
                return xd 

        elif(processing_type == "fit"): # only fit 

            # Xdawn instance
            xd = Xdawn(n_components=n_components)
            
            # Fit xdawn
            xd.fit(self.epoch_obj)

            if(return_filter): 
                return xd 
        
        elif(processing_type == "apply"): # apply only (pass fitted filter !) 
            # apply only 
            epochs_denoised = xd.apply(self.epoch_obj)
            self.epochs = epochs_denoised[self.event_ids].get_data() 


    def applyxDAWNToWindows(self, xd, n_components = 2):
        """
        This function applies an already trained Xdawn filter to windowed timeseries data. 

        Parameters
        ----------
        xd : instance of Xdawn
            This is an instance of an Xdawn filter object
        n_components : int, optional
            The number of components to decompose the signals, by default 2

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """         
        
        new_windows = np.zeros((self.windows.shape[0], n_components, self.windows.shape[2], self.windows.shape[3])) # reduced channel dim

        for window_idx in range(0, self.windows.shape[3]): 
            windows_filtered= xd.transform(self.windows[:, :, :, window_idx])
            new_windows[:, :, :, window_idx] = windows_filtered

        self.windows = new_windows # replace old windows 

    # TODO: write this proper 
    # def onlineWindowPredictionPostprocessing_v2(self, window_wise_predicts, high_tresh, low_tresh, short_samp, long_samp):
    #     """
    #     Missing

    #     Parameters
    #     ----------
    #     window_wise_predicts : Missing
    #         Missing
    #     high_tresh : Missing
    #         Missing
    #     low_tresh : Missing
    #         Missing
    #     short_samp : Missing
    #         Missing
    #     long_samp : Missing
    #         Missing

    #     Returns
    #     -------
    #     Missing
    #         _description_

    #     Author
    #     ------
    #     Author : Niklas Kueper \n
    #     Last changed: Missing (by Niklas Kueper)
    #     """         

    #     classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

    #     for trial_idx in range(0, window_wise_predicts.shape[0]): 
    #         for window_idx in range(0, window_wise_predicts.shape[2]): 
                
    #             # get prediction scores of current trial and window 
    #             current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 

    #             tested_sampel_range = np.arange(short_samp, long_samp, step = -1)
    #             #print("sampel range", tested_sampel_range)
    #             tresh_step = -1*(high_tresh-low_tresh)/len(tested_sampel_range) # from high to low tresh (short sampels to long sampels)
    #             tested_tresh_range = np.arange(high_tresh, low_tresh, step = tresh_step)
    #             #print("Tresh range", tested_tresh_range)

    #             for index in range(0, len(tested_sampel_range)): 
    #                 mean_val = np.mean(current_predicts[tested_sampel_range[index]:]) 

    #                 if (mean_val > tested_tresh_range[index]): 
    #                     classified_windows[trial_idx, window_idx] = 1.0
    #                     break

    #     return classified_windows


    # TODO: write this proper for integrating again in toolbox (do not remove completely)
    # def onlineWindowPredictionPostprocessing_v3(self, window_wise_predicts, thresh, start_samp):
    #     """
    #     This function determine the class label for each window in each trial based on the mean of the prediction scores in regard to a threshold.

    #     Parameters
    #     ----------
    #     window_wise_predicts : Numpy array
    #         Representing predictions for different trials, time steps, and windows
    #     thresh : int
    #         A threshold value used for decision making during postprocessing
    #     start_samp : int
    #         The starting sample inde

    #     Returns
    #     -------
    #     Numpy array
    #         classified_windows : float
    #             A 2D NumPy array with shape (n_trials, n_windows), where each entry is either 0.0 or 1.0, indicating the class label assigned to a specific window in a particular trial.
        
    #     Author
    #     ------
    #     Author : Niklas Kueper \n
    #     Last changed: Missing (by Niklas Kueper)
    #     """        
        
    #     classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

    #     for trial_idx in range(0, window_wise_predicts.shape[0]): 
    #         for window_idx in range(0, window_wise_predicts.shape[2]): 
                
    #             # get prediction scores of current trial and window 
    #             current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 

    #             mean_detections = np.mean(current_predicts[start_samp:]) 

    #             # if one of both criteriums (short or long detection) is fulfilled the window gets the positive class label  
    #             if(mean_detections > thresh): 
    #                 classified_windows[trial_idx, window_idx] = 1.0 
    #             else: 
    #                 classified_windows[trial_idx, window_idx] = 0.0

    #     return classified_windows


    def calcTrialMetric(self, predict_scores, pos_class_start_time, f_samp, decision_bound, num_class_instances): # TODO: Sample rate does not have to be a parameter here
        """
        This function seperates, calculates and returns the prediction of posivie and negative classes based on the specified bounds

        Parameters
        ----------
        predict_scores : Numpy ndarray 
            A numpy ndarray containing the prediction scores (shape: (n_trials, n_predictions)). 
        pos_class_start_time : float
            The time in ms where the positive class starts or is defined (binary classification only) in each trial. 
        f_samp : int
            The samplerate in Hz
        decision_bound : list
            A list specifying the bounds for positive class predictions
        num_class_instances : int
            The number of positive class instances

        Returns
        -------
        float
            ba : Balanced accuracy
            tnr : True negative rate
            tpr : True positive rate

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """       

        pos_class_start_samp = int((pos_class_start_time/1000) * f_samp)
        tns = 0 
        tps = 0 
        fns = 0 
        fps = 0

        for trial in predict_scores: 
            # seperate the predictions of both classes 
            pos_class_predicts = trial[pos_class_start_samp:]
            neg_class_predicts = trial[0:len(trial)+pos_class_start_samp]

            # calc the number of times the score is over the decision bound  
            pos_class_predicts_over_bound = np.sum(pos_class_predicts > decision_bound) 
            neg_class_predicts_over_bound = np.sum(neg_class_predicts > decision_bound) 

            # calc metrics on numbers of times 
            if(pos_class_predicts_over_bound > num_class_instances): # at leat one pos class instance detected = true prediction 
                tps = tps +1  
            else: 
                fns = fns+1 # no pos class instance detected = falsely predicted negative class 

            if(neg_class_predicts_over_bound < num_class_instances): # no pos class detected, all true 
                tns = tns +1
            else: 
                fps = fps +1 # wrongly detected positive class 

            # calc rates and metric 
            tnr = tns/(tns+fps) 
            tpr = tps/(tps+fns)
            ba = (tnr+tpr)/2

        return ba, tnr, tpr 
    

    def OnechannelFFT(self, one_channel_data, plot = False, window = "hamming", beta = 1, title = None):
        """
        This function calculates the FFT for one channel of the timeseries data

        Parameters
        ----------
        one_channel_data : Numpy array
            The data for one channel
        plot : bool, optional
            If True the FFT spectrum of the data is plotted, by default False
        window : str, optional
            The windowing function that can be applied before the FFT is calculated (see usable scipy window functions), by default "hamming"
        beta : int, optional
            Shape parameter, determines trade-off between main-lobe width and side lobe level. As beta gets large, the window narrows, by default 1
        title : str, optional
            The Title of the plot, by default None

        Returns
        -------
        Numpy array
            xf : float
            The frequency axis
            yfn : float
            The FFT magnitude values of the frequency spectrum

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)
        """         

        N = len(one_channel_data)
        # sample spacing
        dT = 1.0/self.__fsamp
        x = np.linspace(0.0, N*dT, N, endpoint=False)
        
        # apply window before calculating fft 
        if(window == "hamming"): 
            window_fct = sig.windows.hamming(N)
            y = window_fct*one_channel_data
        elif (window == "hann"): 
            window_fct = sig.windows.hann(N)
            y = window_fct*one_channel_data

        elif (window == "kaiser"): 
            window_fct = sig.windows.kaiser(N, beta = beta)
            y = window_fct*one_channel_data

        else: 
            y = one_channel_data


        yf = fft(y)

        xf = fftfreq(N, dT)[:N//2]
        yfn = 2.0/N * np.abs(yf[0:N//2])


        if (plot): 
            fig = plt.figure()
            plt.plot(xf, yfn)
            plt.ylabel("|H|")
            if(title): 
                plt.title(title)
            else: 
                plt.title("FFT Spectrum")
            plt.xlabel("Frequency in Hz")
            plt.show()

        return xf, yfn
     
    
    def detrendWindows(self):
        """
        This funcition uses detrending to remove linear trends from each window of the data.
        Self.windwos array will contain detrended time series data for each trial, window, and channel

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        # get the features in one dim for all trials and windows 
        for trial_idx in range(0, self.windows.shape[0]):
            for window_idx in range(0, self.windows.shape[3]):
                for channel_idx in range(0, self.windows.shape[1]):
                    current_window = self.windows[trial_idx, channel_idx, :, window_idx]
                    current_window_corr = sig.detrend(current_window)
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_window_corr


    def featureExtractionFromWindows(self,  feature_type = "timepoints", feature_indices_windows = None, use_mean = False, N = 1, add_neightbour_diffs  = False, neighbours_list = [("C1", "CZ")], psd_method = "multitaper"): 
        """
        Apply method to extract time or frequency features from time series data. See feature_types parameter for the types of features that are supported. 

        Parameters
        ----------
        feature_type : str, optional
            _description_, by default "timepoints"
        feature_indices_windows : Numpy array, optional
            Numpy array with time feature indices, by default None
        use_mean : bool, optional
            If True, the mean of the timepoints is calculated as features, by default False
        N : int, optional
            The numbers of samples when calculating mean features, by default 1
        add_neightbour_diffs : bool, optional
            If True, the differences between channel features are added as additional features (e.g. for neighbour channels), by default False
        neighbours_list : list, optional
            A list of tuples specifying the neighbour channels for adding the neighbour features (only used when add_neighbour_diffs == True), by default [("C1", "CZ")]
        psd_method : str, optional
            The method to be used for calculating psd features (see compute_pow_freq_bands of mne_features for detailled information), by default "multitaper"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        # (n_trials, n_channels, n_sampels, n_windows).

        if(feature_type == "timepoints"): 
            
            feature_times_indices = ((feature_indices_windows/1000)*self.__fsamp).astype(int)

            # init stuff 
            if(use_mean): 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], N*self.windows.shape[1]))

            else: 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], int(len(feature_times_indices)*self.windows.shape[1]))) # shape: trials, windows, features

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], len(feature_times_indices)*len(neighbours_list))) # shape: trials, windows, features

            mean_feat_buffer = np.zeros((self.windows.shape[1], N))


            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]): 
                    
                    if(use_mean):
                        # shape channel, sampels
                        #  
                        current_wind = self.windows[trial_idx, :, feature_times_indices[0]:feature_times_indices[-1], window_idx] # use the first and las value only 
                        k = int(current_wind.shape[1]/N) 

                        for idx in range(0, N): 
                            mean_feat_buffer[:, idx] = np.mean(current_wind[:, (idx*k):((idx*k) +k)], axis = 1) 

                        x_train_features[trial_idx, window_idx, :] = mean_feat_buffer.flatten() # use mean of timepoints
                    else: 
                        x_train_features[trial_idx, window_idx, :] = self.windows[trial_idx, :, feature_times_indices, window_idx].flatten()

                    # neighbour diff features 
                    if (add_neightbour_diffs): # if you want to add local feature diffs 
                        features_add = np.zeros((x_train_features.shape [0], x_train_features.shape[1], len(feature_times_indices), len(neighbours_list))) # if use mean values only one dim 
                        i = 0

                        for channel_tup in neighbours_list: # loop over all indice values and calc diff of features 
                            idx1 = self.__ch_names.index(channel_tup[0])
                            idx2 = self.__ch_names.index(channel_tup[1])

                            # shape: trials, windows, features 
                            current_wind_feat = self.windows[trial_idx, :, feature_times_indices, window_idx].T # get current window features: channel, sampels

                            features_add[trial_idx, window_idx, :, i] = (current_wind_feat[idx1, :] - current_wind_feat[idx2, :])
                            i = i+1

                        # flatten the feature dims 
                        x_train_features_add[trial_idx, window_idx, :] = features_add[trial_idx, window_idx, :, :].flatten()

                    
            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)


            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()

            # flatten data 
            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()



        elif(feature_type == "meanfreqs" or feature_type == "medianfreqs"): #fix this 


            # (n_trials, n_channels, n_sampels, n_windows).
            num_of_freq_bands = 5 

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*len(neighbours_list))) # shape: trials, windows, features

            x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*self.windows.shape[1])) # shape: trials, windows, features

            #print(x_train_features.shape)
            features = np.zeros((num_of_freq_bands, self.windows.shape[1]))
            
            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]):
                    for channel_idx in range(0, self.windows.shape[1]):
                         
                        xf, yf = self.OnechannelFFT(self.windows[trial_idx, channel_idx,:, window_idx], self.__fsamp) # do fft of sliced data

                        normed_freqs = (xf*yf)/np.sum(yf) # norm the frequencies

                        if(feature_type == "meanfreqs"): 
                            freqs_arr = np.array([np.mean(normed_freqs[0:4]), np.mean(normed_freqs[4:8]), np.mean(normed_freqs[8:14]), np.mean(normed_freqs[14:30]), np.mean(normed_freqs[30:40])]) #  mean band frequencies
                        else: 
                            freqs_arr  = np.array([np.median(normed_freqs[0:4]), np.median(normed_freqs[4:8]), np.median(normed_freqs[8:14]), np.median(normed_freqs[14:30]), np.median(normed_freqs[30:40])]) #  mean band frequencies

                        features[:, channel_idx] = freqs_arr # freq band features for each channels 

                    if (add_neightbour_diffs): # if you want to add local feature diffs 
                        features_add = np.zeros((features.shape[0], len(neighbours_list)))
                        i = 0

                        for channel_tup in neighbours_list: # loop over all indice values and calc diff of features 
                            idx1 = self.__ch_names.index(channel_tup[0])
                            idx2 = self.__ch_names.index(channel_tup[1])
                            features_add[:, i] = features[:, idx1] -features[:, idx2] # calculate local feature diffs 
                            i = i+1

                        # flatten the feature dims 
                        x_train_features_add[trial_idx, window_idx, :] = features_add.flatten()

                    x_train_features[trial_idx, window_idx, :] = features.flatten()


            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()

            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()


        elif(feature_type == "freqBandPower"): 

            #windows (n_trials, n_channels, n_sampels, n_windows).

            freq_bands = np.array([0.5, 4., 8., 13., 30., 100.]) # default that is used 
            num_of_freq_bands = len(freq_bands)-1

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*len(neighbours_list))) # shape: trials, windows, features

            x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*self.windows.shape[1])) # shape: trials, windows, features

            #print(x_train_features.shape)
            features = np.zeros((num_of_freq_bands, self.windows.shape[1])) # 5, 34
            
            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]):
                         
                    current_wind = self.windows[trial_idx, :, :, window_idx] # shape channel, sampels
                    features = mne_feat.compute_pow_freq_bands(sfreq = self.__fsamp, data =current_wind, freq_bands=freq_bands, normalize = False, psd_method = psd_method)
                    x_train_features[trial_idx, window_idx, :] = features

            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()

            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()



        # how to proceed with features (both conditions)
        if(add_neightbour_diffs): 
            self.feature_vec = np.concatenate((x_train, x_train_add), axis = 1)
        else: 
            self.feature_vec = x_train 


    def addFeatures(self, x):
        """
        This function add features to the feature_vec by concatinating them

        Parameters
        ----------
        x : numpy array
            The features that should be added. An array of floats with the same shape as feature_vec

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """ 

        features = np.concatenate((self.feature_vec, x), axis = 1)
        self.feature_vec = features 

    def printFeatureShape(self):
        """
        This function prints the shape of feature_vec

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """   

        print("feature shape: ", self.feature_vec.shape)

    def setWindowLabels(self, label_list):
        """
        Set / encode the class labels of segmented windows fot the classifiction task

        Parameters
        ----------
        label_list : list of floats
            A list of labels that correspond to the window class labels (e.g. [0.0, 0.0, 1.0, 1.0])

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.07.2023 (by Niklas Kueper)
        """         

        y = np.zeros((self.windows.shape[0], self.windows.shape[3])).astype(dtype=np.float64)
        for trial_idx in range(0, y.shape[0]): 
            y[trial_idx, :] = np.array(label_list)  # shape: trials, window labels

        y = y.flatten() # flatten the labels
        y_temp = np.zeros((y.shape[0], 1))
        y_temp[:, 0] = y
        y = y_temp

        self.labels = y 

    def dtwFeatureVecWindows(self):
        """
        This method generates a feature vector for the dtw algorithm based on windowed data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  

        # windows in trials, channel, sampels, windows 
        self.feature_vec = np.zeros((int(self.windows.shape[0]*self.windows.shape[3]), self.windows.shape[1], self.windows.shape[2]))

        for channel_idx in range(0, self.windows.shape[1]): 
            for sample_idx in range(0, self.windows.shape[2]): 
                self.feature_vec[:, channel_idx, sample_idx] = self.windows[:, channel_idx, sample_idx, :].flatten()
    

    def calcEEGWindowOnset(self, window_predicts, num_pos_windows):
        """
        The function returns a numpy array, indicating the onset of positive labels for each trial and window.

        Parameters
        ----------
        window_predicts : Numpy array
            Representing predicted labels for each trial and window
        num_pos_windows : int
            The number of consecutive positive windows required to trigger an onset

        Returns
        -------
        Numpy array
            onset_window_predicts : indicating the onset of positive labels for each trial and window

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """         

        onset_window_predicts = np.zeros(window_predicts.shape)
        trial_idx = 0

        for trial in window_predicts:
            count_pos_windows = 0 

            for wind_idx in range(0, window_predicts.shape[1]): 
                if(trial[wind_idx] > 0.5): # window has pos label 
                    count_pos_windows = count_pos_windows+1
                
                if (count_pos_windows >= num_pos_windows): 
                    onset_window_predicts[trial_idx, wind_idx] = 1.0 
                    count_pos_windows = 0
                    break

            trial_idx = trial_idx+1

        return onset_window_predicts

    def calcTrialMetricWindows(self, predict_labels, bounds, num_class_instances):
        """
        This function seperates, calculates and returns the prediction of posivie and negative classes based on the specified bounds


        Parameters
        ----------
        predict_labels : list   
            A list of predicted labels for each trial
        bounds : list
            A list specifying the bounds for positive class predictions
        num_class_instances : int
            The number of positive class instances

        Returns
        -------
        float
            ba : Balanced accuracy
            tnr : True negative rate
            tpr : True positive rate

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """ 

        tns = 0 
        tps = 0 
        fns = 0 
        fps = 0

        for trial in predict_labels: 
            # seperate the predictions of both classes 
            pos_class_predicts = trial[bounds[0]:bounds[1]]
            neg_class_predicts = trial[0:bounds[0]]
            
            # calc the number of times the score is over the decision bound  
            pos_class_predicts_over_bound = np.sum(pos_class_predicts) 
            neg_class_predicts_over_bound = np.sum(neg_class_predicts) 

            # calc metrics on numbers of times 
            if(pos_class_predicts_over_bound >= num_class_instances): # at leat one pos class instance detected = true prediction 
                tps = tps +1  
            else: 
                fns = fns+1 # no pos class instance detected = falsely predicted negative class 

            if(neg_class_predicts_over_bound < num_class_instances): # no pos class detected, all true 
                tns = tns +1
            else: 
                fps = fps +1 # wrongly detected positive class 

            # calc rates and metric 
            tnr = tns/(tns+fps) 
            tpr = tps/(tps+fns)
            ba = (tnr+tpr)/2

        return ba, tnr, tpr 


    def applyRelabelling(self, predicted_labels, determine_labels, searching_bounds):
        """
        This function apllies the relabelling method to the classification output in order to get the "true ground truth" labels. This function schould be carefully use since it creates new ground truth labels for the evaluation of the classifier!

        Parameters
        ----------
        predicted_labels : Numpy array
            The predicted labels as 1D-Numpy array (flatten the arry if it has more dimensions)
        determine_labels : int
            The amount of negative classes that are counted from the right side(end of each epoch/trial to specify the "label change point")
        searching_bounds : list
            A list with boundaries([lower bound, upper bound]) in which the label change point for the relabelling is searched (e.g. for LRP the numbers of the window for -1000 ms and 0 ms)

        Returns
        -------
        Numpy array
            new_true_labels : Containing the new ground truth labels (0.0 neg class; 1.0 pos class) with shape (n_trials, n_samples)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """        

        new_true_labels = np.zeros(predicted_labels.shape)
        new_true_labels[:, -1] = 1.0 # last label has to be positive class 

        #predicted_labels_cut = predicted_labels[:, searching_bounds[0]:searching_bounds[1]] # cut to searching bounds where the label change point is searched 
        
        trial = 0 

        for predicted_trial_label in predicted_labels: 
            # reset for every trial 
            neg_class_counter = 0
            index =  searching_bounds[1]-1 # do this like that for now but change afterwards,  upper bound could lay in the middle ! 

            while index > searching_bounds[0]: # go from the back to front until the minimum point 
                # count number of neg classes 
                if (predicted_trial_label[index] == 0): 
                    neg_class_counter = neg_class_counter+1
                else: 
                    neg_class_counter = 0
                
                if(neg_class_counter >= determine_labels): 
                    new_true_labels[trial, index+determine_labels:] = 1.0 # set true labels after label change point
                    break # stop when change point was found
                
                index = index -1

            # no consecutive determine label numbers of neg class found 
            if(index <= searching_bounds[0] and neg_class_counter < determine_labels): 
                new_true_labels[trial, searching_bounds[0]:] = 1.0

            trial = trial +1

        return new_true_labels


    def calcWindowMetrics(self, wind_arr, evaluation_time_per_window, window_step, f_samp_eeg, n_samp_features, use_relabelling, determine_labels, searching_bounds):
        """
        This function calculates the metrics of a window wise classification output

        Parameters
        ----------
        wind_arr : Numpy array
            The windowes class predictions with shape (n_trials, n_samples, n_windows)
        evaluation_time_per_window : int
            The time in ms at the end of each window for which the window metric is calculated (-evaluation_time_per_window to 0 ms are used)
        window_step : int
            The slinding step size of the windows in ms (standard is 50 ms)
        f_samp_eeg : int
            The sampling rate of the EEG-data in Hz
        n_samp_features : int
            The number of samples that are used as features (-n_samp_features: 0 of each epoch)
        use_relabelling : bool
            If True, the relabelling method is applied to calculate the metrics. This method should be trated with care since it effects the classification perfomance!
        determine_labels : int
            The number of concecutive negative classes that are counted when estimating the label change point of both classes
        searching_bounds : list
            A list of the lower and upper bound of window numbers ([lower bound, upper bound]) where the label change point is searched

        Returns
        -------
        float
            tnr : True negative rate
            tpr : True positive rate
            acc : Accuracy
            ba : Balanced accuracy
            window_predictions : The prediction values (0 - 1) for all windows and trials
        Numpy array
            window_eval_true_labels : float
                The true labels that are specified or the new true labels when relabelling is used

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 29.11.2022 (by Niklas Kueper)
        """         

        window_step_samp = int((window_step/1000) * f_samp_eeg) 
        evaluation_samp_per_window = int((evaluation_time_per_window/1000) * f_samp_eeg) 
        window_predictions_samp = np.sum(wind_arr[:, -evaluation_samp_per_window:, :], axis = 1)
        
        # convert the label of each sampels to windowwise labels 
        window_predictions = (window_predictions_samp > evaluation_samp_per_window/2).astype(float)
        
        
        window_eval_true_labels = np.zeros(window_predictions.shape)
        num_erp_windows = int(np.round(n_samp_features/window_step_samp)) 
        trial_erp_label = window_eval_true_labels[0, :] 
        trial_erp_label[-num_erp_windows:] = 1.0

        for trial_nr in range(0, window_eval_true_labels.shape[0]): 
            window_eval_true_labels[trial_nr, :] = trial_erp_label

        #get metrics for window evaluation 
        if (use_relabelling == False): 
            tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), window_eval_true_labels.flatten())
        else: 
            relabelled_true_labels = applyRelabelling(window_predictions, determine_labels, searching_bounds)
            tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), relabelled_true_labels.flatten())
            window_eval_true_labels = relabelled_true_labels # return the relabelled true labels instead 

        return tnr, tpr, acc, ba, window_predictions, window_eval_true_labels


class OnlineEEG(EEGData): 
    """
    This class is based on the EEGData class and includes additional methods for online data processing and classification. 

    Parameters
    ----------
    channel_names : list
        A list of the channel names
    n_channels : int, optional
        The number of channels, by default 34
    n_samples : int, optional
        The number of samples, by default 500
    dt_process_data : float, optional
        The time in s how fast the processing and classification loop is running, by default 0.05
    f_samp_eeg : float, optional
        The sample frequency of the EEG, by default 500.0

    Parameters
    ----------
    EEGData : _type_
        _description_

    Author
    ------
    Author : Niklas Kueper \n
    Last changed: 08.03.2024 (by Niklas Kueper)
    """    

    def __init__(self, channel_names, n_channels=34, n_samples= 500, dt_process_data = 0.05, f_samp_eeg = 500.0): 
        """
        The constructor of the OnlineEEG class. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        self.n_channels = n_channels
        self.buffersize = n_samples
        self.dt_process_data = dt_process_data
        self.data_buffer = np.zeros((1, n_channels, self.buffersize, 1)) # data buffer has shape (trials, n_channels, sampels, windows)

        super().__init__(format = "Live", f_samp = f_samp_eeg, channel_names = channel_names)
        

    def sendDetectedEventToAPI(self, timestamp_buffer_vals, local_clock_time, team_name = "example_team", secret_id = 5, url = 'http://10.250.223.221:5000/results'):
        """
        This function gathers all the relevant results and sends it to the host
        This function should be called every time aan error is detected

        Parameters
        ----------
        timestamp_buffer_vals : int
            Subset of the timestamp_buffer array at the instant when you have predicted an error and want to send the current result. Basically the i-th element of the timestamp_buffer array
        local_clock_time : floar
            Current LSL local clock time when you have run your classifier and predicted an error. This can be determined with the helf of "local_clock()" call
        team_name : str, optional
            Each team will be assigned a team name, by default "example_team"
        secret_id : int, optional
            Each team will be provided with a secret code, by default 5
        url : str, optional
            The URL where the results are stored, by default 'http://10.250.223.221:5000/results'

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        
    
        # calculate the final values for the timings 
        comm_delay = timestamp_buffer_vals[1] -timestamp_buffer_vals[0] -timestamp_buffer_vals[2]
        computation_time = local_clock_time - timestamp_buffer_vals[1]

        # connection to API for sending the results online 
        
        myobj = {'team': team_name,
                'secret': secret_id,
                'host_timestamp': timestamp_buffer_vals[0], 
                'comp_time': computation_time, 
                'comm_delay': comm_delay}

        x = requests.post(url, json = myobj)


    def printStreamMetadata(self, stream_info_obj):
        """
        This function prints some basic meta data of the stream

        Parameters
        ----------
        stream_info_obj : StreamInlet object
            A pylsl StreamInlet object which contains alle the information of the stream

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        print("") 
        print("Meta data")
        print("Name:", stream_info_obj.name())
        print("Type:", stream_info_obj.type())
        print("Number of channels:", stream_info_obj.channel_count())
        print("Nominal sampling rate:", stream_info_obj.nominal_srate())
        print("Channel format:",stream_info_obj.channel_format())
        print("Source_id:",stream_info_obj.source_id())
        print("Version:",stream_info_obj.version())
        print("")

    
    def updateBuffer(self, chunk, channel_indices = None, check_sample_loss = True):  #current_local_time, timestamp_offset, 
        """
        This function provides the most recent data samples and timestamps in a buffer (fist val is oldest, last the newest)

        Parameters
        ----------
        chunk : list
            Current data chunk with shape (samples, channels)
        channel_indices : list, optional
            If only selected channel indices should be extraced, by default None
        check_sample_loss : bool, optional
            If True the function is checkinf for sample losses, by default True

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        #data 
        
        current_chunk = (np.array(chunk).T) # chunk is sampels, channels, after transpose then channels, sampels !
        
        #print("chunk shape", current_chunk.shape) # should be channels, sampels 

        if(channel_indices): 
            current_chunk = current_chunk[channel_indices, :]

        # #print(current_chunk.shape)
        # current_chunk = current_chunk[0:n_channels, :] # use first n channels

        n_samples = current_chunk.shape[1] 

        if (n_samples > self.data_buffer.shape[2]): # print error message 
            print("Buffer overflow")

        
        self.data_buffer = np.roll(self.data_buffer, shift = int(-1*n_samples), axis = 2) # shift array by n samples  data_buffer: shape (trials, channel, sampels, windows)
        self.data_buffer[0, :, int(-1*n_samples):, 0] = current_chunk # channels, sampels shape , update latest values in buffer  --> is this correct 

        if (check_sample_loss): 
            # check for sample loss 
            sample_indices = self.data_buffer[0, -3, :, 0].astype(int) # sample indice channel
            for i in range(0, len(sample_indices) -1): 
                if sample_indices[i] + 1 != sample_indices[i+1]:
                    warnings.warn(f"Sample loss at {i}: {sample_indices[i:i+2]}")

    def BufferToWindows(self, num_non_data_channels = 3):
        """
        This method converts the buffered data from the data_buffer into data windows for further processing.  

        Parameters
        ----------
        num_non_data_channels : int, optional
            Number of channels to be removed, by default 3
        """      

        self.windows = self.data_buffer[:, 0:self.n_channels-num_non_data_channels, :, :] # assuming last num_non_data_channels are appended at the end (as done by LiveAmp connector)

    def getDataBuffer(self): 
        """
        This function returns the data_buffer

        Returns
        -------
        Numpy array
            data_butter : float

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  

        return self.data_buffer
    
    # ZMQ stuff 
    #TODO: remove this in the future, it is implemented in another repository now which is marker_sync_utils 
    def startZMQServer(self, port_name):
        """
        This function creates a socket connection as a publisher to send commands
        
        Parameters
        ----------
        port_name : str
            The address string. This has the form "tcp://interface:port"

        Returns
        -------
        zmq.Socket instance
            The created socket object for the zmq connection.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """

        # create a socket connection as a publisher to send commands 
        my_context = zmq.Context()
        my_socket = my_context.socket(zmq.PUB)
        my_socket.bind("tcp://*:"+port_name)
        print("Publisher ready")
        return my_socket
    
    #zmq server
    #TODO: remove this in the future, it is implemented in another repository now which is marker_sync_utils 
    def establishZMQ(port_name):
        """
        This function creats a socket connection as a pubisher to send commands

        Parameters
        ----------
        port_name : str
           The address string. This has the form "tcp://interface:port"

        Returns
        -------
        zmq.Socket instance
            The created socket object for the zmq connection.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  
        
        # create a socket connection as a publisher to send commands 
        my_context = zmq.Context()
        my_socket = my_context.socket(zmq.PUB)
        my_socket.bind("tcp://*:"+port_name)
        print("Publisher ready")
        return my_socket


