# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import warnings 

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************
class EMGData:
    """
    This class includes useful methods for the processing and visualization of EMG data. It mainly depends on numpy.

    Parameters
    ----------
    format : str, optional
        The formal in which the data is loaded", by default "ANTmini"
    data_path : str, optional
        The path where the data is stored, by default None
    filename : str, optional
        The file to be loaded, by default None
    f_samp : int, optional
        The sampling rate of the EMG system in Hz, by default None
    channel_names : list of str, optional
        A list of strings with the channel names/muscles, by default None 

    Author
    ------
    Author : Niklas Kueper \n
    Last changed: Missing (by Niklas Kueper)
    """    

    def __init__(self, format="ANTmini",data_path = None, filename = None, f_samp = None, channel_names = None): 

        """_summary_
        
        Parameters
        ----------
        format : str, optional
            _description_, by default "ANTmini"
        data_path : _type_, optional
            _description_, by default None
        filename : _type_, optional
            _description_, by default None
        f_samp : _type_, optional
            _description_, by default None
        channel_names : _type_, optional
            _description_, by default None
        """
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """        

        # parameter 
        self.__fsamp = f_samp
        self.channel_names = channel_names
        self.data = None
        self.__data_state = "continious" # the current state of processing of the data, can be "continious", "epochs", "windows"

        if(filename):

            if(format == "ANTmini"):
                raw_data, self.time_axis,  = self.loadMiniANTEMGData(data_path, filename, self.__fsamp)
            else: 
                raw_data, self.time_axis, self.channel_names = self.loadCometaEMGData(data_path, filename)

        self.data = raw_data
        # print("data shape:", self.data.shape)

    def getEMGData(self):
        """
        Returns a tuple wit the EMG data and EMG time axis

        Returns
        -------
        tuple
            data : Numpy array
                The EMG data with shape (n_samples, n_channel)
            time_axis : Numpy array
                The time axis of the EMG data with shape (n_samples,)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """

        return self.data, self.time_axis
    
    def getChannelNames(self): 
        """
        Returns the EMG channel names

        Returns
        -------
        Numpy arry
            channel_names : str
                The EMG channel names/muscles (names specified in the recording software) with shape (n_channels, )

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """

        return self.channel_names
        
    def getEMGfiltered(self):
        """
        Returns the filtered EMG data 

        Returns
        -------
        Numpy array
            emg_filtered : float
                The filtered EMG data witch have the same shape as data with shape (n_samples, n_channel)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """

        return self.emg_filtered
    
    def getSamplingRate(self):
        """
        Returns the sampling rate of the EMG

        Returns
        -------
        int
            __fsamp
                The sampling rate the data was recorded with

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """

        return self.__fsamp

    def loadCometaEMGData(self, data_path, file_str):
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

        filename = data_path+file_str
        emg_data = np.loadtxt(filename, dtype = float, delimiter=None, skiprows=5)

        # extract EMG channel names 
        channel_names = np.loadtxt(filename, dtype = str, delimiter=':', max_rows=1, skiprows=4)
        channel_names = channel_names[1:-1] # cut off last and first values since they are not EMG channel names 

        emg_data_channel = emg_data[:, 1:] # channel dimensions 
        emg_time_axis = emg_data[:, 0] # time axis 

        # calc sampling freq from data 
        dt = emg_time_axis[1] - emg_time_axis[0]
        self.__fsamp = float(1/dt)

        emg_data_channel = emg_data_channel.T # transpose for same data format (channels, sampels)

        return emg_data_channel, emg_time_axis, channel_names


    def loadMiniANTEMGData(self, data_path, file_str, f_samp): 
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
        emg_data_raw = np.loadtxt(os.path.join(data_path, file_str)) # at least th
        emg_data = emg_data_raw[:-1, :-2]
        time_axis = np.arange(0, (len(emg_data)/f_samp), step = 1/f_samp)

        emg_data = emg_data.T # transpose for timeseries data format 

        return emg_data, time_axis 
    

    def showEMGData(self):
        """
        This function is plotting the EMG data

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """        

        if (self.data.ndim > 1): 
            num_channels = self.data.shape[0]

            for n_channel in range(0, num_channels): 
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


    def channelSelection(self, selected_channels, inverse): 
        """
        This function selects EMG channel and edits the EMG data

        Parameters
        ----------
        selected_channels : list
            A list of channels. If the given channels are kept oder dropped is defined by the inverse parameter
        inverse : bool
            If False only the given channels are kept. Otherwise the given channels are dropped

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """        

    #     ch_indices = []

    #     for selected_names in selected_channels: 
    #         ch_indices.append(np.where(self.channel_names == selected_names)[0][0])

    #     if(inverse == True): 
    #         new_ch_indices = np.arange(0, len(self.channel_names)-1)
    #         ch_indices = np.delete(new_ch_indices, np.array(ch_indices))

    #     remaining_emg_data = self.data[:, ch_indices]
    #     if(remaining_emg_data.shape[1] == 1): # if only one channel cut of second dimension 
    #         remaining_emg_data = remaining_emg_data[:, 0]

    #     remaining_emg_channels = self.channel_names[ch_indices]

    #     self.data = remaining_emg_data
    #     self.channel_names = remaining_emg_channels


    def decimateEMGData(self, emg_data, time_axis, target_frequency, fsamp_emg):
        """
        This function is performing a downsampling to a given target frequency

        Parameters
        ----------
        emg_data : numpy array
            The EMG data with shape (n_samples, n_channel)
        time_axis : numpy array
            The time axis of the EMG data with shape (n_samples, )
        target_frequency : int
            The target frequency to which it should be downsampled in Hz
        fsamp_emg : int
            The sampling frequency in Hz

        Returns
        -------
        tuple
            dec_emg_data : numpy array
                The new downsampled EMG data with shape (n_samples, n_channel)
            new_time_axis : numpy array
                The new downsampled time axis of the EMG data with shape (n_samples)
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """        
        
    #     down_factor = int(fsamp_emg/target_frequency)

    #     if(emg_data.ndim >1):
    #         emg_decimated = np.zeros((int(emg_data.shape[0]/down_factor), emg_data.shape[1]))

    #         for channel_idx in range(0, emg_data.shape[1]): 
    #             emg_dec = sig.decimate(emg_data[:, channel_idx], down_factor)

    #             # check for length differences 
    #             if (len(emg_dec) == emg_decimated.shape[0]):
    #                 emg_decimated[:, channel_idx] = emg_dec
    #             else: 
    #                 emg_decimated[:, channel_idx] = emg_dec[1:]

            
    #         dec_emg_data = emg_decimated
    #     else: 
    #         dec_emg_data = sig.decimate(emg_data, down_factor)

    #     # change time axis 
    #     dt = (time_axis[1]-time_axis[0])*down_factor
    #     new_time_axis = np.arange(time_axis[0], time_axis[-1], step = dt)

    #     return dec_emg_data, new_time_axis
    

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

        if (self.__data_state == "continious"): 
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

    
    def applyVarianceFilter(self, n_var):
        """
        This function applies a variance filter to the EMG data

        Parameters
        ----------
        n_var : int
            The filter length of the variance filter

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: Missing (by Niklas Kueper)
        """    

        # signal init 
        emg_filtered = np.zeros(self.data.shape)

        if(emg_filtered.ndim > 1): 
            
            for channel in range(0, emg_filtered.shape[1]): 

                for index in range(0, emg_filtered.shape[0]): 

                    if (index < n_var): 
                        emg_filtered[index, channel] = 0 # just set values to zero if filterlength is not reached yet 
                    else: 
                        emg_filtered[index, channel] = np.var(self.data[index-n_var:index, channel])

        else: 
            for index in range(0, emg_filtered.shape[0]): 

                    if (index < n_var): 
                        emg_filtered[index] = 0 # just set values to zero if filterlength is not reached yet 
                    else: 
                        emg_filtered[index] = np.var(self.data[index-n_var:index])

        self.data = emg_filtered


    def epocheEMGData(self, emg_data, marker_indices, fsamp, t_start, t_stop):
        """
        This function processes the EMG data by using the epoching technique on continous data according to marker/event indices

        Parameters
        ----------
        emg_data : numpy array
            The EMG data with shape (n_samples, n_channel)
        marker_indices : int
            The marker indices i.e. the events for epoching the EMG data (e.g. can be derived from timestamps of a EEG system)
        fsamp : int
            The sampling rate of the EMG data in Hz
        t_start : int
            The start time where the epoch starts in relation to the events (marker indices) in seconds
        t_stop : int
            The stop time where the epoch ends in relation to the events (marker indices) in seconds

        Returns
        -------
        Numpy array
            emg_epochs : float
            The epoched EMG data with shape (n_epochs, n_channel, n_samples)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 31.01.2023 (by Niklas Kueper)
        """

    #     start_samp = int(t_start)*fsamp # convert and then times sample rate 
    #     stop_samp = int(t_stop)*fsamp # convert and then times sample rate
    #     len_of_epoch = start_samp-stop_samp

    #     # init array 
    #     emg_epochs = np.zeros((len(marker_indices), emg_data.shape[1], np.abs(len_of_epoch))) # emg epochs have shape (n_epochs, n_channel, n_samples) according to epochs from mne 

    #     for marker_idx in range(0, len(marker_indices)): 
    #         for channel_idx in range(0, emg_data.shape[1]):
    #             start_idx = marker_indices[marker_idx]+start_samp
    #             stop_idx = marker_indices[marker_idx]+stop_samp

    #             emg_epochs[marker_idx, channel_idx, :] = emg_data[start_idx:stop_idx, channel_idx]

    #     return emg_epochs


    def applyBPFilter(self, f_high, f_low, N = 8):
        """
        This function applies a bandpass filter to the EMG data

        Parameters
        ----------
        f_high : float
            The high cutoff frequency in Hz
        f_low : float
            The lof cutoff frequency in Hz
        N : int, optional
            The order of the filter, by default 8

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 23.03.2023 (by Niklas Kueper)
        """

        #Calc filtercoeff.  
        b1, a1 = sig.butter(N, f_high, 'high', analog=False, fs = self.__fsamp)
        b2, a2 = sig.butter(N, f_low, 'low', analog=False, fs = self.__fsamp)

    #     if (self.data.ndim > 1): 
    #         (sampels, channels) = self.data.shape
    #         emg_data_processed = np.zeros((sampels, channels))
    #         print(emg_data_processed.shape)

    #         for channel_idx in range(0, channels): 
                
    #             filtered_emg_1 = sig.filtfilt(b2, a2, self.data[:, channel_idx])
    #             filtered_emg = sig.filtfilt(b1, a1, filtered_emg_1)

    #             emg_data_processed[:, channel_idx] = filtered_emg

    #     else: 
    #         filtered_emg_1 = sig.filtfilt(b2, a2, self.data)
    #         emg_data_processed = sig.filtfilt(b1, a1, filtered_emg_1)

    #     self.data = emg_data_processed

