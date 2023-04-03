# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

def loadCometaEMGData(file_str): 

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
    

def showEMGData(emg_data, time_axis, ch_names): 

    """
    This function shows the EMG signals. For each channel one figure is created. 
    Arguments:
        emg_data_channel: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
        time_axis: The time axis of the EMG data as numpy array (shape: (n_sampels,)).
        ch_names: The EMG channel names/muscles (names specified in the recording software) as numpy array (shape: (n_channels,)). 

    Returns:
        -

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 31.01.2023 (by Niklas Kueper)
    """

    if (emg_data.ndim > 1): 
        num_channels = emg_data.shape[1]

        for n_channel in range(0, num_channels): 
            plt.figure()
            plt.plot(time_axis, emg_data[:, n_channel])
            plt.title(ch_names[n_channel])
            plt.xlabel("Time in seconds")
            plt.ylabel("Voltage in uV")
    else:
        plt.figure()
        plt.plot(time_axis, emg_data)
        plt.title(ch_names)
        plt.xlabel("Time in seconds")
        plt.ylabel("Voltage in uV")

    plt.show()


def channelSelection(emg_data, ch_names, selected_channels, inverse): 

    """
    This function can be used to specify EMG channels that are either kept or removed from the data. 
    Arguments:
        emg_data: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
        ch_names: The EMG channel names/muscles (names specified in the recording software) as numpy array (shape: (n_channels,)). 
        selected_channels: A list of EMG channel names (list of Strings) that are kept if inverse = False, otherwise the channels are dropped. 
        inverse: If False, the specified channels are kept in the EMG data, otherwise the channels are dropped. 

    Returns:
        remaining_emg_data: The selected (remaining) EMG data as numpy array (shape: (n_sampel, n_channel)). 
        remaining_emg_channels: The remaining EMG channel names as numpy array (shape: (n_channels,)). 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 31.01.2023 (by Niklas Kueper)
    """
    
    ch_indices = []
    ch_names = np.array(ch_names) # just to be sure that it is np array

    for selected_names in selected_channels: 
 
        ch_indices.append(np.where(ch_names == selected_names)[0][0])

    if(inverse == True): 
        new_ch_indices = np.arange(0, len(ch_names)-1)
        ch_indices = np.delete(new_ch_indices, np.array(ch_indices))

    remaining_emg_data = emg_data[:, ch_indices]
    if(remaining_emg_data.shape[1] == 1): # if only one channel cut of second dimension 
        remaining_emg_data = remaining_emg_data[:, 0]

    remaining_emg_channels = ch_names[ch_indices]

    return remaining_emg_data, remaining_emg_channels


def decimateEMGData(emg_data, time_axis, target_frequency, fsamp_emg): 
    
    """
    This function decimates the EMG data to a specified target frequency (uses scipy.signal.decimate() to do this).   
    Arguments:
        emg_data: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
        time_axis: The time axis of the EMG data as numpy array (shape: (n_sampels,)).
        target_frequency: The target frequency for the decimation specified in Hz. 
        fsamp_emg: The sampling rate of the EMG data in Hz. 

    Returns:
        dec_emg_data: The decimated EMG data as numpy array (shape: (n_sampel, n_channel)). 
        new_time_axis: The new time axis of the decimated EMG data as numpy array (shape: (n_sampels,)).

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 31.01.2023 (by Niklas Kueper)
    """

    down_factor = int(fsamp_emg/target_frequency)

    if(emg_data.ndim >1):
        emg_decimated = np.zeros((int(emg_data.shape[0]/down_factor), emg_data.shape[1]))

        for channel_idx in range(0, emg_data.shape[1]): 
            emg_dec = sig.decimate(emg_data[:, channel_idx], down_factor)

            # check for length differences 
            if (len(emg_dec) == emg_decimated.shape[0]):
                emg_decimated[:, channel_idx] = emg_dec
            else: 
                emg_decimated[:, channel_idx] = emg_dec[1:]

        
        dec_emg_data = emg_decimated
    else: 
        dec_emg_data = sig.decimate(emg_data, down_factor)

    # change time axis 
    dt = (time_axis[1]-time_axis[0])*down_factor
    new_time_axis = np.arange(time_axis[0], time_axis[-1], step = dt)

    return dec_emg_data, new_time_axis

 
def applyVarianceFilter(signal, n_var):

    """
    This function applies a variance filter of length n for preprocessing the EMG signals of one or multiple channels. 
    Arguments:
        signal: The EMG signal as numpy array (shape: (n_sampel, n_channel)). 
        n_var: The length (in sampels) of the variance filter. 

    Returns:
        emg_filtered: The filtered EMG signals as numpy array (shape: (n_sampel, n_channel)). The first n_var values are currently set to zero, consider e.g. padding if this is an issue. 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 31.01.2023 (by Niklas Kueper)
    """

    # signal init 
    emg_filtered = np.zeros(signal.shape)

    if(emg_filtered.ndim > 1): 
    
        for channel in range(0, emg_filtered.shape[1]): 

            for index in range(0, emg_filtered.shape[0]): 

                if (index < n_var): 
                    emg_filtered[index, channel] = 0 # just set values to zero if filterlength is not reached yet 
                else: 
                    emg_filtered[index, channel] = np.var(signal[index-n_var:index, channel])

    else: 
         for index in range(0, emg_filtered.shape[0]): 

                if (index < n_var): 
                    emg_filtered[index] = 0 # just set values to zero if filterlength is not reached yet 
                else: 
                    emg_filtered[index] = np.var(signal[index-n_var:index])

    return emg_filtered


def epocheEMGData(emg_data, marker_indices, fsamp, t_start, t_stop): 

    """
    This function processes the EMG data by using the epoching technique on continous data according to marker/event indices.  
    Arguments:
        emg_data: The EMG data as numpy array (shape: (n_sampel, n_channel)). 
        marker_indices: The marker indices i.e. the events for epoching the EMG data (e.g. can be derived from timestamps or a EEG system)
        fsamp: The sampling rate of the EMG data in Hz. 
        t_start: The start time where the epoch starts in relation to the events (marker indices) in seconds. 
        t_stop: The stop time where the epoch ends in relation to the events (marker indices) in seconds. 

    Returns:
        emg_epochs: The epoched EMG data as numpy array (shape (n_epochs, n_channel, n_samples)). 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 31.01.2023 (by Niklas Kueper)
    """

    start_samp = int(t_start)*fsamp # convert and then times sample rate 
    stop_samp = int(t_stop)*fsamp # convert and then times sample rate
    len_of_epoch = start_samp-stop_samp

    # init array 
    emg_epochs = np.zeros((len(marker_indices), emg_data.shape[1], np.abs(len_of_epoch))) # emg epochs have shape (n_epochs, n_channel, n_samples) according to epochs from mne 

    for marker_idx in range(0, len(marker_indices)): 
        for channel_idx in range(0, emg_data.shape[1]):
            start_idx = marker_indices[marker_idx]+start_samp
            stop_idx = marker_indices[marker_idx]+stop_samp

            emg_epochs[marker_idx, channel_idx, :] = emg_data[start_idx:stop_idx, channel_idx]

    return emg_epochs


def applyBPFilterRectifying(f_samp, f_high, f_low, emg_data):

    """
    This function... to be written !
       

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 23.03.2023 (by Niklas Kueper)
    """
    #Calc filtercoeff.  
    b1, a1 = sig.butter(8, f_high, 'high', analog=False, fs = f_samp)
    b2, a2 = sig.butter(8, f_low, 'low', analog=False, fs = f_samp)

    if (emg_data.ndim > 1): 
        (sampels, channels) = emg_data.shape
        emg_data_processed = np.zeros((sampels, channels))
        for channel_idx in range(0, channels): 

            filtered_emg_1 = sig.filtfilt(b2, a2, emg_data[:, channel_idx])
            filtered_emg = sig.filtfilt(b1, a1, filtered_emg_1)

            emg_data_processed[:, channel_idx] = filtered_emg

    else: 
        filtered_emg_1 = sig.filtfilt(b2, a2, emg_data)
        filtered_emg = sig.filtfilt(b1, a1, filtered_emg_1)

        emg_data_processed = filtered_emg

    return emg_data_processed
