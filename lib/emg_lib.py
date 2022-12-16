# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import matplotlib as mpl

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

def loadCometaEMGData(file_str): 

    #seperate between data, meta and channel names 
    emg_data = np.loadtxt(file_str, dtype = float, delimiter=None, skiprows=5)

    # extract EMG channel names 
    channel_names = np.loadtxt(file_str, dtype = str, delimiter=':', max_rows=1, skiprows=4)
    channel_names = channel_names[1:]
    
    emg_data_channel = emg_data[:, 1:] # channel dimensions 
    emg_time_axis = emg_data[:, 0] # time axis 

    return emg_data_channel,emg_time_axis, channel_names
    

def showEMGData(emg_data, time_axis, ch_names): 

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

    ch_indices = []

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
    
    down_factor = int(fsamp_emg/target_frequency)

    if(emg_data.ndim >1):
        emg_decimated = np.zeros((int(emg_data.shape[0]/down_factor), emg_data.shape[1]))

        for channel_idx in range(0, emg_data.shape[1]): 
            emg_decimated[:, channel_idx] = sig.decimate(emg_data[:, channel_idx], down_factor)
        
        dec_emg_data = emg_decimated
    else: 
        dec_emg_data = sig.decimate(emg_data, down_factor)

    # change time axis 
    dt = (time_axis[1]-time_axis[0])*down_factor
    new_time_axis = np.arange(time_axis[0], time_axis[-1], step = dt)

    return dec_emg_data, new_time_axis

