# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************
class EMGData:

    def __init__(self):
        pass
 

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

