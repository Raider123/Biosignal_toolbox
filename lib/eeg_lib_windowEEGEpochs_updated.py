# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.utils import shuffle


# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************


def windowEEGEpochs(epochs, f_samp_eeg, windows_to_include, window_size = 1000, window_step = 50, with_channel_dim = True):

    """
    This function cuts (overlapping) windows from continues EEG-signals (currently only for postprocessing without channel dimension).

    Arguments:
        epochs: The EEG-epochs as numpy array with shape: (n_epochs, n_channel, n_samples) or for postprocessing (with_channel_dim = False) with shape:(n_trials, n_sampels).
        window_size: The size of the windows in ms to be cutout (default: 1000).
        window_step: The stepsize of the sliding window (sliding step) in ms (default: 50)
        with_channel_dim: If True (default) the epochs numpy array contains a channel dimension (standard when epoching in mne), otherwise set to False when windowing is used for other purposes e.g. postprocessing by windowing.
        windows_to_include: Specify the windows you want to work with (0 = 5000ms bis 4000ms, 1 = 4950ms bis 3950ms, ..., 80 = 1000ms bis 0ms)

    Returns:
        wind_arr: Numpy array with windowed EEG-data with shape (n_trials, n_channel, n_sampels, n_windows) if with_channel_dim = True or (n_trials, n_sampels, n_windows) for postprocessing (no channel dim)
        num_of_windows: The total number of windows that are created.
        wind_names: A list of the window names according to the pySPACE naming of window definitions.

    Meta information:
        Author: Niklas Kueper
        Last changed: 04.03.2023 added the windows_to_include option (by Christian Bojahr)
    """

    window_size_samp = int((window_size / 1000) * f_samp_eeg)
    window_step_samp = int((window_step / 1000) * f_samp_eeg)
    #print("window_size_samp", window_size_samp)
    #print("window_step_samp", window_step_samp)

    if with_channel_dim:
        epochs_arr_cut = epochs[:, :, 1:]
        num_of_windows = len(windows_to_include)
        #print("num_of_windows", num_of_windows)
        wind_arr = np.zeros((epochs_arr_cut.shape[0], epochs_arr_cut.shape[1], window_size_samp, num_of_windows))
        wind_names = []

        for i, win_nr in enumerate(windows_to_include):
            wind_start_idx = win_nr * window_step_samp
            #print("wind_start_idx", wind_start_idx)
            wind_end_idx = window_size_samp + wind_start_idx
            wind_name = "bis" + str(int((((epochs_arr_cut.shape[2] - wind_end_idx) * -1) / f_samp_eeg) * 1000))
            wind_names.append(wind_name)
            wind_arr[:, :, :, i] = epochs_arr_cut[:, :, wind_start_idx:wind_end_idx]

    else:
        epochs_arr_cut = epochs[:, 1:]
        num_of_windows = len(windows_to_include)
        #print("num_of_windows", num_of_windows)
        wind_arr = np.zeros((epochs_arr_cut.shape[0], window_size_samp, num_of_windows))
        wind_names = []

        for i, win_nr in enumerate(windows_to_include):
            wind_start_idx = win_nr * window_step_samp
            wind_end_idx = window_size_samp + wind_start_idx
            wind_name = "bis" + str(int((((epochs_arr_cut.shape[1] - wind_end_idx) * -1) / f_samp_eeg) * 1000))
            wind_names.append(wind_name)
            wind_arr[:, :, i] = epochs_arr_cut[:, wind_start_idx:wind_end_idx]

    #print(wind_arr)
    return wind_arr, num_of_windows, wind_names




