# * This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

# ! ************************************************
# ! Imports
# ! ************************************************

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import os
import time

# own libs
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData

from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, createOutputDir, createReadme, \
    plotResults

import warnings

warnings.formatwarning = customWarningFormat


# ? load config file
config_filename = 'pipeline_mlp_to_cnn.yaml'
cfg = loadConfig(filename=config_filename)
save_dir = cfg.filepath.save_predictions_path

# ? Initialise arrays for appending all filenames
emg_filenames = []
quali_e_filenames = []
quali_sf_filenames = []
quali_ss_filenames = []

# ! ************************************************
# ! Get training filenames
# ! ************************************************
# Define weights and movement types
weights = cfg.data_param.weights
mov_types = cfg.data_param.mov_type

# Initialize a 2D list: rows = weights, columns = movements
emg_table = [[[] for _ in mov_types] for _ in weights]

# Fill the tables
for w_idx, wgt_idx in enumerate(weights):
    for m_idx, mov_type in enumerate(mov_types):
        for set_idx in cfg.data_param.set_num:
            emg_file_pattern = f"{cfg.filepath.emg_file_prefix}_{wgt_idx}_{mov_type}_{set_idx}.txt"

            path = cfg.filepath.data_path + cfg.filepath.emg_path + emg_file_pattern
            abs_path = getAbsolutePath(path)
            if os.path.exists(abs_path):
                print(abs_path)

            # EMG files
            emg_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.emg_path).glob(emg_file_pattern))
            if emg_files:
                emg_table[w_idx][m_idx].append(emg_files[0])

# ! ************************************************
# ! Load training, testing data
# ! ************************************************
current_idx = 0

channel_cum = []

for wgt_idx, wgt in enumerate(weights):
    for mov_idx, mov in enumerate(mov_types):
        time_preproc_start = time.perf_counter()

        # ? Loading and epoching for training
        EMG_Data = EMGData(format="ANTmini", filenames=emg_table[wgt_idx][mov_idx], data_path=cfg.filepath.data_path,
                           f_samp=cfg.preprocess_param.f_samp, channel_names=cfg.preprocess_param.channel_names_emg)

        # print(EMG_Data.events)
        # ? Plotting the raw EMG data
        if cfg.plot_param.is_plot_raw:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="uV",
                             title="Raw EMG plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in uV",
                             is_grid_on=True)



        channel_names = EMG_Data.getChannelNames()
        print("Channel Names: ", channel_names)
        print("Channel Length: ", len(channel_names))
        print("")

        # ! ************************************************
        # ! Data Pre-processing
        # ! ************************************************

        # ? High pass filter
        # * design the bandpass filter
        sos_hp = EMG_Data.designFilter(f_high=cfg.preprocess_param.f_cutoff_hpf,
                                       f_low=cfg.preprocess_param.f_cutoff_lpf,
                                       order=cfg.preprocess_param.filter_order,
                                       filter_type="scipy_butter",
                                       return_type="sos")
        # * apply bandpass filter
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                    sos=sos_hp)

        #np.save(getAbsolutePath("src/JTE_Project/offline/filter_tests/highpass_offline.npy"), EMG_Data.data)

        # ? Plotting bandpass filtered data
        if cfg.plot_param.is_plot_hpf:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="uV",
                             title="Band-Pass Filtered plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in uV",
                             is_grid_on=True)

        EMG_Data_freq = deepcopy(EMG_Data)

        # ? Apply Variance Filter from variance_tools_api
        print("Applying Variance filter...")
        width = cfg.preprocess_param.var_filter_width
        ring_buffer = np.zeros(width)
        index = 0
        EMG_Data.applyVarianceFilter_data(ring_buffer=ring_buffer,
                                          width=width,
                                          index=index)
        print("Variance Filter applied!!\n")

        # ? Plot and print specific variance filtered windows
        # var_filtered_window_x = EMG_Data.filtered_data
        # print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
        # print(f"Variance filtered windows: {var_filtered_window_x[4,:]}")

        if cfg.plot_param.is_plot_var_filter:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="uV",
                             title="Variance Filtered EMG plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in uV",
                             is_grid_on=True)

        # ? Input Normalisation
        print("Calculating the channel-wise MVC for EMG...")
        channelwise_mvc = np.max(np.abs(EMG_Data.data), axis=1)
        channel_cum.append(channelwise_mvc)


        print("Performing Input Normalization with Max Voluntary Contraction ...")
        if cfg.preprocess_param.normalisation_method == 'overall_mvc':
            EMG_Data.normalizeContinuousData(mvc=np.max(channelwise_mvc))
        elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
            EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)
        else:
            warnings.warn("This method is not yet implemented!! Omitting!")
        print("Input Normalization with Max Voluntary Contraction performed!!\n")

        #np.save(getAbsolutePath("src/JTE_Project/offline/filter_tests/normalization_offline.npy"), EMG_Data.data)


        # ? Low pass filter to smoothen the EMG signal
        # * design the lowpass filter
        sos_lp = EMG_Data.designFilter(f_low=cfg.preprocess_param.f_cutoff_sm_lpf,
                                       order=cfg.preprocess_param.sm_filter_order,
                                       filter_type="scipy_butter",
                                       return_type="sos")
        # * apply lpf filter
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                    sos=sos_lp)

        # ? Plot normalised and smoothened data
        if cfg.plot_param.is_plot_smoothed:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="V",
                             title="Normalised and Smoothed EMG plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in V",
                             is_grid_on=True)


        # ? Calculate Neural Activation Force
        if cfg.preprocess_param.use_activation_fncn:
            print("Replacing sample with its force activation value...")
            EMG_Data.calculateActivationForceFunction(d=cfg.preprocess_param.act_delay,
                                                      b1=cfg.preprocess_param.act_beta1,
                                                      b2=cfg.preprocess_param.act_beta2,
                                                      g=cfg.preprocess_param.act_gamma,
                                                      nonlinear_shape_factor=cfg.preprocess_param.act_A)
            print("Replaced each sample with its force activation value!!\n")

        # ? Plot force activation data
        if cfg.plot_param.is_plot_act:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="V",
                             title="Force Activated EMG plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in V",
                             is_grid_on=True)
                             


''' 
## Storing channelwise mvc (now the mean is used)
channel_cum = np.array(channel_cum)
print(channel_cum.shape)
# find the mean mvc per channel
channel_max = np.mean(channel_cum,axis=0)
print(channel_max.shape)
print(channel_max)
np.save(getAbsolutePath("src/JTE_Project/offline/saved_online_models/channelwise_mvc.npy"), channel_max)
'''