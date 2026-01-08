# * This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

# ! ************************************************
# ! Imports
# ! ************************************************

import numpy as np
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import joblib
import random
from collections import defaultdict
import os
import time

# own libs
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, createOutputDir, createReadme, \
    plotResults

import warnings

warnings.formatwarning = customWarningFormat

# Für Keras / Model-Train
from tensorflow.keras import Input, layers, models, optimizers
# Für Post-Filter (nutze SciPy falls vorhanden)
from scipy.signal import medfilt, savgol_filter

# ! ************************************************
# ! User Parameters and Data Collection
# ! ************************************************


if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.reset_memory_stats('GPU:0')


def build_model_old(input_shape_time, filters, stacks, dropout_rate, kernel_size):
    """
    Builds a Temporal Convolutional Network (TCN) for multi-task torque estimation.

    This model uses dilated causal convolutions to learn temporal patterns from EMG
    signals without looking into the future (causal), making it suitable for
    real-time applications.

    Args:
        input_shape_time (tuple): Shape of input data (TimeSteps, Channels).
        filters (int): Number of filters (feature maps) in convolutional layers.
        stacks (int): Number of residual blocks. Determines the 'memory' (receptive field).
        dropout_rate (float): Spatial dropout rate for regularization (0.0 - 1.0).
        kernel_size (int): Length of the temporal convolution kernel.

    Returns:
        tf.keras.Model: A compiled Keras functional model with 3 output heads.
    """

    # ---------------------------------------------------------
    # 1. Input Layer
    # ---------------------------------------------------------
    # Expects shape: (Batch, TimeSteps, Channels)
    # e.g., (None, 50, 8) for 50 samples history and 8 muscles.
    inp_time = Input(shape=input_shape_time, name='emg_input')
    x = inp_time

    # ---------------------------------------------------------
    # 2. TCN Backbone (Temporal Feature Extraction)
    # ---------------------------------------------------------
    # We stack multiple residual blocks. Each block looks further back in time
    # due to the increasing dilation rate (1, 2, 4, 8...).
    for s in range(stacks):
        dilation_rate = 2 ** s  # Exponential dilation: 1, 2, 4, 8...

        # --- Start of Residual Block ---
        # Save the input 'x' for the residual connection later
        residual = x

        # -- First Convolution Layer --
        # 'causal' padding ensures we only use past data (no future peeking).
        y = layers.Conv1D(filters,
                          kernel_size,
                          padding='causal',
                          dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(x)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)  # Drops entire feature maps

        # -- Second Convolution Layer --
        y = layers.Conv1D(filters,
                          kernel_size,
                          padding='causal',
                          dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(y)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)

        # -- Residual Connection (Skip Connection) --
        # If the number of filters changed (or at the first block), project 'x'
        # to match the shape of 'y' using a 1x1 convolution.
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding='same',
                                     kernel_initializer='he_normal')(residual)

        # Add the original input to the processed output (ResNet principle)
        x = layers.add([residual, y])
        # --- End of Residual Block ---

    # ---------------------------------------------------------
    # 3. Readout (Temporal Aggregation)
    # ---------------------------------------------------------
    # We strictly take only the LAST time step.
    # Because of causal padding and dilation, this last step effectively
    # contains the aggregated information of the entire input window.
    x = layers.Lambda(lambda t: t[:, -1, :], name='last_step_readout')(x)

    # ---------------------------------------------------------
    # 4. Dense Layers & Multi-Task Output
    # ---------------------------------------------------------
    # High-level feature processing
    combined = layers.Dense(64, activation="relu")(x)
    combined = layers.Dropout(dropout_rate)(combined)

    # Output Heads: One regression output for each joint torque
    out_e = layers.Dense(1, name='torque_elbow')(combined)
    out_f = layers.Dense(1, name='torque_shoulder_front')(combined)
    out_s = layers.Dense(1, name='torque_shoulder_side')(combined)

    return models.Model(inputs=inp_time, outputs=[out_e, out_f, out_s], name="MTL_TCN")


def build_model(input_shape_time, input_shape_static, filters, stacks, dropout_rate, kernel_size):
    """
    Builds a TCN with dual inputs:
    1. Time-series EMG data (processed via TCN)
    2. Static categorical data (concatenated before Dense layers)
    """

    # --- Input 1: Time Series (EMG) ---
    inp_time = Input(shape=input_shape_time, name='emg_input')
    x = inp_time

    # ... TCN Backbone (wie bisher) ...
    for s in range(stacks):
        dilation_rate = 2 ** s
        residual = x
        y = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(x)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)

        y = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(y)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)

        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding='same', kernel_initializer='he_normal')(residual)

        x = layers.add([residual, y])

    # Readout (Last Timestep)
    x = layers.Lambda(lambda t: t[:, -1, :], name='last_step_readout')(x)

    # --- Input 2: Static Features (One-Hot) ---
    inp_static = Input(shape=input_shape_static, name='static_input')

    # --- Merge ---
    # Hier werden die extrahierten Zeit-Features mit den statischen Infos kombiniert
    combined_features = layers.concatenate([x, inp_static])

    # --- Dense Layers ---
    combined = layers.Dense(64, activation="relu")(combined_features)
    combined = layers.Dropout(dropout_rate)(combined)

    # Output Heads
    out_e = layers.Dense(1, name='torque_elbow')(combined)
    out_f = layers.Dense(1, name='torque_shoulder_front')(combined)
    out_s = layers.Dense(1, name='torque_shoulder_side')(combined)

    return models.Model(inputs=[inp_time, inp_static], outputs=[out_e, out_f, out_s], name="MTL_TCN_DualInput")

time_preproc = 0
time_feat = 0

# ? load config file
config_filename = 'pipeline_jte_bu62d.yaml'
cfg = loadConfig(filename=config_filename)
save_dir = cfg.filepath.save_predictions_path
print("Using the following CONFIG FILE: ", config_filename)

# ? init early stopping
if cfg.model_param.is_early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=cfg.model_param.monitor,
                                                      min_delta=cfg.model_param.min_delta,
                                                      patience=cfg.model_param.patience,
                                                      verbose=cfg.model_param.verbose,
                                                      baseline=cfg.model_param.baseline,
                                                      restore_best_weights=cfg.model_param.restore_best_weights)
else:
    early_callback = None

# ? init performance results list
perf_results_total_MLP = []

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
quali_e_table = [[[] for _ in mov_types] for _ in weights]
quali_sf_table = [[[] for _ in mov_types] for _ in weights]
quali_ss_table = [[[] for _ in mov_types] for _ in weights]

# Fill the tables
for w_idx, wgt_idx in enumerate(weights):
    for m_idx, mov_type in enumerate(mov_types):
        for set_idx in cfg.data_param.set_num:
            emg_file_pattern = f"{cfg.filepath.emg_file_prefix}_{wgt_idx}_{mov_type}_{set_idx}.txt"
            quali_e_file_pattern = f"{cfg.filepath.quali_torque_prefix[0]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_type}_{set_idx}.npy"
            quali_sf_file_pattern = f"{cfg.filepath.quali_torque_prefix[1]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_type}_{set_idx}.npy"
            quali_ss_file_pattern = f"{cfg.filepath.quali_torque_prefix[2]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_type}_{set_idx}.npy"

            path = cfg.filepath.data_path + cfg.filepath.emg_path + emg_file_pattern
            abs_path = getAbsolutePath(path)
            if os.path.exists(abs_path):
                print(abs_path)

            # EMG files
            emg_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.emg_path).glob(emg_file_pattern))
            if emg_files:
                emg_table[w_idx][m_idx].append(emg_files[0])

            # Quali E
            quali_e_files = list(
                getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_e_file_pattern))
            if quali_e_files:
                quali_e_table[w_idx][m_idx].append(quali_e_files[0])

            # Quali SF
            quali_sf_files = list(
                getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_sf_file_pattern))
            if quali_sf_files:
                quali_sf_table[w_idx][m_idx].append(quali_sf_files[0])

            # Quali SS
            quali_ss_files = list(
                getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_ss_file_pattern))
            if quali_ss_files:
                quali_ss_table[w_idx][m_idx].append(quali_ss_files[0])

# ! ************************************************
# ! Load training, testing data (PHASE 1: LOAD & PRE-FILTER)
# ! ************************************************
X_train_combined = []
Y_train_combined = []

X_test_combined = []
Y_test_combined = []

X_val_combined = []
Y_val_combined = []

Y_scaler_info = []
Y_scaler_dict = defaultdict(lambda: defaultdict(lambda: None))

meta_list_train = []
meta_list_test = []
meta_list_val = []

current_idx = 0

X_train_feats_list = []
X_test_feats_list = []
X_val_feats_list = []

# Container to store objects before Normalization to avoid Data Leakage
data_containers = []

print("PHASE 1: Loading Data and Applying Linear Filters (BPF, VarFilter)...")

for wgt_idx, wgt in enumerate(weights):
    for mov_idx, mov in enumerate(mov_types):
        #######
        # if certain emg files are excluded, the script will not crash (allows for singular files)
        if not emg_table[wgt_idx][mov_idx]:
            print(f"WARNUNG: Keine EMG-Dateien gefunden für Gewicht: {wgt}, Bewegung: {mov}. Überspringe...")
            continue
        #######
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

        # #? Loading the target values for the 3 joints
        # print("Creating Quali Elbow object...")
        Quali_Data_Elbow = EEGData(format="NumpyQualisys",
                                   filenames=quali_e_table[wgt_idx][mov_idx],
                                   data_path=cfg.filepath.data_path,
                                   f_samp=cfg.preprocess_param.f_samp,
                                   channel_names=cfg.preprocess_param.channel_names_quali,
                                   add_marker_channel=True)

        # print("Creating Quali Shoulder Front object...")
        Quali_Data_Front = EEGData(format="NumpyQualisys",
                                   filenames=quali_sf_table[wgt_idx][mov_idx],
                                   data_path=cfg.filepath.data_path,
                                   f_samp=cfg.preprocess_param.f_samp,
                                   channel_names=cfg.preprocess_param.channel_names_quali,
                                   add_marker_channel=True)

        # print("Creating Quali Shoulder Side object...")
        Quali_Data_Side = EEGData(format="NumpyQualisys",
                                  filenames=quali_ss_table[wgt_idx][mov_idx],
                                  data_path=cfg.filepath.data_path,
                                  f_samp=cfg.preprocess_param.f_samp,
                                  channel_names=cfg.preprocess_param.channel_names_quali,
                                  add_marker_channel=True)

        channel_names = EMG_Data.getChannelNames()
        # print("Channel Names: ", channel_names)
        # print("Channel Length: ", len(channel_names))
        # print("")

        # print(Quali_Data_Elbow.data.shape)
        if cfg.plot_param.is_plot_quali:
            plt.figure()
            plt.plot(np.arange(0, Quali_Data_Elbow.data[0, :].shape[0], 1) / Quali_Data_Elbow.f_samp,
                     Quali_Data_Elbow.data[0, :])
            plt.title("Elbow Torque plot for right arm")
            plt.xlabel("Time in s")
            plt.ylabel("Torque in N-m")
            plt.grid()
            plt.show()

        # ! ************************************************
        # ! Data Pre-processing (Filters only)
        # ! ************************************************

        # ? Band pass filter
        # * design the bandpass filter
        sos_hp = EMG_Data.designFilter(f_high=cfg.preprocess_param.f_cutoff_hpf,
                                       f_low=cfg.preprocess_param.f_cutoff_lpf,
                                       order=cfg.preprocess_param.filter_order,
                                       filter_type="scipy_butter",
                                       return_type="sos")
        # * apply bandpass filter
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                    sos=sos_hp)
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
        # print("Applying Variance filter...")
        width = cfg.preprocess_param.var_filter_width
        ring_buffer = np.zeros(width)
        index = 0
        EMG_Data.applyVarianceFilter_data(ring_buffer=ring_buffer,
                                          width=width,
                                          index=index)
        # print("Variance Filter applied!!\n")

        if cfg.plot_param.is_plot_var_filter:
            EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                             unit="uV",
                             title="Variance Filtered EMG plot for Channel 5",
                             xlabel="Time in s",
                             ylabel="Voltage in uV",
                             is_grid_on=True)

        # STOP! MVC calculation needs to happen globally now.
        # Store processed objects for Phase 2
        data_containers.append({
            'wgt': wgt,
            'mov': mov,
            'EMG_Data': EMG_Data,
            'EMG_Data_freq': EMG_Data_freq,
            'Quali_Data_Elbow': Quali_Data_Elbow,
            'Quali_Data_Front': Quali_Data_Front,
            'Quali_Data_Side': Quali_Data_Side
        })

# ! ************************************************
# ! PHASE 2: CALCULATE MVC ON TRAINING DATA ONLY
# ! ************************************************
print("PHASE 2: Calculating MVC (Training Split Only)...")

mvc_save_path = str(getAbsolutePath("src/JTE_Project/offline/saved_offline_models/channelwise_mvc.npy"))

if not cfg.model_param.load_models:
    # 1. Temporarily concatenate all EMG data to simulate the full dataset
    # EMG_Data.data shape is (Channels, Samples)
    all_emg_data = np.concatenate([d['EMG_Data'].data for d in data_containers], axis=1)
    total_samples = all_emg_data.shape[1]

    # 2. Determine split index (assuming time-series split as per shuffle=False later)
    # Note: Later we split features, here we split raw samples.
    # Because of windowing, this is an approximation, but strictly prevents leakage.
    # For Online we can use all of this samples mvc_split = 1, in offline: cfg.model_param.train_test_split
    mvc_split = 1
    train_split_idx = int(total_samples * mvc_split)

    # 3. Extract Training Data Portion
    emg_train_raw = all_emg_data[:, :train_split_idx]

    # 4. Calculate MVC on Training Data
    print(f"Calculating MVC on first {train_split_idx} samples (Train Set) out of {total_samples}...")
    channelwise_mvc = np.max(np.abs(emg_train_raw), axis=1).reshape(-1, 1)

    # Save for later/online use
    os.makedirs(os.path.dirname(mvc_save_path), exist_ok=True)
    np.save(mvc_save_path, channelwise_mvc)

else:
    print("Loading pre-calculated MVC...")
    channelwise_mvc = np.load(mvc_save_path)
    # Ensure shape
    channelwise_mvc = channelwise_mvc.reshape(8, 1)

print(f"MVC Values used: \n{channelwise_mvc.flatten()}")

# ! ************************************************
# ! PHASE 3: NORMALIZE & FEATURE EXTRACTION
# ! ************************************************
print("PHASE 3: Normalization, Smoothing, Windowing, Feature Extraction...")

for entry in data_containers:
    # Unpack objects
    wgt = entry['wgt']
    mov = entry['mov']
    EMG_Data = entry['EMG_Data']
    EMG_Data_freq = entry['EMG_Data_freq']
    Quali_Data_Elbow = entry['Quali_Data_Elbow']
    Quali_Data_Front = entry['Quali_Data_Front']
    Quali_Data_Side = entry['Quali_Data_Side']

    time_preproc_start = time.perf_counter()  # Restart timing for this phase

    # print(f"Performing Input Normalization with Max Voluntary Contraction for {wgt} {mov}...")

    if cfg.preprocess_param.normalisation_method == 'overall_mvc':
        EMG_Data.normalizeContinuousData(mvc=np.max(channelwise_mvc))
    elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
        EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)
    else:
        warnings.warn("This method is not yet implemented!! Omitting!")

    # print("Input Normalization performed!!\n")

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

    # ? Low pass filter to smoothen the torques
    Quali_Data_Elbow.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                        sos=sos_lp)
    Quali_Data_Front.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                        sos=sos_lp)
    Quali_Data_Side.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                       sos=sos_lp)

    # ? Plot normalised and smoothened data
    if cfg.plot_param.is_plot_smoothed:
        EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                         unit="V",
                         title="Normalised and Smoothed EMG plot for Channel 5",
                         xlabel="Time in s",
                         ylabel="Voltage in V",
                         is_grid_on=True)
        Quali_Data_Side.plotEMG(data=Quali_Data_Side.data[0, :],
                                unit="V",
                                title="Normalised and Smoothed Elbow Torques plot",
                                xlabel="Time in s",
                                ylabel="Torque in N-m",
                                is_grid_on=True)

    # ? Calculate Neural Activation Force
    if cfg.preprocess_param.use_activation_fncn:
        # print("Replacing sample with its force activation value...")
        EMG_Data.calculateActivationForceFunction(d=cfg.preprocess_param.act_delay,
                                                  b1=cfg.preprocess_param.act_beta1,
                                                  b2=cfg.preprocess_param.act_beta2,
                                                  g=cfg.preprocess_param.act_gamma,
                                                  nonlinear_shape_factor=cfg.preprocess_param.act_A)
        # print("Replaced each sample with its force activation value!!\n")

    # ? Plot force activation data
    if cfg.plot_param.is_plot_act:
        EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                         unit="V",
                         title="Force Activated EMG plot for Channel 5",
                         xlabel="Time in s",
                         ylabel="Voltage in V",
                         is_grid_on=True)

    time_preproc_end = time.perf_counter()
    time_preproc += (time_preproc_end - time_preproc_start)

    # ? Windowing the data
    emg_window_boundary_idx, _ = EMG_Data.windowContinuousData(startmarkernumber=1,
                                                               stopmarkernumber=2,
                                                               window_size=cfg.preprocess_param.window_size_x,
                                                               window_step=cfg.preprocess_param.window_step,
                                                               start_index_offset=0,
                                                               start_channel_pick=0,
                                                               end_channel_pick=8,
                                                               return_window_end_indices=True)

    _, _ = EMG_Data_freq.windowContinuousData(startmarkernumber=1,
                                              stopmarkernumber=2,
                                              window_size=cfg.preprocess_param.window_size_x,
                                              window_step=cfg.preprocess_param.window_step,
                                              start_index_offset=0,
                                              start_channel_pick=0,
                                              end_channel_pick=8,
                                              return_window_end_indices=True)

    quali_window_boundary_idx, _ = Quali_Data_Elbow.windowContinuousData(startmarkernumber=1,
                                                                         stopmarkernumber=2,
                                                                         window_size=cfg.preprocess_param.window_size_y,
                                                                         window_step=cfg.preprocess_param.window_step,
                                                                         start_index_offset=0,
                                                                         start_channel_pick=0,
                                                                         end_channel_pick=1,
                                                                         return_window_end_indices=True)

    _, _ = Quali_Data_Front.windowContinuousData(startmarkernumber=1,
                                                 stopmarkernumber=2,
                                                 window_size=cfg.preprocess_param.window_size_y,
                                                 window_step=cfg.preprocess_param.window_step,
                                                 start_index_offset=0,
                                                 start_channel_pick=0,
                                                 end_channel_pick=1,
                                                 return_window_end_indices=True)

    _, _ = Quali_Data_Side.windowContinuousData(startmarkernumber=1,
                                                stopmarkernumber=2,
                                                window_size=cfg.preprocess_param.window_size_y,
                                                window_step=cfg.preprocess_param.window_step,
                                                start_index_offset=0,
                                                start_channel_pick=0,
                                                end_channel_pick=1,
                                                return_window_end_indices=True)

    # ? Ensure the number of windows of each file are the same for inp and target
    emg_window_len = EMG_Data.getWindows().shape[3]
    quali_window_len = Quali_Data_Elbow.getWindows().shape[3]
    if emg_window_len != quali_window_len:
        print(f" Window mismatch in {wgt},{mov}... EMG: {emg_window_len} and Quali: {quali_window_len}!!")
        min_windows = min(emg_window_len, quali_window_len)
        EMG_Data.windows = EMG_Data.windows[..., :min_windows]
        EMG_Data_freq.windows = EMG_Data.windows[..., :min_windows]

        Quali_Data_Elbow.windows = Quali_Data_Elbow.windows[..., :min_windows]
        Quali_Data_Front.windows = Quali_Data_Front.windows[..., :min_windows]
        Quali_Data_Side.windows = Quali_Data_Side.windows[..., :min_windows]
        print(f"Windows trimmed to {min_windows} windows!!")

    print("EMG_Data Window Shape ", EMG_Data.getWindows().shape)

    # ? One Hot Encoding for 3 weights and 2 movements
    weights_code = []
    mov_code = []

    if wgt == '0g':
        weights_code += [0] * EMG_Data.getWindows().shape[3]
    elif wgt == '1100g':
        weights_code += [1] * EMG_Data.getWindows().shape[3]
    elif wgt == '1850g':
        weights_code += [2] * EMG_Data.getWindows().shape[3]
    else:
        raise ValueError("Wrong weight string added... weight should be either 0g, 1100g or 1850g!!")

    if mov == 'grasp':
        mov_code += [0] * EMG_Data.getWindows().shape[3]
    elif mov == 'complex':
        mov_code += [1] * EMG_Data.getWindows().shape[3]
    else:
        raise ValueError("Wrong mov type string added... move should be either grasp or complex!!")

    # ? Merge weights_code and mov_code to form categorical feat set
    x_cat = np.column_stack([weights_code, mov_code])

    # ? Plot specific filtered windows for debugging
    if cfg.plot_param.is_plot_filt_win:
        EMG_Data.plotEMG(data=EMG_Data.getWindows()[0, 4, :, 18],  # [trl,chn,smpl,wnd]
                         n_samples=EMG_Data.getWindows().shape[2],
                         unit="V",
                         title="Pre-processed EMG plot for Channel 5 Window 18",
                         xlabel="Time in s",
                         ylabel="Voltage in V",
                         is_grid_on=True)

    # ! ************************************************
    # ! Feature Extraction
    # ! ************************************************
    print("Extracting features from windowed data...")
    time_feat_start = time.perf_counter()

    # ? EMG signal timepoints feature extraction
    window_size_ms = cfg.preprocess_param.window_size_x * 1000 / EMG_Data.f_samp
    feature_indices_windows_x = np.array([0, window_size_ms])
    EMG_Data.featureExtractionFromWindows(feature_type="timepoints",
                                          feature_indices_windows=feature_indices_windows_x)
    EMG_Data.printFeatureShape()

    if cfg.settings.feature_extraction:
        current_file_feats = []

        # ! ***************************************************************
        # ! 1. Time Domain Features
        # ! ***************************************************************
        '''
        # T1: RMS (Root Mean Square)
        rms = EMG_Data.getRMSFeatures_windows(n_channels=len(channel_names))
        current_file_feats.append(rms)

        # T2: WFL (Waveform Length)
        wfl = EMG_Data.getWaveformLengthFeatures_windows(n_channels=len(channel_names))
        current_file_feats.append(wfl)

        # T3: SSC (Slope Sign Change)
        ssc = EMG_Data.getSlopeSignChangeFeatures_windows(n_channels=len(channel_names), threshold=0.02)
        current_file_feats.append(ssc)

        # ! ***************************************************************
        # ! 2. Frequency Domain Features (Using EMG_Data_freq Copy)
        # ! ***************************************************************
        # We use the 'EMG_Data_freq' copy to avoid messing up the time-series
        # data in the main 'EMG_Data' object used for the TCN input.

        # F1: Frequency Band Power (PSD via Multitaper)
        EMG_Data_freq.featureExtractionFromWindows(feature_type="freqBandPower",
                                                   psd_method="multitaper",
                                                   freq_bands=[15, 50, 100, 150, 200, 245])

        # Retrieve the calculated features from the freq object
        fbp_feature = EMG_Data_freq.getFeatures()
        current_file_feats.append(fbp_feature)

        
        # F2: Morlet Wavelet Coefficients (Time-Frequency)
        freqs = np.arange(start=50, stop=226, step=25)
        n_cycles = np.ones(len(freqs)) * 5
        n_cycles[0] = 3
        n_cycles[1] = 4

        mwc_feature = EMG_Data_freq.getMorletWaveletCoeffFeatures_windows(freqs=freqs,
                                                                          n_cycles=n_cycles)
        current_file_feats.append(mwc_feature)
        '''

        # ? Peak Detection:: Change between consecutive samples (window i and wind i+1)
        peak_detection = np.diff(EMG_Data.getFeatures(), axis=0, prepend=EMG_Data.getFeatures()[0:1, :])
        current_file_feats.append(peak_detection)

        # ! ***************************************************************
        # ! 3. Temporal Change Frequency Features (Differentiation)
        # ! ***************************************************************
        '''
        # Combine all features collected so far to calculate their derivative
        features_so_far = np.concatenate(current_file_feats, axis=1)

        # Calculate the change between consecutive windows (velocity of features)
        feat_diff = np.diff(features_so_far, axis=0, prepend=features_so_far[0:1, :])

        current_file_feats.append(feat_diff)
        '''
        # ! ***************************************************************
        # ! 4. Final Merge & Splitting
        # ! ***************************************************************

        # Concatenate all features for this specific file
        X_feat_raw = np.concatenate(current_file_feats, axis=1)

        # Split 1: Train (temp) / Test
        X_train_feat_temp, X_test_feat_curr = train_test_split(
            X_feat_raw,
            train_size=cfg.model_param.train_test_split,
            shuffle=False
        )

        # Split 2: Train / Val
        X_train_feat_curr, X_val_feat_curr = train_test_split(
            X_train_feat_temp,
            train_size=1 - cfg.model_param.validation_split,
            shuffle=False
        )

        if cfg.settings.advanced_pipeline == False:
            # Store file specific feature list in global lists (defined before the loop)
            X_train_feats_list.append(X_train_feat_curr)
            X_test_feats_list.append(X_test_feat_curr)
            X_val_feats_list.append(X_val_feat_curr)

    # ? Output feature extraction
    window_size_ms = cfg.preprocess_param.window_size_y * 1000 / Quali_Data_Elbow.f_samp
    if cfg.preprocess_param.target_feature_select == 'mean':
        feature_indices_windows_y = np.array([0, window_size_ms])
        use_mean_bool = True
    elif cfg.preprocess_param.target_feature_select == 'mid':
        feature_indices_windows_y = np.array([(window_size_ms / 2) - 2, window_size_ms / 2])
        use_mean_bool = False
    elif cfg.preprocess_param.target_feature_select == 'end':
        feature_indices_windows_y = np.array([window_size_ms - 2, window_size_ms])
        use_mean_bool = False
    else:
        raise ValueError(
            f"Provided target_feature_select {cfg.preprocess_param.target_feature_select} is not yet implemented... Please choose between 'mean', 'mid', and 'end'")

    Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints",
                                                  feature_indices_windows=feature_indices_windows_y,
                                                  use_mean=use_mean_bool)
    Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints",
                                                  feature_indices_windows=feature_indices_windows_y,
                                                  use_mean=use_mean_bool)
    Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints",
                                                 feature_indices_windows=feature_indices_windows_y,
                                                 use_mean=use_mean_bool)
    # print("Feature extraction from windowed data completed!!\n")

    time_feat_end = time.perf_counter()
    time_feat += (time_feat_end - time_feat_start)

    # ? Get the input features from EMG_Data
    input_features = EMG_Data.getFeatures()

    # ? Merge output features
    target_features = np.concatenate(
        [Quali_Data_Elbow.getFeatures(), Quali_Data_Front.getFeatures(), Quali_Data_Side.getFeatures()], axis=1)

    # ? Print input feature and target feature length
    # print(f"Input feature shape (pre-merge): {EMG_Data.getFeatures().shape}")
    # print(f"Target feature shape (pre-merge): {target_features.shape}")

    # ? Split data into train, validation, and test sets
    X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(input_features,
                                                                  target_features,
                                                                  train_size=cfg.model_param.train_test_split,
                                                                  shuffle=False)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp,
                                                      Y_train_temp,
                                                      train_size=1 - cfg.model_param.validation_split,
                                                      shuffle=False)

    if cfg.settings.advanced_pipeline:
        # ? Split categorical data into train, validation, and test sets
        X_train_cat_temp, X_test_cat = train_test_split(x_cat,
                                                        train_size=cfg.model_param.train_test_split,
                                                        shuffle=False)

        X_train_cat, X_val_cat, = train_test_split(X_train_cat_temp,
                                                   train_size=1 - cfg.model_param.validation_split,
                                                   shuffle=False)

        if cfg.settings.feature_extraction:
            # Use StackHistory for the features too because they are concatenated with the one-hot-encoded features
            history_len = 1
            X_train_feat_curr, _, _ = EMG_Data.stackHistoryCatMeta_windows(x_num=X_train_feat_curr, y_num=Y_train,
                                                                           x_cat=X_train_cat, history_len=history_len,
                                                                           wgt=wgt, mov=mov)
            X_test_feat_curr, _, _ = EMG_Data.stackHistoryCatMeta_windows(x_num=X_test_feat_curr, y_num=Y_test,
                                                                          x_cat=X_test_cat, history_len=history_len,
                                                                          wgt=wgt, mov=mov)
            X_val_feat_curr, _, _ = EMG_Data.stackHistoryCatMeta_windows(x_num=X_val_feat_curr, y_num=Y_val,
                                                                         x_cat=X_val_cat, history_len=history_len,
                                                                         wgt=wgt,
                                                                         mov=mov)

            X_train_feats_list.append(X_train_feat_curr)
            X_test_feats_list.append(X_test_feat_curr)
            X_val_feats_list.append(X_val_feat_curr)

        # ? One Hot encoding
        encoder = OneHotEncoder(sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train_cat)
        X_test_cat = encoder.transform(X_test_cat)
        X_val_cat = encoder.transform(X_val_cat)

        # ? Creating history of features (deactivated with 1)
        history_len = 1
        X_train, Y_train, meta_train = EMG_Data.stackHistoryCatMeta_windows(x_num=X_train,
                                                                            y_num=Y_train,
                                                                            x_cat=X_train_cat,
                                                                            history_len=history_len,
                                                                            wgt=wgt,
                                                                            mov=mov)

        X_test, Y_test, meta_test = EMG_Data.stackHistoryCatMeta_windows(x_num=X_test,
                                                                         y_num=Y_test,
                                                                         x_cat=X_test_cat,
                                                                         history_len=history_len,
                                                                         wgt=wgt,
                                                                         mov=mov)
        X_val, Y_val, meta_val = EMG_Data.stackHistoryCatMeta_windows(x_num=X_val,
                                                                      y_num=Y_val,
                                                                      x_cat=X_val_cat,
                                                                      history_len=history_len,
                                                                      wgt=wgt,
                                                                      mov=mov)

    if cfg.settings.yscaler:
        Y_scaler, Y_train, Y_test, Y_val = EMG_Data.scaleFeatures_windows(train_data=Y_train,
                                                                          test_data=Y_test,
                                                                          val_data=Y_val,
                                                                          method="MinMaxScaler",
                                                                          feature_range=(-1, 1))

    start_idx = current_idx
    end_idx = current_idx + Y_test.shape[0]

    if cfg.settings.yscaler:
        Y_scaler_dict[wgt][mov] = Y_scaler
    Y_scaler_info.append((wgt, mov, start_idx, end_idx))
    current_idx = end_idx

    X_train_combined.append(X_train)
    Y_train_combined.append(Y_train)

    X_test_combined.append(X_test)
    Y_test_combined.append(Y_test)

    X_val_combined.append(X_val)
    Y_val_combined.append(Y_val)

    if cfg.settings.advanced_pipeline:
        meta_list_train.extend(meta_train)
        meta_list_test.extend(meta_test)
        meta_list_val.extend(meta_val)

X_train = np.concatenate(X_train_combined, axis=0)
Y_train = np.concatenate(Y_train_combined, axis=0)

X_test = np.concatenate(X_test_combined, axis=0)
Y_test = np.concatenate(Y_test_combined, axis=0)

X_val = np.concatenate(X_val_combined, axis=0)
Y_val = np.concatenate(Y_val_combined, axis=0)

if cfg.settings.feature_extraction:
    # Frequency Features
    X_train_feats = np.concatenate(X_train_feats_list, axis=0)
    X_test_feats = np.concatenate(X_test_feats_list, axis=0)
    X_val_feats = np.concatenate(X_val_feats_list, axis=0)
    from sklearn.preprocessing import StandardScaler
    feat_scaler = StandardScaler()
    X_train_feats_scaled = feat_scaler.fit_transform(X_train_feats)
    X_test_feats_scaled = feat_scaler.transform(X_test_feats)
    X_val_feats_scaled = feat_scaler.transform(X_val_feats)


# Calculate Scaling Time
time_scaling_start = time.perf_counter()

if cfg.settings.scaling:
    # ? Pre-PCA scaling of input features
    pre_emg_scaler, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train,
                                                                            test_data=X_test,
                                                                            val_data=X_val,
                                                                            method="StandardScaler")

    import joblib

    # joblib.dump(pre_emg_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/pre_emg_scaler.pkl"))

    # ? Dimensionality Reduction - PCA
    pca_scaler, X_train, X_test, X_val = EMG_Data.reduceDimensions_windows(train_data=X_train,
                                                                           test_data=X_test,
                                                                           val_data=X_val,
                                                                           method="PCA",
                                                                           n_components=0.99,
                                                                           mode="offline")

    # joblib.dump(pca_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/pca_scaler.pkl"))

    # ? Scale the input features -> StandardScaler
    post_emg_scaler, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train,
                                                                             test_data=X_test,
                                                                             val_data=X_val,
                                                                             method="StandardScaler")

    # joblib.dump(post_emg_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/post_emg_scaler.pkl"))

time_scaling_end = time.perf_counter()
time_scaling = time_scaling_end - time_scaling_start

if cfg.settings.advanced_pipeline:
    # ? One-hot encoding
    wgt_train, mov_train = EMG_Data.convertMetaToArray(meta_list_train)
    wgt_val, mov_val = EMG_Data.convertMetaToArray(meta_list_val)
    wgt_test, mov_test = EMG_Data.convertMetaToArray(meta_list_test)

    weights_categories = [['0g', '1100g', '1850g']]
    moves_categories = [['grasp', 'complex']]

    encoder_wgt = OneHotEncoder(sparse_output=False, categories=weights_categories)
    wgt_train_onehot = encoder_wgt.fit_transform(wgt_train)
    wgt_test_onehot = encoder_wgt.transform(wgt_test)
    wgt_val_onehot = encoder_wgt.transform(wgt_val)

    encoder_mov = OneHotEncoder(sparse_output=False, categories=moves_categories)
    mov_train_onehot = encoder_mov.fit_transform(mov_train)
    mov_test_onehot = encoder_mov.transform(mov_test)
    mov_val_onehot = encoder_mov.transform(mov_val)

    X_train_onehot = np.concatenate([wgt_train_onehot, mov_train_onehot], axis=1)
    X_test_onehot = np.concatenate([wgt_test_onehot, mov_test_onehot], axis=1)
    X_val_onehot = np.concatenate([wgt_val_onehot, mov_val_onehot], axis=1)
    print(f"One-Hot-Feature Shape: {X_train_onehot.shape}")


# Create Static Feature Set
if cfg.settings.feature_extraction and cfg.settings.advanced_pipeline:
    X_train_static = np.concatenate([X_train_feats_scaled, X_train_onehot], axis=1)
    X_test_static = np.concatenate([X_test_feats_scaled, X_test_onehot], axis=1)
    X_val_static = np.concatenate([X_val_feats_scaled, X_val_onehot], axis=1)
    print(f"Final Static Input Shape: {X_train_static.shape}")
elif cfg.settings.feature_extraction and cfg.settings.advanced_pipeline == False:
    X_train_static = X_train_feats_scaled
    X_test_static = X_test_feats_scaled
    X_val_static = X_val_feats_scaled
    print(f"Final Static Input Shape: {X_train_static.shape}")
elif cfg.settings.feature_extraction == False and cfg.settings.advanced_pipeline:
    X_train_static = X_train_onehot
    X_test_static = X_test_onehot
    X_val_static = X_val_onehot
    print(f"Final Static Input Shape: {X_train_static.shape}")

# ? Shuffle training sets
perm = np.random.permutation(X_train.shape[0])
X_train[:] = X_train[perm]
Y_train[:] = Y_train[perm]
if cfg.settings.advanced_pipeline or cfg.settings.feature_extraction:
    X_train_static[:] = X_train_static[perm]

# ! ************************************************
# ! Train, Load, or Test Model
# ! ************************************************

# --- set global seed ---
seed_arr = [1]

r2_e_arr = []
r2_sf_arr = []
r2_ss_arr = []

# for prefiltering
rho_e_arr = []
rho_sf_arr = []
rho_ss_arr = []

# for postfiltering
rho_e_arr_pf = []
r2_e_arr_pf = []

rho_sf_arr_pf = []
r2_sf_arr_pf = []

r2_ss_arr_pf = []
rho_ss_arr_pf = []

# for timings
time_train = []
time_prediction = []

for seed in seed_arr:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # ---------------------------
    # Reshaping the input vector for the model training
    # ---------------------------
    # Implementation for TCN: (Batch_Size, Timepoints per Window, 8 channels)
    neurons_inp = X_train.shape[1]  # (11044,400) --> (400)
    n_channels = 8
    n_timpoints = int(neurons_inp / n_channels)  # (400 / 8) --> (50) like the window size !

    X_train_cnn = X_train.reshape((-1, n_timpoints, n_channels))
    X_val_cnn = X_val.reshape((-1, n_timpoints, n_channels))
    X_test_cnn = X_test.reshape((-1, n_timpoints, n_channels))

    # If one hot encoding deactivated, then create dummy inputs for the second input of the model
    if not cfg.settings.advanced_pipeline:
        X_train_static = np.zeros((X_train_cnn.shape[0], 1))
        X_val_static = np.zeros((X_val_cnn.shape[0], 1))
        X_test_static = np.zeros((X_test_cnn.shape[0], 1))

    # --- Preserve existing huber/weight logic from your script: compute weights_inp ---
    if cfg.model_param.huber_weight_method == 'var':
        var_torques = np.var(Y_train, axis=0, ddof=1)
        weights_inp = 1.0 / (var_torques ** 0.5 + 1e-6)
        max_weight = np.percentile(weights_inp, 95)
        min_weight = np.percentile(weights_inp, 5)
        weights_inp = np.clip(weights_inp, min_weight, max_weight)
        weights_inp = weights_inp / np.mean(weights_inp)
    elif cfg.model_param.huber_weight_method == 'smooth_var':
        weights_inp = MLModel.getSmoothVarWeights(y_train=Y_train,
                                                  clip_percentile=[5, 75],
                                                  window_len=5,
                                                  poly_order=2)
    elif cfg.model_param.huber_weight_method == 'manual':
        weights_inp = [5, 5, 1]
    elif cfg.model_param.huber_weight_method == 'dynamic_huber':
        weights_inp = [1, 1, 1]
    else:
        raise ValueError(f"Wrong Huber weight method chosen {cfg.model_param.huber_weight_method}...")

    if cfg.model_param.load_models == False:
        # --- Build TCN model ---
        filters = 64  # 32
        stacks = 4  # 2
        dropout_rate = 0.2  # 0.1
        kernel_size = 5  # 3
        # receptive field calculation!
        time_train_start = time.perf_counter()

        input_shape_time = (n_timpoints, n_channels)
        input_shape_static = (X_train_static.shape[1],)
        tcn_model = build_model(input_shape_time,input_shape_static, filters, stacks, dropout_rate, kernel_size)
        print("MODEL shape time: ", input_shape_time)
        print("MODEL shape static: ", input_shape_static)

        if hasattr(cfg.model_param, 'loss_fcn') and 'huber' in cfg.model_param.loss_fcn.lower():
            loss_fn = tf.keras.losses.Huber()
        else:
            loss_fn = tf.keras.losses.MeanSquaredError()

        loss_weights = {'torque_elbow': weights_inp[0],
                        'torque_shoulder_front': weights_inp[1],
                        'torque_shoulder_side': weights_inp[2]}

        tcn_model.compile(
            optimizer=optimizers.Adam(learning_rate=getattr(cfg.model_param, 'learning_rate', 1e-3)),
            loss={'torque_elbow': loss_fn,
                  'torque_shoulder_front': loss_fn,
                  'torque_shoulder_side': loss_fn},
            loss_weights=loss_weights,
            metrics=['mae']
        )

        # --- Train ---
        save_model_path = getAbsolutePath("src/JTE_Project/offline/saved_offline_models")

        callbacks_list = [early_callback] if early_callback is not None else None

        history = tcn_model.fit(
            [X_train_cnn, X_train_static],
            {'torque_elbow': Y_train[:, 0],
             'torque_shoulder_front': Y_train[:, 1],
             'torque_shoulder_side': Y_train[:, 2]},
            validation_data=([X_val_cnn, X_val_static], {
                'torque_elbow': Y_val[:, 0],
                'torque_shoulder_front': Y_val[:, 1],
                'torque_shoulder_side': Y_val[:, 2]
            }),
            epochs=cfg.model_param.n_epochs,
            batch_size=cfg.model_param.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        time_train_end = time.perf_counter()
        time_train.append(time_train_end - time_train_start)

        # Save model analogous to previous saving behaviour
        if cfg.model_param.is_save_model:
            tcn_model.save(os.path.join(save_model_path, "tcn_model.keras"))

    else:  # Infer
        from tensorflow.keras.models import load_model

        save_model_path = getAbsolutePath("src/JTE_Project/offline/saved_offline_models")

        tcn_model = load_model(os.path.join(save_model_path, "tcn_model.keras"), compile=False)

    # --- Predict auf Testdaten ---
    time_prediction_start = time.perf_counter()

    preds = tcn_model.predict([X_test_cnn, X_test_static])  # preds ist [elbow, front, side], je shape (N_test,1)
    perf_results_TCN_scaled = np.concatenate([preds[0], preds[1], preds[2]], axis=1)  # (N_test, 3)

    time_prediction_end = time.perf_counter()
    time_prediction.append(time_prediction_end - time_prediction_start)

    # -- Test der Trainingsdaten auf das Modell ---
    preds_training = tcn_model.predict([X_train_cnn, X_train_static])
    preds_train_TCN = np.concatenate([preds_training[0], preds_training[1], preds_training[2]], axis=1)

if cfg.model_param.load_models and cfg.settings.yscaler:
    # -- Load old Y-Scaler File if "Loading Mode"
    y_scaler_loaded = joblib.load(getAbsolutePath("src/JTE_Project/offline/saved_offline_models/Y_scaler.pkl"))
    # Overwrite the calculated Y_scaler (because it is always calculated)
    Y_scaler = y_scaler_loaded

# --- Inverse-scaling (wie ursprünglich mit Y_scaler_dict / Y_scaler_info)
Y_ref = []
perf_results_TCN = []

for wgt, mov, start_idx, end_idx in Y_scaler_info:
    Y_ref_scaled = Y_test[start_idx:end_idx]
    Y_pred_scaled = perf_results_TCN_scaled[start_idx:end_idx]

    scaler = Y_scaler_dict[wgt][mov]
    if cfg.settings.yscaler:
        Y_ref.append(scaler.inverse_transform(Y_ref_scaled))
        perf_results_TCN.append(scaler.inverse_transform(Y_pred_scaled))
    else:
        # if y_scaler is not used, use below!
        Y_ref.append(Y_ref_scaled)
        perf_results_TCN.append(Y_pred_scaled)

Y_ref = np.concatenate(Y_ref, axis=0)
perf_results_TCN = np.concatenate(perf_results_TCN, axis=0)

print(Y_ref.shape)
print(perf_results_TCN.shape)

save_dir = getAbsolutePath(save_dir)
fullpath = save_dir / f"ref_data.npy"
fullpath.parent.mkdir(parents=True, exist_ok=True)
np.save(fullpath, Y_ref)

print("##########################################################################################")
# --- Trainings R2 Werte
r2_elbow_t, _, _ = MLModel.calculateEvalMetrics(Y_train[:, 0], preds_train_TCN[:, 0], is_Pearson=True)
r2_front_t, _, _ = MLModel.calculateEvalMetrics(Y_train[:, 1], preds_train_TCN[:, 1], is_Pearson=True)
r2_side_T, _, _ = MLModel.calculateEvalMetrics(Y_train[:, 2], preds_train_TCN[:, 2], is_Pearson=True)

print("Training Eval Metrics (TCN)!!")
print(f"Elbow R2 Train: {r2_elbow_t}")
print(f"Front R2 Train: {r2_front_t}")
print(f"Side R2 Train: {r2_side_T}\n")

# ! ************************************************
# ! Pre-filtering Eval Metrics
# ! ************************************************
print("##########################################################################################")
print("Pre-filtering Eval Metrics (TCN)!!")
r2_elbow, rmse_elbow, rho_elbow = MLModel.calculateEvalMetrics(Y_ref[:, 0], perf_results_TCN[:, 0], is_Pearson=True)
r2_front, rmse_front, rho_front = MLModel.calculateEvalMetrics(Y_ref[:, 1], perf_results_TCN[:, 1], is_Pearson=True)
r2_side, rmse_side, rho_side = MLModel.calculateEvalMetrics(Y_ref[:, 2], perf_results_TCN[:, 2], is_Pearson=True)

r2_e_arr.append(r2_elbow)
rho_e_arr.append(rho_elbow)

r2_sf_arr.append(r2_front)
rho_sf_arr.append(rho_front)

r2_ss_arr.append(r2_side)
rho_ss_arr.append(rho_side)
'''
print("The results across seed are...\n")
print(f"Elbow R2: {r2_e_arr}")
print(f"Front R2: {r2_sf_arr}")
print(f"Side R2: {r2_ss_arr}\n")
'''
print(f"Elbow R2 stats: Mean: {np.mean(r2_e_arr)}  Std. : {np.std(r2_e_arr)}")
print(f"Front R2 stats: Mean: {np.mean(r2_sf_arr)}  Std. : {np.std(r2_sf_arr)}")
print(f"Side R2 stats: Mean: {np.mean(r2_ss_arr)}  Std. : {np.std(r2_ss_arr)}\n")
'''
print(f"Elbow Pearson stats: Mean: {np.mean(rho_e_arr)}  Std. : {np.std(rho_e_arr)}")
print(f"Front Pearson stats: Mean: {np.mean(rho_sf_arr)}  Std. : {np.std(rho_sf_arr)}")
print(f"Side Pearson stats: Mean: {np.mean(rho_ss_arr)}  Std. : {np.std(rho_ss_arr)}")
'''
# ! ************************************************
# ! Post-filtering Eval Metrics
# ! ************************************************

# Median filter
if cfg.post_train_param.filter_type == 'median':
    for i in range(3):
        perf_results_TCN[:, i] = medfilt(perf_results_TCN[:, i], kernel_size=cfg.post_train_param.filter_size)

# Savitzky-Golay
if getattr(cfg.post_train_param, 'savgol_window_len', None) is not None:
    for i in range(3):
        perf_results_TCN[:, i] = savgol_filter(perf_results_TCN[:, i],
                                               cfg.post_train_param.savgol_window_len,
                                               cfg.post_train_param.savgol_poly_order)
np.save(f"{save_dir}/pred_results_seed{seed}.npy", perf_results_TCN)

print("##########################################################################################")
print("ALL RESULTS WITH POST-FILTERING")
# Post-filter Eval
print("Post-filtering Eval Metrics (TCN)!!")
r2_elbow_pf, rmse_elbow_pf, rho_elbow_pf = MLModel.calculateEvalMetrics(Y_ref[:, 0], perf_results_TCN[:, 0],
                                                                        is_Pearson=True)
r2_front_pf, rmse_front_pf, rho_front_pf = MLModel.calculateEvalMetrics(Y_ref[:, 1], perf_results_TCN[:, 1],
                                                                        is_Pearson=True)
r2_side_pf, rmse_side_pf, rho_side_pf = MLModel.calculateEvalMetrics(Y_ref[:, 2], perf_results_TCN[:, 2],
                                                                     is_Pearson=True)
r2_e_arr_pf.append(r2_elbow_pf)
rho_e_arr_pf.append(rho_elbow_pf)

r2_sf_arr_pf.append(r2_front_pf)
rho_sf_arr_pf.append(rho_front_pf)

r2_ss_arr_pf.append(r2_side_pf)
rho_ss_arr_pf.append(rho_side_pf)
'''
print("The results across seed are...\n")
print(f"Elbow R2: {r2_e_arr_pf}")
print(f"Front R2: {r2_sf_arr_pf}")
print(f"Side R2: {r2_ss_arr_pf}\n")
'''
print(f"Elbow R2 stats: Mean: {np.mean(r2_e_arr_pf)}  Std. : {np.std(r2_e_arr_pf)}")
print(f"Front R2 stats: Mean: {np.mean(r2_sf_arr_pf)}  Std. : {np.std(r2_sf_arr_pf)}")
print(f"Side R2 stats: Mean: {np.mean(r2_ss_arr_pf)}  Std. : {np.std(r2_ss_arr_pf)}\n")
'''
print(f"Elbow Pearson stats: Mean: {np.mean(rho_e_arr)}  Std. : {np.std(rho_e_arr)}")
print(f"Front Pearson stats: Mean: {np.mean(rho_sf_arr)}  Std. : {np.std(rho_sf_arr)}")
print(f"Side Pearson stats: Mean: {np.mean(rho_ss_arr)}  Std. : {np.std(rho_ss_arr)}")
'''

# Ausgabe der Datenshapes
print("TRAIN DATA Shape: ", X_train_cnn.shape)
print("VAL DATA Shape: ", X_val_cnn.shape)
print("TEST DATA Shape: ", X_test_cnn.shape)

'''
# Printing Model Parameters (Size on GPU etc.)
if tf.config.list_physical_devices('GPU'):
  # Returns a dict in the form {'current': <current mem usage>,
  #                             'peak': <peak mem usage>}
  gpu_mem = tf.config.experimental.get_memory_info('GPU:0')

input_shape = (1,) + X_train.shape[1:]

concrete_func = tf.function(tcn_model).get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(concrete_func)
graph_def = frozen_func.graph.as_graph_def()

opts = ProfileOptionBuilder.float_operation()
flops = profile(frozen_func.graph, options=opts)

print('=' * 50)
print(f"Model summary: ")
tcn_model.summary()
print(f"\nTotal FLOPs: {flops.total_float_ops:,}")
print('=' * 50)
print(f"Peak GPU Memory usage:  {gpu_mem['peak']/1e6 :.2f} MB")

print('=' * 50)
'''

# Timings
print(f"Preprocessing Zeit: {time_preproc :.4f} Sekunden")

print(f"Scaling Zeit: {time_scaling :.4f} Sekunden")

print(f"Feature Extraction Zeit: {time_feat :.4f} Sekunden")

if cfg.model_param.load_models == False:
    time_train_mean = np.mean(time_train)
    time_train_std = np.std(time_train)
    print(f"Model Training Zeit: {time_train_mean :.4f} ± {time_train_std :.4f} Sekunden")

time_prediction_mean = np.mean(time_prediction)
time_prediction_std = np.std(time_prediction)
print(f"Model Prediction Zeit: {time_prediction_mean :.4f} ± {time_prediction_std :.4f} Sekunden")


#Data Visualization
def calculate_metrics(y_true, y_pred):
    """Berechnet RMSE, R² und Pearson-Korrelationskoeffizient."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return rmse, r2, pearson_corr

def plotResults(y_true, label_true, y_pred, label_pred, title,
                ylabel="Torque", is_grid_on=True):
    rmse, r2, pearson_corr = calculate_metrics(y_true, y_pred)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label=label_true, color="orange")
    plt.plot(y_pred, label=label_pred, color="blue", linestyle="-")
    plt.title(f"{title}\nRMSE={rmse:.3f}, R²={r2:.3f}, Pearson={pearson_corr:.3f}")
    plt.xlabel("Samples")
    plt.ylabel(ylabel)
    if is_grid_on:
        plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Anwendung auf deine Daten:
plotResults(Y_ref[:, 0], "Real", perf_results_TCN[:, 0], "Prediction",
            title="Elbow", ylabel="Torque in N-m")

plotResults(Y_ref[:, 1], "Real", perf_results_TCN[:, 1], "Prediction",
            title="Shoulder Front", ylabel="Torque in N-m")

plotResults(Y_ref[:, 2], "Real", perf_results_TCN[:, 2], "Prediction",
            title="Shoulder Side", ylabel="Torque in N-m")

if not cfg.model_param.load_models and cfg.settings.yscaler:
    # For saving the Y-Scaler if needed
    try:
        scaler_save_path = save_model_path / "Y_scaler.pkl"

        # Wandelt das äußere UND alle inneren defaultdicts in normale dicts um
        scaler_to_save = {wgt: dict(mov_dict) for wgt, mov_dict in Y_scaler_dict.items()}
        joblib.dump(scaler_to_save, scaler_save_path)
        print(f"Scaler-Wörterbuch erfolgreich gespeichert unter: {scaler_save_path}")


    except Exception as e:
        print(f"Fehler beim Speichern des Scaler-Wörterbuchs: {e}")
