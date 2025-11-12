
#* This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import tensorflow as tf
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

#own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, createOutputDir, createReadme, plotResults

import warnings
warnings.formatwarning = customWarningFormat

# Für Keras / Model-Train
from tensorflow.keras import Input, layers, models, optimizers
# Für Post-Filter (nutze SciPy falls vorhanden)
from scipy.signal import medfilt, savgol_filter


#! ************************************************
#! User Parameters and Data Collection
#! ************************************************

def build_model(input_shape_time, filters, stacks, dropout_rate, kernel_size):

    inputs = []
    branches = []

    # Zeit-Pfad (TCN)
    inp_time = Input(shape=input_shape_time, name='emg_input')
    x = inp_time

    for s in range(stacks):
        d = 2 ** s  # Dilatation: 1,2,4,8,...
        y = layers.Conv1D(filters,
                          kernel_size,
                          padding='causal',
                          dilation_rate=d,
                          kernel_initializer='he_normal')(x)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)

        y = layers.Conv1D(filters,
                          kernel_size,
                          padding='causal',
                          dilation_rate=d,
                          kernel_initializer='he_normal')(y)
        y = layers.ReLU()(y)
        y = layers.LayerNormalization()(y)

        # Residual-Shortcut ggf. an Kanäle anpassen
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same',
                              kernel_initializer='he_normal')(x)

        x = layers.add([x, y])

    # Seq-to-one Readout
    x = layers.GlobalAveragePooling1D()(x)
    inputs.append(inp_time)
    branches.append(x)

    combined = branches[0]
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dropout(dropout_rate)(combined)

    out_e = layers.Dense(1, name='torque_elbow')(combined)
    out_f = layers.Dense(1, name='torque_shoulder_front')(combined)
    out_s = layers.Dense(1, name='torque_shoulder_side')(combined)
    return models.Model(inputs, [out_e, out_f, out_s], name="MTL_TCN")

time_preproc = 0
time_feat = 0

#? load config file
config_filename = 'pipeline_jte_bu62d.yaml'
cfg = loadConfig(filename=config_filename)
save_dir = cfg.filepath.save_predictions_path
print("Using the following CONFIG FILE: ", config_filename)

#? init early stopping
if cfg.model_param.is_early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=cfg.model_param.monitor,
                                                      min_delta=cfg.model_param.min_delta,
                                                      patience=cfg.model_param.patience,
                                                      verbose=cfg.model_param.verbose,
                                                      baseline=cfg.model_param.baseline,
                                                      restore_best_weights=cfg.model_param.restore_best_weights)
else:
    early_callback = None


#? init performance results list
perf_results_total_MLP = []

#? Initialise arrays for appending all filenames
emg_filenames = []
quali_e_filenames = []
quali_sf_filenames = []
quali_ss_filenames = []

#! ************************************************
#! Get training filenames
#! ************************************************
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
            quali_e_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_e_file_pattern))
            if quali_e_files:
                quali_e_table[w_idx][m_idx].append(quali_e_files[0])

            # Quali SF
            quali_sf_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_sf_file_pattern))
            if quali_sf_files:
                quali_sf_table[w_idx][m_idx].append(quali_sf_files[0])

            # Quali SS
            quali_ss_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_ss_file_pattern))
            if quali_ss_files:
                quali_ss_table[w_idx][m_idx].append(quali_ss_files[0])

#! ************************************************
#! Load training, testing data
#! ************************************************
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

channel_cum_mvc = []

for wgt_idx, wgt in enumerate(weights):
    for mov_idx, mov in enumerate(mov_types):
        time_preproc_start = time.perf_counter()

        #? Loading and epoching for training   
        EMG_Data = EMGData(format="ANTmini", filenames=emg_table[wgt_idx][mov_idx], data_path=cfg.filepath.data_path, f_samp=cfg.preprocess_param.f_samp, channel_names=cfg.preprocess_param.channel_names_emg)

        # print(EMG_Data.events)
        #? Plotting the raw EMG data
        if cfg.plot_param.is_plot_raw:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="uV", 
                                title="Raw EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in uV", 
                                is_grid_on=True)

        # #? Loading the target values for the 3 joints
        print("Creating Quali Elbow object...")
        Quali_Data_Elbow = EEGData(format="NumpyQualisys", 
                                    filenames=quali_e_table[wgt_idx][mov_idx],
                                    data_path=cfg.filepath.data_path, 
                                    f_samp=cfg.preprocess_param.f_samp, 
                                    channel_names=cfg.preprocess_param.channel_names_quali, 
                                    add_marker_channel=True)

        print("Creating Quali Shoulder Front object...")
        Quali_Data_Front = EEGData(format="NumpyQualisys", 
                                    filenames=quali_sf_table[wgt_idx][mov_idx],
                                    data_path=cfg.filepath.data_path, 
                                    f_samp=cfg.preprocess_param.f_samp, 
                                    channel_names=cfg.preprocess_param.channel_names_quali, 
                                    add_marker_channel=True)

        print("Creating Quali Shoulder Side object...")
        Quali_Data_Side = EEGData(format="NumpyQualisys", 
                                    filenames=quali_ss_table[wgt_idx][mov_idx], 
                                    data_path=cfg.filepath.data_path, 
                                    f_samp=cfg.preprocess_param.f_samp, 
                                    channel_names=cfg.preprocess_param.channel_names_quali, 
                                    add_marker_channel=True)


        channel_names = EMG_Data.getChannelNames()
        print("Channel Names: ", channel_names)
        print("Channel Length: ", len(channel_names))
        print("")

        # print(Quali_Data_Elbow.data.shape)
        if cfg.plot_param.is_plot_quali:
            plt.figure()
            plt.plot(np.arange(0,Quali_Data_Elbow.data[0,:].shape[0], 1)/Quali_Data_Elbow.f_samp,Quali_Data_Elbow.data[0,:])
            plt.title("Elbow Torque plot for right arm")
            plt.xlabel("Time in s")
            plt.ylabel("Torque in N-m")
            plt.grid()
            plt.show()

        #! ************************************************
        #! Data Pre-processing
        #! ************************************************

        #? Band pass filter
        #* design the bandpass filter
        sos_hp = EMG_Data.designFilter(f_high=cfg.preprocess_param.f_cutoff_hpf,
                                        f_low=cfg.preprocess_param.f_cutoff_lpf,
                                        order=cfg.preprocess_param.filter_order,
                                        filter_type="scipy_butter",
                                        return_type="sos")
        #* apply bandpass filter 
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method, 
                                    sos=sos_hp)
        #? Plotting bandpass filtered data
        if cfg.plot_param.is_plot_hpf:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="uV", 
                                title="Band-Pass Filtered plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in uV", 
                                is_grid_on=True)

        EMG_Data_freq = deepcopy(EMG_Data)
            
        #? Apply Variance Filter from variance_tools_api
        print("Applying Variance filter...")
        width           = cfg.preprocess_param.var_filter_width
        ring_buffer     = np.zeros(width)
        index           = 0
        EMG_Data.applyVarianceFilter_data(ring_buffer=ring_buffer,
                                        width=width,
                                        index=index)
        print("Variance Filter applied!!\n")

        if cfg.plot_param.is_plot_var_filter:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="uV", 
                                title="Variance Filtered EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in uV", 
                                is_grid_on=True)

        #? Input Normalisation
        print("Calculating the channel-wise MVC for EMG...")
        channelwise_mvc = np.max(np.abs(EMG_Data.data), axis=1).reshape(-1,1)
        # For storing the numpy file (online case)
        c_mvc_cum = np.max(np.abs(EMG_Data.data), axis=1)
        channel_cum_mvc.append(c_mvc_cum)

        print("Performing Input Normalization with Max Voluntary Contraction ...")
        if cfg.preprocess_param.normalisation_method == 'overall_mvc':
            EMG_Data.normalizeContinuousData(mvc=np.max(channelwise_mvc))
        elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
            EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)
        else:
            warnings.warn("This method is not yet implemented!! Omitting!")
        print("Input Normalization with Max Voluntary Contraction performed!!\n")

        #? Low pass filter to smoothen the EMG signal
        #* design the lowpass filter
        sos_lp = EMG_Data.designFilter(f_low=cfg.preprocess_param.f_cutoff_sm_lpf,
                                        order=cfg.preprocess_param.sm_filter_order,
                                        filter_type="scipy_butter",
                                        return_type="sos")
        #* apply lpf filter 
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method,
                                    sos=sos_lp)

        #? Plot normalised and smoothened data
        if cfg.plot_param.is_plot_smoothed:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="V", 
                                title="Normalised and Smoothed EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in V", 
                                is_grid_on=True)

        #? Low pass filter to smoothen the torques
        Quali_Data_Elbow.filterData_offline(filter_method=cfg.preprocess_param.filter_method, 
                                        sos=sos_lp)
        Quali_Data_Front.filterData_offline(filter_method=cfg.preprocess_param.filter_method, 
                                    sos=sos_lp)
        Quali_Data_Side.filterData_offline(filter_method=cfg.preprocess_param.filter_method, 
                                    sos=sos_lp)

        #? Plot normalised and smoothened data
        if cfg.plot_param.is_plot_smoothed:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="V", 
                                title="Normalised and Smoothed EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in V", 
                                is_grid_on=True)
            Quali_Data_Side.plotEMG(data=Quali_Data_Side.data[0,:], 
                                unit="V", 
                                title="Normalised and Smoothed Elbow Torques plot", 
                                xlabel="Time in s", 
                                ylabel="Torque in N-m", 
                                is_grid_on=True)

        #? Calculate Neural Activation Force
        if cfg.preprocess_param.use_activation_fncn:
            print("Replacing sample with its force activation value...")
            EMG_Data.calculateActivationForceFunction(d=cfg.preprocess_param.act_delay, 
                                                        b1=cfg.preprocess_param.act_beta1, 
                                                        b2=cfg.preprocess_param.act_beta2, 
                                                        g=cfg.preprocess_param.act_gamma, 
                                                        nonlinear_shape_factor=cfg.preprocess_param.act_A)
            print("Replaced each sample with its force activation value!!\n")

        #? Plot force activation data
        if cfg.plot_param.is_plot_act:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="V", 
                                title="Force Activated EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in V", 
                                is_grid_on=True)
            
        time_preproc_end = time.perf_counter()
        time_preproc += (time_preproc_end - time_preproc_start)
        time_feat_start = time.perf_counter()

        #? Windowing the data
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
                                                    stopmarkernumber=2, window_size=cfg.preprocess_param.window_size_y, 
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
        
        #? Ensure the number of windows of each file are the same for inp and target
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

        #? One Hot Encoding for 3 weights and 2 movements
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

        #? Merge weights_code and mov_code to form categorical feat set
        x_cat = np.column_stack([weights_code, mov_code])

        #? Plot specific filtered windows for debugging
        if cfg.plot_param.is_plot_filt_win:
            EMG_Data.plotEMG(data=EMG_Data.getWindows()[0,4,:,18], #[trl,chn,smpl,wnd]
                                n_samples= EMG_Data.getWindows().shape[2],
                                unit="V", 
                                title="Pre-processed EMG plot for Channel 5 Window 18", 
                                xlabel="Time in s", 
                                ylabel="Voltage in V", 
                                is_grid_on=True)
            
        #! ************************************************
        #! Feature Extraction
        #! ************************************************
        print("Extracting features from windowed data...")

        #? EMG signal timepoints feature extraction
        window_size_ms = cfg.preprocess_param.window_size_x * 1000 / EMG_Data.f_samp
        feature_indices_windows_x = np.array([0, window_size_ms])
        EMG_Data.featureExtractionFromWindows(feature_type="timepoints", 
                                            feature_indices_windows=feature_indices_windows_x)
        EMG_Data.printFeatureShape()

        if cfg.settings.feature_extraction:
            #? time domain feature extraction
            ## EMG Feature Extraction
            rms_feature = EMG_Data.getRMSFeatures_windows(n_channels=len(channel_names)) # RMS value
            EMG_Data.addFeatures(rms_feature)
            # print(EMG_Data.getFeatures()[1,:])
            # EMG_Data.printFeatureShape()

            wfl_feature = EMG_Data.getWaveformLengthFeatures_windows(n_channels=len(channel_names))  # Waveform length
            EMG_Data.addFeatures(wfl_feature)
            # print(EMG_Data.getFeatures()[1,:])
            # EMG_Data.printFeatureShape()

            ssc_feature = EMG_Data.getSlopeSignChangeFeatures_windows(n_channels=len(channel_names),
                                                                    threshold=0.02)    # Slope Sign Change
            EMG_Data.addFeatures(ssc_feature)
            # print(EMG_Data.getFeatures()[1,:])
            # EMG_Data.printFeatureShape()

            #? freq domain feature extraction
            EMG_Data_freq.featureExtractionFromWindows(feature_type="freqBandPower",
                                                    psd_method="multitaper",
                                                    freq_bands=[15, 50, 100, 150, 200, 245])
            fbp_feature = EMG_Data_freq.getFeatures()
            EMG_Data.addFeatures(fbp_feature)
            # EMG_Data.printFeatureShape()

            #? time-freq domain feature extraction
            freqs = np.arange(start=50, stop=226, step=25)
            n_cycles = np.ones(len(freqs)) * 5
            n_cycles[0] = 3
            n_cycles[1] = 4
            mwc_feature = EMG_Data_freq.getMorletWaveletCoeffFeatures_windows(freqs=freqs,
                                                                            n_cycles=n_cycles)    # Morlet transform
            EMG_Data.addFeatures(mwc_feature)
            # print(f"Total EMG features extracted: {EMG_Data.getFeatures().shape}")

            #? Change between consecutive samples (window i and wind i+1)
            peak_detection = np.diff(EMG_Data.getFeatures(), axis=0, prepend=EMG_Data.getFeatures()[0:1,:])
            EMG_Data.addFeatures(peak_detection)
            print(f"Total EMG features extracted: {EMG_Data.getFeatures().shape}")

       
        #? Output feature extraction
        window_size_ms = cfg.preprocess_param.window_size_y * 1000 / Quali_Data_Elbow.f_samp
        if cfg.preprocess_param.target_feature_select == 'mean':
            feature_indices_windows_y = np.array([0, window_size_ms])
            use_mean_bool = True
        elif cfg.preprocess_param.target_feature_select == 'mid':
            feature_indices_windows_y = np.array([(window_size_ms/2)-2, window_size_ms/2])
            use_mean_bool = False
        elif cfg.preprocess_param.target_feature_select == 'end':
            feature_indices_windows_y = np.array([window_size_ms-2, window_size_ms])
            use_mean_bool = False
        else:
            raise ValueError(f"Provided target_feature_select {cfg.preprocess_param.target_feature_select} is not yet implemented... Please choose between 'mean', 'mid', and 'end'")

        Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints", 
                                                        feature_indices_windows=feature_indices_windows_y,
                                                        use_mean=use_mean_bool)
        Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints", 
                                                        feature_indices_windows=feature_indices_windows_y,
                                                        use_mean=use_mean_bool)
        Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints", 
                                                        feature_indices_windows=feature_indices_windows_y,
                                                        use_mean=use_mean_bool)
        print("Feature extraction from windowed data completed!!\n")

        #? Merge output features
        target_features = np.concatenate([Quali_Data_Elbow.getFeatures(), Quali_Data_Front.getFeatures(), Quali_Data_Side.getFeatures()], axis=1)

        input_features = EMG_Data.getFeatures()

        #? Print input feature and target feature length
        print(f"Input feature shape (pre-merge): {EMG_Data.getFeatures().shape}")
        print(f"Target feature shape (pre-merge): {target_features.shape}")

        #? Split data into train, validation, and test sets
        X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(input_features,
                                                            target_features, 
                                                            train_size=cfg.model_param.train_test_split,
                                                            shuffle=False)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp,
                                                        Y_train_temp, 
                                                        train_size= 1 - cfg.model_param.validation_split,
                                                        shuffle=False)

        if cfg.settings.advanced_pipeline:
            # ? Split categorical data into train, validation, and test sets
            X_train_cat_temp, X_test_cat = train_test_split(x_cat,
                                                            train_size=cfg.model_param.train_test_split,
                                                            shuffle=False)

            X_train_cat, X_val_cat, = train_test_split(X_train_cat_temp,
                                                       train_size=1 - cfg.model_param.validation_split,
                                                       shuffle=False)

            print("Split data into train, test, and val!!")

            # ? One Hot encoding
            encoder = OneHotEncoder(sparse_output=False)
            X_train_cat = encoder.fit_transform(X_train_cat)
            X_test_cat = encoder.transform(X_test_cat)
            X_val_cat = encoder.transform(X_val_cat)

            print("Y TRAIN SHAPE: ", Y_train.shape)
            print("Y_val ", Y_val.shape)
            print("Y_test ", Y_test.shape)

            # ? Creating history of features
            history_len = 3
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

            print(f"{history_len} feature vectors stacked together!!")
            print(f"Stacked x_train feature shape: {X_train.shape}")
            print(f"Stacked y_train feature shape: {Y_train.shape}")


        #? Scale output features -> [-1,1] for tanh

        Y_scaler, Y_train, Y_test, Y_val = EMG_Data.scaleFeatures_windows(train_data=Y_train, 
                                                                        test_data=Y_test, 
                                                                        val_data=Y_val, 
                                                                        method="MinMaxScaler",
                                                                        feature_range=(-1,1),
                                                                          scaler_file=None)
        '''
        # Feasibility-Test (Applying Stored Y_Scaler)
        stored_standard_scaler = Test_Y_Scaler[wgt][mov]
        _, Y_train, Y_test, Y_val = EMG_Data.scaleFeatures_windows(train_data=Y_train,
                                                                        test_data=Y_test,
                                                                        val_data=Y_val,
                                                                        method="MinMaxScaler",
                                                                        feature_range=(-1,1),
                                                                   scaler_file = stored_standard_scaler)
        '''
        start_idx = current_idx
        end_idx = current_idx + Y_test.shape[0]

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

        time_feat_end = time.perf_counter()
        time_feat += (time_feat_end - time_feat_start)

## Storing channelwise mvc (now the mean is used) ONLINE only
channel_cum = np.array(channel_cum_mvc)
channel_max = np.mean(channel_cum,axis=0)
#np.save(getAbsolutePath("src/JTE_Project/online/resources/mvc/channelwise_mvc.npy"), channel_max)


X_train = np.concatenate(X_train_combined, axis=0)
Y_train = np.concatenate(Y_train_combined, axis=0)

X_test = np.concatenate(X_test_combined, axis=0)
Y_test = np.concatenate(Y_test_combined, axis=0)

X_val = np.concatenate(X_val_combined, axis=0)
Y_val = np.concatenate(Y_val_combined, axis=0)

# Calculate Scaling Time
time_scaling_start = time.perf_counter()

if cfg.settings.scaling:
    #? Pre-PCA scaling of input features
    pre_emg_scaler, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train,
                                                            test_data=X_test,
                                                            val_data=X_val,
                                                            method="StandardScaler")

    import joblib
    #joblib.dump(pre_emg_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/pre_emg_scaler.pkl"))

    #? Dimensionality Reduction - PCA
    pca_scaler, X_train, X_test, X_val = EMG_Data.reduceDimensions_windows(train_data=X_train,
                                    test_data = X_test,
                                    val_data = X_val,
                                    method="PCA",
                                    n_components=0.99,
                                    mode="offline")


    #joblib.dump(pca_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/pca_scaler.pkl"))


    #? Scale the input features -> StandardScaler
    post_emg_scaler, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train,
                                                            test_data=X_test,
                                                            val_data=X_val,
                                                            method="StandardScaler")

    #joblib.dump(post_emg_scaler, getAbsolutePath("src/JTE_Project/offline/saved_online_models/post_emg_scaler.pkl"))


time_scaling_end = time.perf_counter()
time_scaling = time_scaling_end - time_scaling_start

if cfg.settings.advanced_pipeline:
    #? One-hot encoding
    wgt_train, mov_train = EMG_Data.convertMetaToArray(meta_list_train)
    wgt_val, mov_val     = EMG_Data.convertMetaToArray(meta_list_val)
    wgt_test, mov_test   = EMG_Data.convertMetaToArray(meta_list_test)

    encoder_wgt = OneHotEncoder(sparse_output=False)
    wgt_train_onehot = encoder_wgt.fit_transform(wgt_train)
    wgt_test_onehot  = encoder_wgt.transform(wgt_test)
    wgt_val_onehot   = encoder_wgt.transform(wgt_val)

    encoder_mov = OneHotEncoder(sparse_output=False)
    mov_train_onehot = encoder_mov.fit_transform(mov_train)
    mov_test_onehot  = encoder_mov.transform(mov_test)
    mov_val_onehot   = encoder_mov.transform(mov_val)

    # #? Add categorical features to the input sets
    #X_train = np.concatenate([X_train, wgt_train_onehot, mov_train_onehot], axis=1)
    #X_test = np.concatenate([X_test, wgt_test_onehot, mov_test_onehot], axis=1)
    #X_val = np.concatenate([X_val, wgt_val_onehot, mov_val_onehot], axis=1)

#? Shuffle training sets
perm = np.random.permutation(X_train.shape[0])
X_train[:] = X_train[perm]
Y_train[:] = Y_train[perm]
#! ************************************************
#! Train, Load, or Test Model
#! ************************************************

# --- set global seed ---
seed_arr = [1]
r2_e_arr = []
r2_sf_arr = []
r2_ss_arr = []

rho_e_arr = []
rho_sf_arr = []
rho_ss_arr = []
 
perf_res_e_arr = []
perf_res_sf_arr = []
perf_res_ss_arr = []

time_train = []
time_prediction = []

for seed in seed_arr:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # ---------------------------
    # Replace MLP training with TCN training (minimal changes)
    # ---------------------------

    neurons_inp = X_train.shape[1]

    # reshape existing 2D feature vectors -> 3D for Conv1D:
    # time dimension = neurons_inp, channels = 1 (minimal change so rest of pipeline unchanged)
    X_train_cnn = X_train.reshape((-1, neurons_inp, 1))
    X_val_cnn = X_val.reshape((-1, neurons_inp, 1))
    X_test_cnn = X_test.reshape((-1, neurons_inp, 1))

    # --- Preserve existing huber/weight logic from your script: compute weights_inp ---
    if cfg.model_param.huber_weight_method == 'var':
        var_torques = np.var(Y_train, axis=0, ddof=1)
        weights_inp = 1.0 / (var_torques ** 0.5 + 1e-6)
        max_weight = np.percentile(weights_inp, 95)
        min_weight = np.percentile(weights_inp, 5)
        weights_inp = np.clip(weights_inp, min_weight, max_weight)
        weights_inp = weights_inp / np.mean(weights_inp)
    elif cfg.model_param.huber_weight_method == 'smooth_var':
        # the original uses MLP_model.getSmoothVarWeights; replicate previous behaviour if needed
        # fallback: call the helper from MLModel (static method) if available:
        # Note: original code used MLP_model.getSmoothVarWeights -> here we call MLModel.getSmoothVarWeights
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
        filters = 32
        stacks = 2
        dropout_rate = 0.10
        kernel_size = 3

        time_train_start = time.perf_counter()

        input_shape_time = (neurons_inp, 1)
        tcn_model = build_model(input_shape_time, filters, stacks, dropout_rate, kernel_size)

        # choose loss function similar to previous pipeline (if cfg.model_param.loss_fcn contains 'huber' use Huber)
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
            X_train_cnn,
            {'torque_elbow': Y_train[:, 0],
             'torque_shoulder_front': Y_train[:, 1],
             'torque_shoulder_side': Y_train[:, 2]},
            validation_data=(X_val_cnn, {
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

    else: # Infer
        from tensorflow.keras.models import load_model

        save_model_path = getAbsolutePath("src/JTE_Project/offline/saved_offline_models")

        tcn_model = load_model(os.path.join(save_model_path, "tcn_model.keras"), compile=False)


    # --- Ausgabe der Testdaten Shape ---
    print("TEST DATA Shape: ", X_test_cnn.shape)

    # --- Predict auf Testdaten ---
    time_prediction_start = time.perf_counter()

    preds = tcn_model.predict(X_test_cnn)  # preds ist [elbow, front, side], je shape (N_test,1)
    perf_results_TCN_scaled = np.concatenate([preds[0], preds[1], preds[2]], axis=1)  # (N_test, 3)

    time_prediction_end = time.perf_counter()
    time_prediction.append(time_prediction_end - time_prediction_start)


# --- Inverse-scaling (wie ursprünglich mit Y_scaler_dict / Y_scaler_info)
Y_ref = []
perf_results_TCN = []

for wgt, mov, start_idx, end_idx in Y_scaler_info:
    Y_ref_scaled = Y_test[start_idx:end_idx]
    Y_pred_scaled = perf_results_TCN_scaled[start_idx:end_idx]

    scaler = Y_scaler_dict[wgt][mov]
    Y_ref.append(scaler.inverse_transform(Y_ref_scaled))
    perf_results_TCN.append(scaler.inverse_transform(Y_pred_scaled))
    # if y_scaler is not used, use below!
    #Y_ref.append(Y_ref_scaled)
    #perf_results_TCN.append(Y_pred_scaled)

Y_ref = np.concatenate(Y_ref, axis=0)
perf_results_TCN = np.concatenate(perf_results_TCN, axis=0)

print(Y_ref.shape)
print(perf_results_TCN.shape)


save_dir = getAbsolutePath(save_dir)
fullpath = save_dir / f"ref_data.npy"
fullpath.parent.mkdir(parents=True, exist_ok=True)
np.save(fullpath, Y_ref)

# --- Eval Metrics (wie früher)
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

# ! ************************************************
# ! Post-prediction Filtering (manuell, da MLP_model.applyFilter_prediction entfällt)
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

# Post-filter Eval
print("Post-filtering Eval Metrics (TCN)!!")
r2_elbow_pf, rmse_elbow_pf, rho_elbow_pf = MLModel.calculateEvalMetrics(Y_ref[:, 0], perf_results_TCN[:, 0],
                                                                        is_Pearson=True)
r2_front_pf, rmse_front_pf, rho_front_pf = MLModel.calculateEvalMetrics(Y_ref[:, 1], perf_results_TCN[:, 1],
                                                                        is_Pearson=True)
r2_side_pf, rmse_side_pf, rho_side_pf = MLModel.calculateEvalMetrics(Y_ref[:, 2], perf_results_TCN[:, 2],
                                                                     is_Pearson=True)

r2_e_arr.append(r2_elbow)
rho_e_arr.append(rho_elbow)
perf_res_e_arr.append(perf_results_TCN[:, 0])

r2_sf_arr.append(r2_front)
rho_sf_arr.append(rho_front)
perf_res_sf_arr.append(perf_results_TCN[:, 1])

r2_ss_arr.append(r2_side)
rho_ss_arr.append(rho_side)
perf_res_ss_arr.append(perf_results_TCN[:, 2])


print("The results across seed are...\n")
print(f"Elbow R2: {r2_e_arr}")
print(f"Front R2: {r2_sf_arr}")
print(f"Side R2: {r2_ss_arr}\n")

print(f"Elbow R2 stats: Mean: {np.mean(r2_e_arr)}  Std. : {np.std(r2_e_arr)}")
print(f"Front R2 stats: Mean: {np.mean(r2_sf_arr)}  Std. : {np.std(r2_sf_arr)}")
print(f"Side R2 stats: Mean: {np.mean(r2_ss_arr)}  Std. : {np.std(r2_ss_arr)}\n")

print(f"Elbow Pearson stats: Mean: {np.mean(rho_e_arr)}  Std. : {np.std(rho_e_arr)}")
print(f"Front Pearson stats: Mean: {np.mean(rho_sf_arr)}  Std. : {np.std(rho_sf_arr)}")
print(f"Side Pearson stats: Mean: {np.mean(rho_ss_arr)}  Std. : {np.std(rho_ss_arr)}")

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


'''
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
'''

try:
    scaler_save_path = save_model_path / "Y_scaler.pkl"

    # Wandelt das äußere UND alle inneren defaultdicts in normale dicts um
    scaler_to_save = {wgt: dict(mov_dict) for wgt, mov_dict in Y_scaler_dict.items()}
    joblib.dump(scaler_to_save, scaler_save_path)
    print(f"Scaler-Wörterbuch erfolgreich gespeichert unter: {scaler_save_path}")


except Exception as e:
    print(f"Fehler beim Speichern des Scaler-Wörterbuchs: {e}")