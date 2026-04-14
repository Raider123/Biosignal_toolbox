
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
import random

#own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, createOutputDir, createReadme, plotResults

import warnings
warnings.formatwarning = customWarningFormat

#! ************************************************
#! User Parameters and Data Collection
#! ************************************************

#? load config file
config_filename = 'pipeline_jte_ww06d.yaml'
cfg = loadConfig(filename=config_filename)

#? init early stopping 
if cfg.model_param.is_early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=cfg.model_param.monitor, 
                                                      min_delta=cfg.model_param.min_delta, 
                                                      patience=cfg.model_param.patience, 
                                                      verbose=cfg.model_param.verbose, 
                                                      baseline=cfg.model_param.baseline, 
                                                      restore_best_weights=cfg.model_param.restore_best_weights, 
                                                      start_from_epoch=cfg.model_param.start_from_epoch)
else:
    early_callback = None

#? filename suffix
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
filename_suffix = timestamp

for i in range(len(cfg.data_param.mov_type)):
    filename_suffix = filename_suffix + '_' + cfg.data_param.mov_type[i]

for i in range(len(cfg.data_param.weights)):
    filename_suffix = filename_suffix + '_' + cfg.data_param.weights[i]

print(f"filename suffix is: {filename_suffix}")

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

for mov_idx, wgt_idx, set_idx in itertools.product(cfg.data_param.mov_type, cfg.data_param.weights, cfg.data_param.set_num):
    emg_file_pattern = f"{cfg.filepath.emg_file_prefix}_{wgt_idx}_{mov_idx}_{set_idx}.txt"

    quali_e_file_pattern = f"{cfg.filepath.quali_torque_prefix[0]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_idx}_{set_idx}.npy"

    quali_sf_file_pattern = f"{cfg.filepath.quali_torque_prefix[1]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_idx}_{set_idx}.npy"

    quali_ss_file_pattern = f"{cfg.filepath.quali_torque_prefix[2]}_{cfg.data_param.subject_code[0]}_{wgt_idx}_{mov_idx}_{set_idx}.npy"

    #? Check if the file exists
    emg_matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.emg_path).glob(emg_file_pattern))
    if not emg_matched_files:
        continue
    else:
        emg_filenames.append(emg_matched_files[0])
    
    quali_e_matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_e_file_pattern))
    if not quali_e_matched_files:
        continue
    else:
        quali_e_filenames.append(quali_e_matched_files[0])
    
    quali_sf_matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_sf_file_pattern))
    if not quali_sf_matched_files:
        continue
    else:
        quali_sf_filenames.append(quali_sf_matched_files[0])
    
    quali_ss_matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_torque_path).glob(quali_ss_file_pattern))
    if not quali_ss_matched_files:
        continue
    else:
        quali_ss_filenames.append(quali_ss_matched_files[0])

#! ************************************************
#! Load training, testing data
#! ************************************************

#? Loading and epoching for training   
EMG_Data = EMGData(format="ANTmini", filenames=emg_filenames, data_path=cfg.filepath.data_path, f_samp=cfg.preprocess_param.f_samp, channel_names=cfg.preprocess_param.channel_names_emg)

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
                            filenames=quali_e_filenames,
                            data_path=cfg.filepath.data_path, 
                            f_samp=cfg.preprocess_param.f_samp, 
                            channel_names=cfg.preprocess_param.channel_names_quali, 
                            add_marker_channel=True)

print("Creating Quali Shoulder Front object...")
Quali_Data_Front = EEGData(format="NumpyQualisys", 
                            filenames=quali_sf_filenames,
                            data_path=cfg.filepath.data_path, 
                            f_samp=cfg.preprocess_param.f_samp, 
                            channel_names=cfg.preprocess_param.channel_names_quali, 
                            add_marker_channel=True)

print("Creating Quali Shoulder Side object...")
Quali_Data_Side = EEGData(format="NumpyQualisys", 
                            filenames=quali_ss_filenames, 
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

#? High pass filter
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

#? Plot and print specific variance filtered windows 
# var_filtered_window_x = EMG_Data.filtered_data
# print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
# print(f"Variance filtered windows: {var_filtered_window_x[4,:]}")

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
# print(channelwise_mvc)

print("Performing Input Normalization with Max Voluntary Contraction ...")
if cfg.preprocess_param.normalisation_method == 'overall_mvc':
    EMG_Data.normalizeContinuousData(mvc=np.max(channelwise_mvc))
elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
    EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)
else:
    warnings.warn("This method is not yet implemented!! Omitting!")
print("Input Normalization with Max Voluntary Contraction performed!!\n")

#? Output Normalisation
# print("Calculating maximum absolute torque for output normalisation...")
# max_torque_e = np.max(np.abs(Quali_Data_Elbow.data), axis=1).reshape(-1,1)
# max_torque_sf = np.max(np.abs(Quali_Data_Front.data), axis=1).reshape(-1,1)
# max_torque_ss = np.max(np.abs(Quali_Data_Side.data), axis=1).reshape(-1,1)

# print("Performing Output Normalisation with Max Value...")
# Quali_Data_Elbow.normalizeContinuousData(mvc=max_torque_e)
# Quali_Data_Front.normalizeContinuousData(mvc=max_torque_sf)
# Quali_Data_Side.normalizeContinuousData(mvc=max_torque_ss)
# print("Output Normalisation with max. value performed!!\n")

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
ref_emg_window_boundary_idx = emg_window_boundary_idx
ref_quali_window_boundary_idx = quali_window_boundary_idx

slice_ref = []
if emg_window_boundary_idx == quali_window_boundary_idx:
    print("Window boundary indices match!!")
else:
    for idx, (emg_win, quali_win) in enumerate(zip(emg_window_boundary_idx, quali_window_boundary_idx)):
        if emg_win != quali_win:
            delta = emg_win - quali_win  # positive if EMG is larger
            print(f"Window mismatch -> EMG: {emg_win}; Quali: {quali_win}; Delta: {delta}")
            for i in range(idx, len(emg_window_boundary_idx)):
                emg_window_boundary_idx[i] -= delta
            slice_ref.append((idx, delta))

#? Slice and concat windows
if len(slice_ref) > 0:
    for i in range (len(slice_ref)):
        idx, delta = slice_ref[i]
        if delta > 0:
            windows_to_remove = list(range(ref_emg_window_boundary_idx[idx]-1, ref_emg_window_boundary_idx[idx]-1+delta))
            print(windows_to_remove)
            EMG_Data.windows = np.delete(EMG_Data.getWindows(), windows_to_remove, axis=-1)
            EMG_Data_freq.windows = np.delete(EMG_Data_freq.getWindows(), windows_to_remove, axis=-1)
        elif delta <0:
            print("Entered quali part")
            delta = np.abs(delta)
            windows_to_remove = list(range(ref_quali_window_boundary_idx[idx-1], ref_quali_window_boundary_idx[idx-1+delta]))
            Quali_Data_Elbow.windows = np.delete(Quali_Data_Elbow.getWindows(), idx, axis=-1)
            Quali_Data_Front.windows = np.delete(Quali_Data_Elbow.getWindows(), idx, axis=-1)
            Quali_Data_Side.windows = np.delete(Quali_Data_Elbow.getWindows(), idx, axis=-1)

ref_window_boundary_idx = emg_window_boundary_idx
print("Windows sliced and equalled!!")
    
assert EMG_Data.windows.shape[3] == Quali_Data_Elbow.windows.shape[3]

#? Get number of windows for each file
ref_window_sizes = [ref_window_boundary_idx[0]]
for i in range(1, len(ref_window_boundary_idx)):
    ref_window_sizes.append(ref_window_boundary_idx[i] - ref_window_boundary_idx[i-1])

#? One Hot Encoding for 3 weights and 2 movements
weights_code = []
mov_code = []
idx_counter = 0

for mov, wgt in itertools.product(cfg.data_param.mov_type, cfg.data_param.weights):
    if mov == 'grasp':
        mov_code += [0] * (ref_window_sizes[2*idx_counter] + ref_window_sizes[2*idx_counter + 1])
    elif mov == 'complex':
        mov_code += [1] * (ref_window_sizes[2*idx_counter] + ref_window_sizes[2*idx_counter + 1])
    else:
        raise ValueError("Wrong mov type string added... move should be either grasp or complex!!")
    
    if wgt == '0g':
        weights_code += [0] * (ref_window_sizes[2*idx_counter] + ref_window_sizes[2*idx_counter + 1])
    elif wgt == '1100g':
        weights_code += [1] * (ref_window_sizes[2*idx_counter] + ref_window_sizes[2*idx_counter + 1])
    elif wgt == '1850g':
        weights_code += [2] * (ref_window_sizes[2*idx_counter] + ref_window_sizes[2*idx_counter + 1])
    else:
        raise ValueError("Wrong weight string added... weight should be either 0g, 1100g or 1850g!!")
    
    idx_counter += 1

#? Merge weights_code and mov_code to form categorical feat set
x_cat = np.column_stack([weights_code, mov_code])

# use the EMG_Data.windows if you want to access the windowed data 
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
# EMG_Data.printFeatureShape()
#? time domain feature extraction
## EMG Feature Extraction
rms_feature = EMG_Data.getRMSFeatures_windows(n_channels=len(channel_names)) # RMS value
EMG_Data.addFeatures(rms_feature)
# print(EMG_Data.getFeatures()[1,:])
wfl_feature = EMG_Data.getWaveformLengthFeatures_windows(n_channels=len(channel_names))  # Waveform length
EMG_Data.addFeatures(wfl_feature)
# print(EMG_Data.getFeatures()[1,:])
ssc_feature = EMG_Data.getSlopeSignChangeFeatures_windows(n_channels=len(channel_names), 
                                                          threshold=0.02)    # Slope Sign Change
EMG_Data.addFeatures(ssc_feature)
# print(EMG_Data.getFeatures()[1,:])

#? freq domain feature extraction
EMG_Data_freq.featureExtractionFromWindows(feature_type="freqBandPower",
                                           psd_method="multitaper",
                                           freq_bands=[15, 50, 100, 150, 200, 245])
fbp_feature = EMG_Data_freq.getFeatures()
EMG_Data.addFeatures(fbp_feature)

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

#? Median filter on target values
# Quali_Data_Elbow.applyMedianFilter_features(window_length=21)
# Quali_Data_Front.applyMedianFilter_features(window_length=21)
# Quali_Data_Side.applyMedianFilter_features(window_length=21)

#? Savitsky-Golay filter on target values
# Quali_Data_Elbow.applySavitskyGolayFilter_features(window_length=21, poly_order=2)
# Quali_Data_Front.applySavitskyGolayFilter_features(window_length=21, poly_order=2)
# Quali_Data_Side.applySavitskyGolayFilter_features(window_length=21, poly_order=2)

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

#? Split categorical data into train, validation, and test sets
X_train_cat_temp, X_test_cat= train_test_split(x_cat,
                                          train_size=cfg.model_param.train_test_split,
                                          shuffle=False)

X_train_cat, X_val_cat, = train_test_split(X_train_cat_temp,
                                          train_size= 1 - cfg.model_param.validation_split,
                                          shuffle=False)

print("Split data into train, test, and val!!")

#? One Hot encoding
encoder = OneHotEncoder(sparse_output=False)
X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)
X_val_cat = encoder.transform(X_val_cat)

#? Creating history of features
history_len = 3
X_train, Y_train, ref_train_idx = EMG_Data.stackHistoryCat_windows(x_num=X_train, 
                                                 y_num=Y_train, 
                                                 x_cat=X_train_cat,
                                                 history_len=history_len)

X_test, Y_test, ref_test_idx = EMG_Data.stackHistoryCat_windows(x_num=X_test, 
                                               y_num=Y_test, 
                                               x_cat=X_test_cat,
                                               history_len=history_len)
X_val, Y_val, ref_val_idx = EMG_Data.stackHistoryCat_windows(x_num=X_val, 
                                             y_num=Y_val, 
                                             x_cat=X_val_cat,
                                             history_len=history_len)

print(f"{history_len} feature vectors stacked together!!")
print(f"Stacked x_train feature shape: {X_train.shape}")
print(f"Stacked y_train feature shape: {Y_train.shape}")

#? Pre-PCA scaling of input features
_, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train, 
                                                           test_data=X_test, 
                                                           val_data=X_val, 
                                                           method="StandardScaler")

#? Dimensionality Reduction - PCA
X_train, X_test, X_val = EMG_Data.reduceDimensions_windows(train_data=X_train,
                                  test_data = X_test,
                                  val_data = X_val,
                                  method="PCA",
                                  n_components=0.99)

#? Scale the input features -> StandardScaler
_, X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train, 
                                                           test_data=X_test, 
                                                           val_data=X_val, 
                                                           method="StandardScaler")

#? Extract categorical features corr to the stacked history frames
X_train_cat = X_train_cat[[end-1 for _, end in ref_train_idx]]
X_test_cat = X_test_cat[[end-1 for _, end in ref_test_idx]]
X_val_cat = X_val_cat[[end-1 for _, end in ref_val_idx]]

#? Add categorical features to the input sets
X_train = np.concatenate([X_train, X_train_cat], axis=1)
X_test = np.concatenate([X_test, X_test_cat], axis=1)
X_val = np.concatenate([X_val, X_val_cat], axis=1)

#? Scale output features -> [-1,1] for tanh
Y_scaler, Y_train, Y_test, Y_val = EMG_Data.scaleFeatures_windows(train_data=Y_train, 
                                                                  test_data=Y_test, 
                                                                  val_data=Y_val, 
                                                                  method="MinMaxScaler",
                                                                  feature_range=(-1,1))

#! ************************************************
#! Train, Load, or Test Model
#! ************************************************

# --- set global seed ---
seed = 7
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

neurons_inp = X_train.shape[1]
#? Init model with norm layer
train_model = AAN_Model(neurons_inp=neurons_inp, 
                        neurons_h1=cfg.model_param.neurons_h1, 
                        act_h1=cfg.model_param.act_h1, 
                        neurons_h2=cfg.model_param.neurons_h2, 
                        act_h2=cfg.model_param.act_h2,
                        neurons_h3=cfg.model_param.neurons_h3, 
                        act_h3=cfg.model_param.act_h3,
                        neurons_h4=cfg.model_param.neurons_h4, 
                        act_h4=cfg.model_param.act_h4,
                        neuron_out=cfg.model_param.neurons_out,
                        act_out=cfg.model_param.act_out)

MLP_model = MLModel(model = train_model, type= "keras")

#? Set the weights and deltas for weighted huber
if cfg.model_param.huber_weight_method == 'var':
    var_torques = np.var(Y_train, axis=0, ddof=1)
    weights_inp = 1.0 / (var_torques ** 0.5 + 1e-6)
    # weights_inp = weights_inp / np.sum(weights_inp)
    max_weight = np.percentile(weights_inp, 95)
    min_weight = np.percentile(weights_inp, 5)
    weights_inp = np.clip(weights_inp, min_weight, max_weight)
    weights_inp = weights_inp / np.mean(weights_inp)
elif cfg.model_param.huber_weight_method == 'smooth_var':
    weights_inp = MLP_model.getSmoothVarWeights(y_train=Y_train,
                                   clip_percentile=[5,75],
                                   window_len=5,
                                   poly_order=2)
elif cfg.model_param.huber_weight_method == 'manual':
    weights_inp = [5,5,1]
elif cfg.model_param.huber_weight_method == 'dynamic_huber':
    weights_inp = [1,1,1]
else:
    raise ValueError(f"Wrong Huber weight method chosen {cfg.model_param.huber_weight_method}... Please choose between 'var','smooth_var', 'manual', and 'dynamic_huber'!!")

MLP_model.setHuberWeights(weights_inp=weights_inp)
MLP_model.setHuberDeltas(deltas_inp=cfg.model_param.huber_deltas)

#? Train model
print("Training MLP model for elbow joint...")
save_model_path = cfg.filepath.save_model_path + filename_suffix
# print(save_model_path)
MLP_model.trainModel(save_trained_model=cfg.model_param.is_save_model, 
                     model_filename=save_model_path, 
                     train_epochs=cfg.model_param.n_epochs, 
                     batch_size=cfg.model_param.batch_size, 
                     class_weights=None, 
                     x_train=X_train, 
                     y_train=Y_train, 
                     x_val=X_val,
                     y_val=Y_val,
                     loss_fcn=cfg.model_param.loss_fcn, 
                     optimizer=cfg.model_param.optimizer, 
                     metrics=cfg.model_param.metrics, 
                     show_train_results=cfg.model_param.show_train_results, 
                     callbacks=early_callback)
print("MLP training done!!\n")

#? Predict and get results 
print("Predicting joint torques...")
MLP_model.predictTarget(data=X_test, 
                          labels=Y_test, 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=cfg.post_train_param.eval_type)

perf_results_MLP_scaled = MLP_model.getPredictionScores()

#? Rescaling output
# max_torques = np.array([max_torque_e[0], max_torque_sf[0], max_torque_ss[0]])
# perf_results_MLP = perf_results_MLP_scaled * max_torques.flatten()
# Y_ref = Y_test * max_torques.flatten()

perf_results_MLP = Y_scaler.inverse_transform(perf_results_MLP_scaled)
Y_ref = Y_scaler.inverse_transform(Y_test)

MLP_model.setPredictionScores(perf_results_MLP)

print("Pre-filtering Eval Metrics!!")
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP[:,0])
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP[:,1])
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP[:,2])
print("\n")
#! ************************************************
#! Post-prediction Filtering
#! ************************************************
#? Median filter for removing spikes/outliers
MLP_model.applyFilter_prediction(method=cfg.post_train_param.filter_type,
                                 window_length=cfg.post_train_param.filter_size)
#? Savitsky Golay filter
MLP_model.applyFilter_prediction(method="savgol",
                                 window_length=cfg.post_train_param.savgol_window_len,
                                 poly_order=cfg.post_train_param.savgol_poly_order)

perf_results_MLP = MLP_model.getPredictionScores()
# print(perf_results_MLP.shape)

#? Calculate the model eval metrics on the filtered predicted values
print("Post-filtering Eval Metrics!!")
r2_elbow, rmse_elbow = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP[:,0])
r2_front, rmse_front = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP[:,1])
r2_side, rmse_side = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP[:,2])

#? Plotting the filtered prediction results
plotResults(data_ref=Y_ref[:,0],
            label_ref="real torque", 
            data_out=perf_results_MLP[:,0], 
            label_out="predicted torque", 
            title=f"Elbow; RMSE: {rmse_elbow}N-m   R2: {r2_elbow}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

plotResults(data_ref=Y_ref[:,1],
            label_ref="real torque", 
            data_out=perf_results_MLP[:,1], 
            label_out="predicted torque", 
            title=f"Shoulder Front; RMSE: {rmse_front}N-m   R2: {r2_front}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

plotResults(data_ref=Y_ref[:,2],
            label_ref="real torque", 
            data_out=perf_results_MLP[:,2], 
            label_out="predicted torque", 
            title=f"Shoulder Side; RMSE: {rmse_side}N-m   R2: {r2_side}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

#? Showing the plots
plt.show()

#? Check if the save dir exists. If not create one
#? Create a readme.txt and include all parameters in it
if cfg.post_train_param.is_save_plot:
    choice = input("Do you want to save the plots? (Y/N)").strip().lower()
    if choice in ["y", "yes"]:
        dir_path = createOutputDir(param_obj=cfg, suffix_str="plot")
        createReadme(param_obj=cfg,
                     dir_path=dir_path)
        
        figs = [("elbow", Y_ref[:, 0], perf_results_MLP[:, 0],
         f"Elbow; RMSE: {rmse_elbow}N-m   R2: {r2_elbow}"),
        ("front", Y_ref[:, 1], perf_results_MLP[:, 1],
         f"Shoulder Front; RMSE: {rmse_front}N-m   R2: {r2_front}"),
        ("side", Y_ref[:, 2], perf_results_MLP[:, 2],
         f"Shoulder Side; RMSE: {rmse_side}N-m   R2: {r2_side}"),]
        
        for name, ref, out, title in figs:
            plt.figure()
            plotResults(data_ref=ref,
            label_ref="real torque", 
            data_out=out, 
            label_out="predicted torque", 
            title=title, 
            ylabel="Torque in N-m", 
            is_grid_on=True)

            plt.savefig(dir_path / f"test_{name}.png")
            plt.close()
        print(f"✅ Saved all plots in {dir_path}")
else:
    print("❌ Plots not saved.")