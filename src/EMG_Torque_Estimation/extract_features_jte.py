
#* This script preprocesses EMG and Qualisys data, extracts relevant temporal, spectral and time-freq features and saves them into a numpy file

#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
from sklearn.model_selection import train_test_split
from datetime import datetime
from joblib import dump

#own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

import warnings
warnings.formatwarning = customWarningFormat

#! ************************************************
#! User Parameters and Data Collection
#! ************************************************

#? load config file
config_filename = 'emg_torque_estimation_jte.yaml'
cfg = loadConfig(filename=config_filename)

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
if emg_window_boundary_idx == quali_window_boundary_idx:
    print("Window boundary indices match!!")
else:
    ref_window_boundary_idx = []
    for emg_win, quali_e_win in zip(emg_window_boundary_idx, quali_window_boundary_idx):
        if emg_win != quali_e_win:
            print(f"Window mismatch -> EMG: {emg_win}; Quali: {quali_e_win}")
        ref_window_boundary_idx.append(min(emg_win, quali_e_win))
    #? Slice and concat windows
    if ref_window_boundary_idx != emg_window_boundary_idx:
        EMG_Data.sliceAndConcatWindows(end_slice=ref_window_boundary_idx,
                                       start_slice=emg_window_boundary_idx)
        EMG_Data_freq.sliceAndConcatWindows(end_slice=ref_window_boundary_idx,
                                       start_slice=emg_window_boundary_idx)
    if ref_window_boundary_idx != quali_window_boundary_idx:
        Quali_Data_Elbow.sliceAndConcatWindows(end_slice=ref_window_boundary_idx,
                                               start_slice=quali_window_boundary_idx)
        Quali_Data_Front.sliceAndConcatWindows(end_slice=ref_window_boundary_idx,
                                               start_slice=quali_window_boundary_idx)
        Quali_Data_Side.sliceAndConcatWindows(end_slice=ref_window_boundary_idx,
                                               start_slice=quali_window_boundary_idx)
    print("Windows sliced and equalled!!")
assert EMG_Data.windows.shape[3] == Quali_Data_Elbow.windows.shape[3]

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
print("Split data into train, test, and val!!")

#? Creating history of features
history_len = 3
X_train, Y_train = EMG_Data.stackHistory_windows(x_inp=X_train, 
                                                 y_inp=Y_train, 
                                                 history_len=history_len)
X_test, Y_test = EMG_Data.stackHistory_windows(x_inp=X_test, 
                                               y_inp=Y_test, 
                                               history_len=history_len)
X_val, Y_val = EMG_Data.stackHistory_windows(x_inp=X_val, 
                                             y_inp=Y_val, 
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
#? Scale output features -> [-1,1] for tanh
Y_scaler, Y_train, Y_test, Y_val = EMG_Data.scaleFeatures_windows(train_data=Y_train, 
                                                                  test_data=Y_test, 
                                                                  val_data=Y_val, 
                                                                  method="MinMaxScaler",
                                                                  feature_range=(-1,1))

#? Save features
if cfg.preprocess_param.is_save_features:
    save_path = getAbsolutePath(input_path=cfg.filepath.save_features_path)
    if cfg.preprocess_param.use_activation_fncn:
        filename_suffix = filename_suffix + "_act"
    else:
        filename_suffix = filename_suffix
    filename_npz = "features_" + filename_suffix
    filename_pkl = "scaler_" + filename_suffix

    np.savez_compressed(save_path / (filename_npz+".npz"), 
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        Y_test=Y_test,
                        X_val=X_val,
                        Y_val=Y_val)
    
    dump(Y_scaler, save_path / (filename_pkl+".pkl"))
    print("Features saved in files!!")