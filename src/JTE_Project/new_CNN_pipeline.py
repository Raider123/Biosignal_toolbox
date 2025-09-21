# * This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

# ! ************************************************
# ! Imports
# ! ************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import time


# own libs
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, createOutputDir, createReadme, plotResults

from tensorflow.keras import layers, models, Input
from keras.utils import set_random_seed

import warnings

warnings.formatwarning = customWarningFormat

# ! ************************************************
# ! Function definition
# ! ************************************************

def build_model(input_shape_time, filters, stacks, dropout_rate, kernel_size):

    inputs = []
    branches = []

    # Zeit-Pfad (TCN)
    inp_time = Input(shape=input_shape_time, name='emg_input')
    x = inp_time

    for s in range(stacks):
        d = 4 ** s  # Dilatation: 1,2,4,8,...
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


# ! ************************************************
# ! User Parameters and Data Collection
# ! ************************************************

filters = 64
stacks = 4
dropout_rate = 0.10
kernel_size = 3

time_preproc = time.perf_counter()

# ? load config file
config_filename = 'new_jte.yaml'
cfg = loadConfig(filename=config_filename)

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

if cfg.workflow_param.use_seed:
    set_random_seed(cfg.workflow_param.seed)
    tf.config.experimental.enable_op_determinism()

# ? Read the filenames and create Arrays containing all the used filenames
emg_filenames = []
quali_e_filenames = []
quali_sf_filenames = []
quali_ss_filenames = []

for mov_type in cfg.data_param.mov_type:
    for weight in cfg.data_param.weights:
        for set_num in cfg.data_param.set_num:

            # EMG Filename
            current_emg_file_name = (
                cfg.filepath.emg_path +
                cfg.filepath.emg_file_prefix +
                weight + '_' +
                mov_type + '_' +
                set_num + '.txt'
            )

            complete_emg_file = getAbsolutePath(cfg.filepath.data_path + current_emg_file_name)

            if Path(complete_emg_file).is_file():
                # Append the EMG Filename
                emg_filenames.append(current_emg_file_name)

                # Qualisys Elbow
                quali_e_file = (
                    cfg.filepath.quali_path[0] +
                    cfg.filepath.quali_tsv_prefix +
                    weight + '_' +
                    mov_type + '_' +
                    set_num + '.npy'
                )
                quali_e_filenames.append(quali_e_file)

                # Qualisys Shoulder Front
                quali_sf_file = (
                    cfg.filepath.quali_path[1] +
                    cfg.filepath.quali_tsv_prefix +
                    weight + '_' +
                    mov_type + '_' +
                    set_num + '.npy'
                )
                quali_sf_filenames.append(quali_sf_file)

                # Qualisys Shoulder Side
                quali_ss_file = (
                    cfg.filepath.quali_path[2] +
                    cfg.filepath.quali_tsv_prefix +
                    weight + '_' +
                    mov_type + '_' +
                    set_num + '.npy'
                )
                quali_ss_filenames.append(quali_ss_file)

print(quali_e_filenames)
print(quali_sf_filenames)
print(quali_ss_filenames)

# ! ************************************************
# ! Load training, testing data
# ! ************************************************

# ? Loading and epoching for training
EMG_Data = EMGData(format="ANTmini", filenames=emg_filenames, data_path=cfg.filepath.data_path,
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
    plt.plot(np.arange(0, Quali_Data_Elbow.data[0, :].shape[0], 1) / Quali_Data_Elbow.f_samp,
             Quali_Data_Elbow.data[0, :])
    plt.title("Elbow Torque plot for right arm")
    plt.xlabel("Time in s")
    plt.ylabel("Torque in N-m")
    plt.grid()
    plt.show()

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
channelwise_mvc = np.max(np.abs(EMG_Data.data), axis=1).reshape(-1, 1)
# print(channelwise_mvc)

print("Performing Input Normalization with Max Voluntary Contraction ...")
if cfg.preprocess_param.normalisation_method == 'overall_mvc':
    EMG_Data.normalizeContinuousData(mvc=np.max(channelwise_mvc))
elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
    EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)
else:
    warnings.warn("This method is not yet implemented!! Omitting!")
print("Input Normalization with Max Voluntary Contraction performed!!\n")

# ? Output Normalisation
""" print("Calculating maximum absolute torque for output normalisation...")
max_torque_e = np.max(np.abs(Quali_Data_Elbow.data), axis=1).reshape(-1, 1)
max_torque_sf = np.max(np.abs(Quali_Data_Front.data), axis=1).reshape(-1, 1)
max_torque_ss = np.max(np.abs(Quali_Data_Side.data), axis=1).reshape(-1, 1)

print("Performing Output Normalisation with Max Value...")
Quali_Data_Elbow.normalizeContinuousData(mvc=max_torque_e)
Quali_Data_Front.normalizeContinuousData(mvc=max_torque_sf)
Quali_Data_Side.normalizeContinuousData(mvc=max_torque_ss)
print("Output Normalisation with max. value performed!!\n") """

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

time_preproc_end = time.perf_counter()

time_feat_ext = time.perf_counter()

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
# Convert to numpy arrays
emg_idx = np.array(emg_window_boundary_idx)
quali_idx = np.array(quali_window_boundary_idx)

# Ensure same length
min_len = min(len(emg_idx), len(quali_idx))
emg_idx = emg_idx[:min_len]
quali_idx = quali_idx[:min_len]

# Take element-wise minimum as reference
ref_idx = np.minimum(emg_idx, quali_idx)

# --- Zusatzfeatures: Gewicht & Bewegung pro Fenster (robust über ref_idx) ---
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# ref_idx enthält die kumulierten Fensterenden pro Datei nach dem Slicing
ref_idx_arr = np.array(ref_idx, dtype=int)
starts = np.concatenate(([0], ref_idx_arr[:-1]))
ends = ref_idx_arr

w_codes = []
m_codes = []
span_slices = []  # (start, end, weight_str, mov_str) für spätere Ziel-Skalierung

for i, (s, e) in enumerate(zip(starts, ends)):
    n = int(e - s)
    parts = Path(emg_filenames[i]).stem.split("_")
    w_str = parts[-3]   # z.B. '0g' | '1100g' | '1850g'
    m_str = parts[-2]   # z.B. 'grasp' | 'complex'
    w_codes.extend([w_str] * n)
    m_codes.extend([m_str] * n)
    span_slices.append((s, e, w_str, m_str))

w_codes = np.array(w_codes)
m_codes = np.array(m_codes)

enc_w = OneHotEncoder(sparse_output=False)
enc_m = OneHotEncoder(sparse_output=False)
w_one = enc_w.fit_transform(w_codes.reshape(-1, 1))
m_one = enc_m.fit_transform(m_codes.reshape(-1, 1))
extra_features = np.hstack([w_one, m_one]).astype(np.float32)


# Slice EMG windows if needed
if not np.array_equal(ref_idx, emg_idx):
    EMG_Data.sliceAndConcatWindows(end_slice=ref_idx, start_slice=emg_idx)
    EMG_Data_freq.sliceAndConcatWindows(end_slice=ref_idx, start_slice=emg_idx)

# Slice Quali windows if needed
if not np.array_equal(ref_idx, quali_idx):
    Quali_Data_Elbow.sliceAndConcatWindows(end_slice=ref_idx, start_slice=quali_idx)
    Quali_Data_Front.sliceAndConcatWindows(end_slice=ref_idx, start_slice=quali_idx)
    Quali_Data_Side.sliceAndConcatWindows(end_slice=ref_idx, start_slice=quali_idx)

# --- Extra Safety: Trim to the minimum number of windows after slicing ---
num_windows = min(EMG_Data.windows.shape[3], Quali_Data_Elbow.windows.shape[3])

EMG_Data.windows = EMG_Data.windows[:, :, :, :num_windows]
EMG_Data_freq.windows = EMG_Data_freq.windows[:, :, :, :num_windows]
Quali_Data_Elbow.windows = Quali_Data_Elbow.windows[:, :, :, :num_windows]
Quali_Data_Front.windows = Quali_Data_Front.windows[:, :, :, :num_windows]
Quali_Data_Side.windows = Quali_Data_Side.windows[:, :, :, :num_windows]

print(f"All windows trimmed to {num_windows} frames. ✅")


# use the EMG_Data.windows if you want to access the windowed data
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
'''
print("Extracting features from windowed data...")

# ? EMG signal timepoints feature extraction
window_size_ms = cfg.preprocess_param.window_size_x * 1000 / EMG_Data.f_samp
feature_indices_windows_x = np.array([0, window_size_ms])
EMG_Data.featureExtractionFromWindows(feature_type="timepoints",
                                      feature_indices_windows=feature_indices_windows_x)
EMG_Data.printFeatureShape()
# ? time domain feature extraction
## EMG Feature Extraction
rms_feature = EMG_Data.getRMSFeatures_windows(n_channels=len(channel_names))  # RMS value
EMG_Data.addFeatures(rms_feature)
# print(EMG_Data.getFeatures()[1,:])
wfl_feature = EMG_Data.getWaveformLengthFeatures_windows(n_channels=len(channel_names))  # Waveform length
EMG_Data.addFeatures(wfl_feature)
# print(EMG_Data.getFeatures()[1,:])
ssc_feature = EMG_Data.getSlopeSignChangeFeatures_windows(n_channels=len(channel_names),threshold=0.02)  # Slope Sign Change
EMG_Data.addFeatures(ssc_feature)
# print(EMG_Data.getFeatures()[1,:])

# ? freq domain feature extraction
EMG_Data_freq.featureExtractionFromWindows(feature_type="freqBandPower",
                                           psd_method="multitaper",
                                           freq_bands=[15, 50, 100, 150, 200, 245])
fbp_feature = EMG_Data_freq.getFeatures()
EMG_Data.addFeatures(fbp_feature)

# ? time-freq domain feature extraction
freqs = np.arange(start=50, stop=226, step=25)
n_cycles = np.ones(len(freqs)) * 5
n_cycles[0] = 3
n_cycles[1] = 4
mwc_feature = EMG_Data_freq.getMorletWaveletCoeffFeatures_windows(freqs=freqs,
                                                                  n_cycles=n_cycles)  # Morlet transform
EMG_Data.addFeatures(mwc_feature)
print(f"Total EMG features extracted: {EMG_Data.getFeatures().shape}")

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
print("Feature extraction from windowed data completed!!\n")

# ? Merge output features
target_features = np.concatenate(
    [Quali_Data_Elbow.getFeatures(), Quali_Data_Front.getFeatures(), Quali_Data_Side.getFeatures()], axis=1)

# ? Creating history of features
history_len = 3
input_features = EMG_Data.getFeatures()
input_features_hist = np.zeros((input_features.shape[0] - history_len + 1, history_len * input_features.shape[1]))
target_features_hist = np.zeros((target_features.shape[0] - history_len + 1, target_features.shape[1]))

for i in range(history_len, input_features.shape[0] + 1):
    input_features_hist[i - history_len] = input_features[i - history_len:i].flatten()
    target_features_hist[i - history_len] = target_features[i - 1]

# input_features_hist = input_features_hist.astype(np.float32)
# target_features_hist = target_features_hist.astype(np.float32)
print(f"Target: {target_features_hist.shape}")
# ? Setter for feature vec
EMG_Data.setFeatures(features_inp=input_features_hist)

# ? Dimensionality Reduction - PCA
#EMG_Data.reduceDimensions_windows(method="PCA",
#                                  n_components=0.98)
#print(f"Reduced feature set: {EMG_Data.getFeatures().shape}")

# ? Split data into train, validation, and test sets
X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(EMG_Data.getFeatures(),
                                                              target_features_hist,
                                                              train_size=cfg.model_param.train_test_split,
                                                              shuffle=False)

X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp,
                                                  Y_train_temp,
                                                  train_size=1 - cfg.model_param.validation_split,
                                                  shuffle=False)

# ? Scale the features -> StandardScaler
X_train, X_test, X_val = EMG_Data.scaleFeatures_windows(train_data=X_train,
                                                        test_data=X_test,
                                                        val_data=X_val,
                                                        method="StandardScaler")
'''

# extract raw emg
x = EMG_Data.getWindows()[0]  # (n_channels, n_samples, n_windows)
# reshape to (n_windows, n_samples, n_channels)
x = np.transpose(x, (2, 1, 0))  # (n_windows, n_samples, n_channels)

# --------------------------------------------
# Zusatzfeatures auf Fenster-Länge bringen
# --------------------------------------------
n_windows = x.shape[0]  # Anzahl der EMG-Fenster

print("Extracting features from windowed data ...")
# defining the feature window sizes
window_size_ms = cfg.preprocess_param.window_size_y * 1000 / Quali_Data_Elbow.f_samp
if cfg.preprocess_param.target_feature_select == 'mean':
    feature_indices_windows_x = np.array([0, window_size_ms])
    feature_indices_windows_y = np.array([0, window_size_ms])
    use_mean_bool = True
elif cfg.preprocess_param.target_feature_select == 'mid':
    feature_indices_windows_x = np.array([(window_size_ms / 2) - 2, window_size_ms / 2])
    feature_indices_windows_y = np.array([(window_size_ms / 2) - 2, window_size_ms / 2])
    use_mean_bool = False
elif cfg.preprocess_param.target_feature_select == 'end':
    feature_indices_windows_x = np.array([window_size_ms - 2, window_size_ms])
    feature_indices_windows_y = np.array([window_size_ms - 2, window_size_ms])
    use_mean_bool = False
else:
    raise ValueError(
        f"Provided target_feature_select {cfg.preprocess_param.target_feature_select} is not yet implemented... Please choose between 'mean', 'mid', and 'end'")

# extracting the input/output features (raw timepoints)
EMG_Data.featureExtractionFromWindows(feature_type="timepoints", feature_indices_windows=feature_indices_windows_x)
Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints",
                                              feature_indices_windows=feature_indices_windows_y,
                                              use_mean=use_mean_bool)
Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints",
                                              feature_indices_windows=feature_indices_windows_y,
                                              use_mean=use_mean_bool)
Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints",
                                             feature_indices_windows=feature_indices_windows_y,
                                             use_mean=use_mean_bool)
print("Feature extraction from windowed data completed !!\n")

# extract output feature labels
y_e = Quali_Data_Elbow.getFeatures()[:, 0]  # (n_windows,)
y_f = Quali_Data_Front.getFeatures()[:, 0]
y_s = Quali_Data_Side.getFeatures()[:, 0]

# Merge all three targets
Y = np.stack([y_e, y_f, y_s], axis=-1)  # (n_windows, 3)

# --- Ziel-Skalierung pro Datei/Gruppe ([-1, 1]) ---
from sklearn.preprocessing import MinMaxScaler

Y = Y.astype(np.float32)
scalers = {}  # key: (w_str, m_str) -> MinMaxScaler

for s, e, w_str, m_str in span_slices:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Y[s:e] = scaler.fit_transform(Y[s:e])
    scalers[(w_str, m_str)] = scaler


# Creating history of features (Y --> target_features_hist but with kernel 3)
history_len = 3
input_features = EMG_Data.getFeatures()
input_features_hist = np.zeros((input_features.shape[0] - history_len + 1, history_len * input_features.shape[1]))
target_features_hist = np.zeros((Y.shape[0] - history_len + 1, Y.shape[1]))

for i in range(history_len, input_features.shape[0] + 1):
    input_features_hist[i - history_len] = input_features[i - history_len:i].flatten()
    target_features_hist[i - history_len] = Y[i - 1]
input_features_hist = input_features_hist.astype(np.float32)
target_features_hist = target_features_hist.astype(np.float32)

# Make sure that input and output feature lengths are equal
min_len = min(x.shape[0], target_features_hist.shape[0])
x = x[:min_len]
target_features_hist = target_features_hist[:min_len]

# ---- Split in Train/Val/Test using k-fold crossvalidation ----
X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(
    x, target_features_hist,
    train_size=cfg.model_param.train_test_split,
    shuffle=False
)

if not cfg.model_param.use_k_fold:
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_temp, Y_train_temp,
        train_size=1 - cfg.model_param.validation_split,
        shuffle=False)
    kf = KFold(n_splits=2)
else:
    kf = KFold(n_splits=cfg.model_param.k_fold_splits, shuffle=cfg.model_param.k_fold_shuffle)

time_feat_ext_end = time.perf_counter()

all_rmse_e, all_rmse_f, all_rmse_s = [], [], []
all_r2_e, all_r2_f, all_r2_s = [], [], []
all_pcc_e, all_pcc_f, all_pcc_s = [], [], []
training_time = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_temp)):
    print(f"--- Fold {fold+1} ---")
    if cfg.model_param.use_k_fold:
        X_train, X_val = X_train_temp[train_idx], X_train_temp[val_idx]
        Y_train, Y_val = Y_train_temp[train_idx], Y_train_temp[val_idx]

    # ! ************************************************
    # ! Preparing the data for the TCN-Model
    # ! ************************************************
    '''
    timesteps = history_len
    n_features = X_train.shape[1] // timesteps

    X_train_seq = X_train.reshape(-1, timesteps, n_features)
    X_val_seq   = X_val.reshape(-1, timesteps, n_features)
    X_test_seq  = X_test.reshape(-1, timesteps, n_features)

    print("Reshaped input arrays for the model training.")
    '''
    # For raw data, reshaping is not necessary (n_channels, n_samples, n_windows)
    timesteps  = X_train.shape[1]
    n_features = X_train.shape[2]
    input_shape_time = (timesteps, n_features)

    X_train_seq = X_train
    X_val_seq   = X_val
    X_test_seq  = X_test


    # Prepare the training data
    Y_train_e = Y_train[:, 0]
    Y_train_f = Y_train[:, 1]
    Y_train_s = Y_train[:, 2]

    Y_val_e = Y_val[:, 0]
    Y_val_f = Y_val[:, 1]
    Y_val_s = Y_val[:, 2]

    # Prepare the test data
    Y_test_e = Y_test[:, 0]
    Y_test_f = Y_test[:, 1]
    Y_test_s = Y_test[:, 2]

    time_model = time.perf_counter()

    # ! ************************************************
    # ! Building and Compiling the TCN-Model
    # ! ************************************************

    model = build_model(
        input_shape_time=input_shape_time,
        filters=filters,
        stacks=stacks,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss={
            "torque_elbow": "mse",
            "torque_shoulder_front": "mse",
            "torque_shoulder_side": "mse",
        })
    '''
    ###EXPERIMENTAL
    # --- Varianz-basierte Loss-Gewichtung (ähnlich MLP 'var') ---
    var_torques = np.var(Y_train, axis=0, ddof=1)
    weights_inp = 1.0 / (np.sqrt(var_torques) + 1e-6)
    # optional clippen/normalisieren wie in MLP
    p95, p05 = np.percentile(weights_inp, 95), np.percentile(weights_inp, 5)
    weights_inp = np.clip(weights_inp, p05, p95)
    weights_inp = weights_inp / np.mean(weights_inp)

    loss_weights = {
        "torque_elbow": float(weights_inp[0]),
        "torque_shoulder_front": float(weights_inp[1]),
        "torque_shoulder_side": float(weights_inp[2]),
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "torque_elbow": tf.keras.losses.Huber(delta=1.0),
            "torque_shoulder_front": tf.keras.losses.Huber(delta=1.0),
            "torque_shoulder_side": tf.keras.losses.Huber(delta=1.0),
        },
        loss_weights=loss_weights,
    )


    ###
    '''

    # ! ************************************************
    # ! Training the TCN-Model
    # ! ************************************************

    history = model.fit(
        X_train_seq,
        {"torque_elbow": Y_train_e,
        "torque_shoulder_front": Y_train_f,
        "torque_shoulder_side": Y_train_s},
        validation_data=(
            X_val_seq,
            {"torque_elbow": Y_val_e,
            "torque_shoulder_front": Y_val_f,
            "torque_shoulder_side": Y_val_s}
        ),
        epochs=cfg.model_param.n_epochs,
        batch_size=cfg.model_param.batch_size,
        callbacks=[early_callback] if early_callback else None
    )

    time_model_end = time.perf_counter()
    passed_time = time_model_end - time_model
    training_time.append(passed_time)
    print(f"Model_Training im K_Fold: {passed_time:.4f} Sekunden")

    # ! ************************************************
    # ! Model-Prediction
    # ! ************************************************

    # Prediction for all three joints
    y_pred = model.predict(X_test_seq)

    # y_pred is a list [pred_elbow, pred_front, pred_side]
    pred_elbow, pred_front, pred_side = y_pred

    predictions_e = pred_elbow.flatten()
    predictions_f = pred_front.flatten()
    predictions_s = pred_side.flatten()

    ''' 
    ### EXPERIMENTAL
    # --- Inverse Skalierung der Test-Predictions auf Originaleinheiten ---
    # Rekonstruiere die globalen Test-Indices im Gesamtsignal (keine Shuffle!)
    test_start = X_train.shape[0] + X_val.shape[0]
    test_end = test_start + X_test.shape[0]

    # Stapel Vorhersagen (skaliert) in (n_test, 3)
    Y_pred_test_scaled = np.stack([predictions_e, predictions_f, predictions_s], axis=-1)

    # Platzhalter für unskalierte Wahrheiten und Vorhersagen
    Y_test_true_unscaled = np.empty_like(Y_pred_test_scaled, dtype=np.float32)
    Y_test_pred_unscaled = np.empty_like(Y_pred_test_scaled, dtype=np.float32)

    cursor = 0  # Fortschritt innerhalb des Test-Bereichs
    for s, e, w_str, m_str in span_slices:
        a = max(s, test_start)
        b = min(e, test_end)
        if a < b:
            k = b - a
            scaler = scalers[(w_str, m_str)]
            Y_test_pred_unscaled[cursor:cursor+k] = scaler.inverse_transform(Y_pred_test_scaled[cursor:cursor+k])
            Y_test_true_unscaled[cursor:cursor+k] = scaler.inverse_transform(Y[a:b])
            cursor += k

    # Überschreibe für nachfolgende Auswertung
    Y_test_e = Y_test_true_unscaled[:, 0]
    Y_test_f = Y_test_true_unscaled[:, 1]
    Y_test_s = Y_test_true_unscaled[:, 2]
    predictions_e = Y_test_pred_unscaled[:, 0]
    predictions_f = Y_test_pred_unscaled[:, 1]
    predictions_s = Y_test_pred_unscaled[:, 2]
    ###
    '''
    # ------------------------------------------------------------------------------
    # Post-Processing
    # ------------------------------------------------------------------------------

    window_length = cfg.post_processing_param.savitzky_window_length
    polyorder = cfg.post_processing_param.savitzky_polyorder

    if cfg.post_processing_param.savitzky_on:
        from scipy.signal import savgol_filter

        predictions_e = savgol_filter(predictions_e, window_length, polyorder)
        predictions_f = savgol_filter(predictions_f, window_length, polyorder)
        predictions_s = savgol_filter(predictions_s, window_length, polyorder)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    filter_window_size = cfg.post_processing_param.moving_av_window_size
    if cfg.post_processing_param.moving_av_on:
        predictions_e = moving_average(predictions_e, filter_window_size)
        predictions_f = moving_average(predictions_f, filter_window_size)
        predictions_s = moving_average(predictions_s, filter_window_size)

    # ------------------------------------------------------------------------------
    # RMSE-Calculation (Test + R²)
    # ------------------------------------------------------------------------------

    # Test-RMSE
    rmse_e = np.sqrt(mean_squared_error(Y_test_e, predictions_e))
    rmse_f = np.sqrt(mean_squared_error(Y_test_f, predictions_f))
    rmse_s = np.sqrt(mean_squared_error(Y_test_s, predictions_s))
    all_rmse_e.append(rmse_e)
    all_rmse_f.append(rmse_f)
    all_rmse_s.append(rmse_s)

    # R² for Test
    r2_e = r2_score(Y_test_e, predictions_e)
    r2_f = r2_score(Y_test_f, predictions_f)
    r2_s = r2_score(Y_test_s, predictions_s)
    all_r2_e.append(r2_e)
    all_r2_f.append(r2_f)
    all_r2_s.append(r2_s)

    # Pearson Correlation coefficient for test
    pcc_e = pearsonr(Y_test_e, predictions_e).statistic
    pcc_f = pearsonr(Y_test_f, predictions_f).statistic
    pcc_s = pearsonr(Y_test_s, predictions_s).statistic
    all_pcc_e.append(pcc_e)
    all_pcc_f.append(pcc_f)
    all_pcc_s.append(pcc_s)

    print('Ergebnisse (Multi-Task):')
    print(f"Ellbogen       -> Test-RMSE: {rmse_e:.2f}  | R²: {r2_e:.3f} | PCC: {pcc_e:.3f}")
    print(f"Schulter Front -> Test-RMSE: {rmse_f:.2f}  | R²: {r2_f:.3f} | PCC: {pcc_f:.3f}")
    print(f"Schulter Side  -> Test-RMSE: {rmse_s:.2f}  | R²: {r2_s:.3f} | PCC: {pcc_s:.3f}")

    if not cfg.model_param.use_k_fold:
        break

print('Ergebnisse (k-Fold):')
print(f"Ellbogen       -> Test-RMSE: {np.mean(all_rmse_e):.2f} ± {np.std(all_rmse_e):.3f}  | R²: {np.mean(all_r2_e):.3f} ± {np.std(all_r2_e):.3f} | PCC: {np.mean(all_pcc_e):.3f} ± {np.std(all_pcc_e):.3f}")
print(f"Schulter Front -> Test-RMSE: {np.mean(all_rmse_f):.2f} ± {np.std(all_rmse_f):.3f}  | R²: {np.mean(all_r2_f):.3f} ± {np.std(all_r2_f):.3f} | PCC: {np.mean(all_pcc_f):.3f} ± {np.std(all_pcc_f):.3f}")
print(f"Schulter Side  -> Test-RMSE: {np.mean(all_rmse_s):.2f} ± {np.std(all_rmse_s):.3f}  | R²: {np.mean(all_r2_s):.3f} ± {np.std(all_r2_s):.3f} | PCC: {np.mean(all_pcc_s):.3f} ± {np.std(all_pcc_s):.3f}")

# Timings
passed_preproc_time = time_preproc_end - time_preproc
print(f"Preprocessing Zeit: {passed_preproc_time :.4f} Sekunden")

passed_feat_time = time_feat_ext_end - time_feat_ext
print(f"Feature Extraction Zeit: {passed_feat_time :.4f} Sekunden")

passed_train_time = np.mean(training_time)
train_time_std = np.std(training_time)
print(f"Model Training Zeit: {passed_train_time :.4f} ± {train_time_std :.2f} Sekunden")
# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
plotResults(Y_test_e, "real torque",
            predictions_e, "predicted torque",
            f"Elbow Joint Filtered; RMSE: {rmse_e:.4f} N-m | R²: {r2_e:.3f} | PCC: {pcc_e:.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(Y_test_f, "real torque",
            predictions_f, "predicted torque",
            f"Shoulder Front Joint Filtered; RMSE: {rmse_f:.4f} N-m | R²: {r2_f:.3f} | PCC: {pcc_f:.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(Y_test_s, "real torque",
            predictions_s, "predicted torque",
            f"Shoulder Side Joint Filtered; RMSE: {rmse_s:.4f} N-m | R²: {r2_s:.3f} | PCC: {pcc_f:.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True)

plt.show()
