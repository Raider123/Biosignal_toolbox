# This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import joblib


# # own libs
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.utils import loadConfig, plotResults

from tensorflow.keras.models import load_model

import warnings

warnings.filterwarnings('ignore')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
# Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ! load config file
from types import SimpleNamespace

config_filename = 'emg_torque_estimation_mav.yaml'

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns

config_param = namespace_to_dict(loadConfig(filename=config_filename))

# Check the script mode (training or re-using saved model for inference)
RUN_MODE  = config_param['workflow_param']['mode'] # "train" | "infer"
MODEL_DIR = Path(config_param['workflow_param']['model_dir'])
MODEL_DIR.mkdir(parents=True, exist_ok=True)                 # legt Ordner an

# options: "time", "freq", "time+freq"
INPUT_MODE = config_param['workflow_param'].get('input_mode', "time+freq")

# ! init early stopping
if config_param['model_param']['is_early_stop']:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=config_param['model_param']['monitor'],
                                                      min_delta=config_param['model_param']['min_delta'],
                                                      patience=config_param['model_param']['patience'],
                                                      verbose=config_param['model_param']['verbose'],
                                                      baseline=config_param['model_param']['baseline'],
                                                      restore_best_weights=config_param['model_param'][
                                                          'restore_best_weights'])
else:
    early_callback = None

# ! feature extraction window
if config_param['preprocess_param']['feature_select'] == "end":
    ## Indices to extract features from the end of the window
    feature_indices_windows_x = np.arange(
        config_param['preprocess_param']['window_size_x'] - config_param['preprocess_param']['feature_size'],
        config_param['preprocess_param']['window_size_x'], step=1)
    feature_indices_windows_e = np.arange(config_param['preprocess_param']['window_size_e'] - 1,
                                          config_param['preprocess_param']['window_size_e'], step=1)
    feature_indices_windows_s = np.arange(config_param['preprocess_param']['window_size_s'] - 1,
                                          config_param['preprocess_param']['window_size_s'], step=1)
    feature_indices_windows_f = np.arange(config_param['preprocess_param']['window_size_f'] - 1,
                                          config_param['preprocess_param']['window_size_f'], step=1)
elif config_param['preprocess_param']['feature_select'] == "mid":
    ## Indices to extract features from the middle of the window
    feature_indices_windows_x = np.arange(
        round(config_param['preprocess_param']['window_size_x'] / 2) - config_param['preprocess_param'][
            'feature_size'] / 2,
        round(config_param['preprocess_param']['window_size_x'] / 2) + config_param['preprocess_param'][
            'feature_size'] / 2, step=1)
    feature_indices_windows_e = np.arange(round(config_param['preprocess_param']['window_size_e'] / 2) - 1,
                                          round(config_param['preprocess_param']['window_size_e'] / 2), step=1)
    feature_indices_windows_s = np.arange(round(config_param['preprocess_param']['window_size_s'] / 2) - 1,
                                          round(config_param['preprocess_param']['window_size_s'] / 2), step=1)
    feature_indices_windows_f = np.arange(round(config_param['preprocess_param']['window_size_f'] / 2) - 1,
                                          round(config_param['preprocess_param']['window_size_f'] / 2), step=1)
else:
    print("Please select a valid feature selection type!")

# ! subject params
if len(config_param['data_param']['mov_type']) > 1:

    scenario_name = config_param['data_param']['mov_type'][0] + '_' + config_param['data_param']['mov_type'][1]
else:
    scenario_name = config_param['data_param']['mov_type'][0]

result_file_name = '_' + config_param['data_param']['weights'][0]

# ! Initialise arrays to append data
n_channels_from_config = int(len(config_param['preprocess_param']['channel_names_emg']))
length_of_each_feature_window = int(config_param['preprocess_param']['window_size_x'])

x_test_combined = np.empty(shape=[0, length_of_each_feature_window])
x_train_combined = np.empty(shape=[0, length_of_each_feature_window])

y_e_train_combined = np.empty((0,))
y_e_test_combined = np.empty((0,))

y_f_train_combined = np.empty((0,))
y_f_test_combined = np.empty((0,))

y_s_train_combined = np.empty((0,))
y_s_test_combined = np.empty((0,))

window_train_combined = np.empty((0, length_of_each_feature_window, n_channels_from_config))  # z.B. (0, 50, 8)
window_test_combined = np.empty((0, length_of_each_feature_window, n_channels_from_config))

freq_train_combined = None
freq_test_combined = None

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
from scipy.signal import welch
import numpy as np

def extract_freq_features(windows, fs):
    """
    windows: np.array, shape (n_windows, n_samples, n_channels)
    fs: sampling rate
    returns: np.array, shape (n_windows, n_channels * n_bands * 2)
             pro Kanal und Band: [log_bandpower, rel_bandpower]
    """

    # typische EMG-Bänder (anpassen je nach fs!)
    #bands = [(20, 60), (60, 120), (120, 250), (250, 450)]
    bands =  [(20, 60), (60, 150), (150, 250)]

    n_windows, n_samples, n_channels = windows.shape
    n_bands = len(bands)
    features = np.zeros((n_windows, n_channels * n_bands * 2), dtype=np.float32)

    nperseg = min(256, n_samples)
    noverlap = nperseg // 2
    eps = 1e-12

    for w in range(n_windows):
        feat_w = []
        for ch in range(n_channels):
            f, Pxx = welch(
                windows[w, :, ch],
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density"
            )
            total_power = np.trapz(Pxx, f) + eps
            for (fmin, fmax) in bands:
                idx = np.logical_and(f >= fmin, f <= fmax)
                if not np.any(idx):
                    band_power = 0.0
                else:
                    band_power = np.trapz(Pxx[idx], f[idx])
                log_bp = np.log10(band_power + eps)
                rel_bp = band_power / total_power
                feat_w.extend([log_bp, rel_bp])
        features[w, :] = np.array(feat_w, dtype=np.float32)

    return features

for typ_idx in range(len(config_param['data_param']['mov_type'])):
    for wgt_idx in range(len(config_param['data_param']['weights'])):
        for set_idx in range(len(config_param['data_param']['set_num'])):

            current_emg_file_name = config_param['filepath']['emg_path'] + config_param['data_param']['weights'][wgt_idx] + '_' +config_param['data_param']['mov_type'][typ_idx] + '_' + config_param['data_param']['set_num'][ set_idx] + '.txt'
            complete_emg_path = config_param['filepath']['project_path'] + config_param['filepath']['data_path'] + current_emg_file_name

            print(complete_emg_path)

            # ! Check if the file exists
            if Path(complete_emg_path).is_file():

                # ! Loading and epoching for training
                EMG_Data = EMGData(format="ANTmini", filenames=[current_emg_file_name],
                                   data_path=config_param['filepath']['project_path'] + config_param['filepath'][
                                       'data_path'], f_samp=config_param['preprocess_param']['f_samp'],
                                   channel_names=config_param['preprocess_param']['channel_names_emg'])

                # ! Plotting the raw EMG data
                if config_param['plot_param']['is_plot_raw']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                                     unit="uV",
                                     title="Raw EMG plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in uV",
                                     is_grid_on=True)

                # ! Loading the target values for the 3 joints
                print("Creating Quali Elbow object!!")
                Quali_Data_Elbow = EEGData(format="NumpyQualisys",
                                           filenames=[config_param['filepath']['quali_path'][0] +
                                                      config_param['data_param']['weights'][wgt_idx] + '_' +
                                                      config_param['data_param']['mov_type'][typ_idx] + '_' +
                                                      config_param['data_param']['set_num'][set_idx]],
                                           data_path=config_param['filepath']['project_path'] +
                                                     config_param['filepath']['data_path'],
                                           f_samp=config_param['preprocess_param']['f_samp'],
                                           channel_names=config_param['preprocess_param']['channel_names_quali'],
                                           file_type='individual',
                                           add_marker_channel=True)

                print("Creating Quali Shoulder Front object!!")
                Quali_Data_Front = EEGData(format="NumpyQualisys",
                                           filenames=[config_param['filepath']['quali_path'][1] +
                                                      config_param['data_param']['weights'][wgt_idx] + '_' +
                                                      config_param['data_param']['mov_type'][typ_idx] + '_' +
                                                      config_param['data_param']['set_num'][set_idx]],
                                           data_path=config_param['filepath']['project_path'] +
                                                     config_param['filepath']['data_path'],
                                           f_samp=config_param['preprocess_param']['f_samp'],
                                           channel_names=config_param['preprocess_param']['channel_names_quali'],
                                           file_type='individual',
                                           add_marker_channel=True)

                print("Creating Quali Shoulder Side object!!")
                Quali_Data_Side = EEGData(format="NumpyQualisys",
                                          filenames=[config_param['filepath']['quali_path'][2] +
                                                     config_param['data_param']['weights'][wgt_idx] + '_' +
                                                     config_param['data_param']['mov_type'][typ_idx] + '_' +
                                                     config_param['data_param']['set_num'][set_idx]],
                                          data_path=config_param['filepath']['project_path'] + config_param['filepath'][
                                              'data_path'],
                                          f_samp=config_param['preprocess_param']['f_samp'],
                                          channel_names=config_param['preprocess_param']['channel_names_quali'],
                                          file_type='individual',
                                          add_marker_channel=True)

                # ! ************************************************
                # ! Data Pre-processing
                # ! ************************************************

                # ? High pass filter
                # * design the highpass filter
                sos_hp = EMG_Data.designFilter(f_high=config_param['preprocess_param']['f_cutoff_hpf'],
                                               f_low=config_param['preprocess_param']['f_cutoff_lpf'],
                                               order=config_param['preprocess_param']['filter_order'],
                                               filter_type="scipy_butter",
                                               return_type="sos")
                # * apply hpf filter
                EMG_Data.filterData_offline(filter_method=config_param['preprocess_param']['filter_method'],
                                            sos=sos_hp)
                # ? Plotting HP filtered data
                if config_param['plot_param']['is_plot_hpf']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                                     unit="uV",
                                     title="High-Pass Filtered plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in uV",
                                     is_grid_on=True)

                # ? Apply Variance Filter from variance_tools_api
                print("Applying Variance filter ...")
                width = config_param['preprocess_param']['var_filter_width']
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

                if config_param['plot_param']['is_plot_var_filter']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                                     unit="uV",
                                     title="Variance Filtered EMG plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in uV",
                                     is_grid_on=True)

                # ? Normalisation
                print("Performing Normalization with Max Voluntary Contraction ...")
                EMG_Data.normalizeContinuousData(mvc=config_param['preprocess_param']['mvc'])
                print("Normalization with Max Voluntary Contraction performed !!\n")

                # ? Low pass filter to smoothen the signal
                # * design the lowpass filter
                sos_lp = EMG_Data.designFilter(f_low=config_param['preprocess_param']['f_cutoff_sm_lpf'],
                                               order=config_param['preprocess_param']['sm_filter_order'],
                                               filter_type="scipy_butter",
                                               return_type="sos")
                # * apply lpf filter
                EMG_Data.filterData_offline(filter_method=config_param['preprocess_param']['filter_method'],
                                            sos=sos_lp)

                # ? Plot normalised and smoothened data
                if config_param['plot_param']['is_plot_smoothed']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                                     unit="V",
                                     title="Normalised and Smoothed EMG plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in V",
                                     is_grid_on=True)

                if config_param['preprocess_param']['neural_activation_fcn']:
                    # ? Calculate Neural Activation Force
                    print("Replacing sample with its force activation value ...")
                    EMG_Data.calculateActivationForceFunction(d=config_param['preprocess_param']['act_delay'],
                                                              b1=config_param['preprocess_param']['act_beta1'],
                                                              b2=config_param['preprocess_param']['act_beta2'],
                                                              g=config_param['preprocess_param']['act_gamma'],
                                                              nonlinear_shape_factor=config_param['preprocess_param']['act_A'])
                    print("Replaced each sample with its force activation value !!\n")

                # ? Plot force activation data
                if config_param['plot_param']['is_plot_act']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4, :],
                                     unit="V",
                                     title="Force Activated EMG plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in V",
                                     is_grid_on=True)

                # ! Windowing the filtered data
                _ = EMG_Data.windowContinuousData(startmarkernumber=1,
                                                  stopmarkernumber=1,
                                                  window_size=config_param['preprocess_param']['window_size_x'],
                                                  window_step=config_param['preprocess_param']['window_step'],
                                                  start_index_offset=0,
                                                  start_channel_pick=0,
                                                  end_channel_pick=8,
                                                  return_window_end_indices=True)

                _ = Quali_Data_Elbow.windowContinuousData(startmarkernumber=1,
                                                          stopmarkernumber=1,
                                                          window_size=config_param['preprocess_param']['window_size_e'],
                                                          window_step=config_param['preprocess_param']['window_step'],
                                                          start_index_offset=0,
                                                          start_channel_pick=0,
                                                          end_channel_pick=3,
                                                          return_window_end_indices=True)

                _ = Quali_Data_Front.windowContinuousData(startmarkernumber=1,
                                                          stopmarkernumber=1,
                                                          window_size=config_param['preprocess_param']['window_size_s'],
                                                          window_step=config_param['preprocess_param']['window_step'],
                                                          start_index_offset=0,
                                                          start_channel_pick=0,
                                                          end_channel_pick=3,
                                                          return_window_end_indices=True)

                _ = Quali_Data_Side.windowContinuousData(startmarkernumber=1,
                                                         stopmarkernumber=1,
                                                         window_size=config_param['preprocess_param']['window_size_f'],
                                                         window_step=config_param['preprocess_param']['window_step'],
                                                         start_index_offset=0,
                                                         start_channel_pick=0,
                                                         end_channel_pick=3,
                                                         return_window_end_indices=True)
                # use the EMG_Data.windows if you want to access the windowed data

                # ! Plot specific filtered windows for debugging
                if config_param['plot_param']['is_plot_filt_win']:
                    EMG_Data.plotEMG(data=EMG_Data.getWindows()[0, 2, :, 18],  # [trl,chn,smpl,wnd]
                                     n_samples=EMG_Data.getWindows().shape[2],
                                     unit="V",
                                     title="Force Activated EMG plot for Channel 5",
                                     xlabel="Time in s",
                                     ylabel="Voltage in V",
                                     is_grid_on=True)

                # **********************************************************************************
                # ******************************* Feature Extraction *******************************
                # **********************************************************************************
                # Fenster extrahieren
                windows = EMG_Data.getWindows()[0]  # (n_channels, n_samples, n_windows)

                # Umformen zu (n_windows, n_samples, n_channels)
                windows = np.transpose(windows, (2, 1, 0))  # (n_windows, n_samples, n_channels)
                #----------------------------------------------------------------------------------
                # Frequenzfeatures berechnen
                freq_features = extract_freq_features(
                    windows,
                    fs=config_param['preprocess_param']['f_samp']
                )
                #---------------------------------------------------------------------------------------
                print("Extracting features from windowed data ...")
                EMG_Data.featureExtractionFromWindows(feature_type="timepoints",
                                                      feature_indices_windows=feature_indices_windows_x)
                Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints",
                                                              feature_indices_windows=feature_indices_windows_e)
                Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints",
                                                              feature_indices_windows=feature_indices_windows_s)
                Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints",
                                                             feature_indices_windows=feature_indices_windows_f)
                print("Feature extraction from windowed data completed !!\n")

                # Labels extrahieren (wie gehabt)
                y_e = Quali_Data_Elbow.getFeatures()[:, 0]  # (n_windows,)
                y_f = Quali_Data_Front.getFeatures()[:, 0]
                y_s = Quali_Data_Side.getFeatures()[:, 0]

                # Sicherstellen, dass alles gleich lang ist
                min_len = min(windows.shape[0], y_e.shape[0], y_f.shape[0], y_s.shape[0])
                windows = windows[:min_len]
                y_e = y_e[:min_len]
                y_f = y_f[:min_len]
                y_s = y_s[:min_len]
                #-------------------------
                freq_features = freq_features[:min_len]
                #-------------------------

                # Split-Index berechnen
                split_idx = int(round(config_param['model_param']['train_test_split'] * min_len))

                # Split durchführen und an die kombinierten Arrays anhängen
                window_train_combined = np.concatenate((window_train_combined, windows[:split_idx]), axis=0)
                window_test_combined = np.concatenate((window_test_combined, windows[split_idx:]), axis=0)

                y_e_train_combined = np.concatenate((y_e_train_combined, y_e[:split_idx]))
                y_e_test_combined = np.concatenate((y_e_test_combined, y_e[split_idx:]))

                y_f_train_combined = np.concatenate((y_f_train_combined, y_f[:split_idx]))
                y_f_test_combined = np.concatenate((y_f_test_combined, y_f[split_idx:]))

                y_s_train_combined = np.concatenate((y_s_train_combined, y_s[:split_idx]))
                y_s_test_combined = np.concatenate((y_s_test_combined, y_s[split_idx:]))

                if freq_train_combined is None:
                    # Erstes Mal: Arrays initialisieren mit richtiger Spaltenzahl
                    freq_train_combined = np.empty((0, freq_features.shape[1]))
                    freq_test_combined = np.empty((0, freq_features.shape[1]))

                # Jetzt normal anhängen
                freq_train_combined = np.concatenate((freq_train_combined, freq_features[:split_idx]), axis=0)
                freq_test_combined = np.concatenate((freq_test_combined, freq_features[split_idx:]), axis=0)


            else:
                continue
# **********************************************************************************
# *************************** Train, load or test Model ****************************
# **********************************************************************************

# Wichtige Standardisierung der Daten für das Modelltraining
from sklearn.preprocessing import StandardScaler

SCALER_FILE_TIME = MODEL_DIR / "std_scaler_time.pkl"
SCALER_FILE_FREQ = MODEL_DIR / "std_scaler_freq.pkl"

if RUN_MODE == "train":
    # --- Zeit-Features ---
    scaler_time = StandardScaler()
    window_train_combined = scaler_time.fit_transform(
        window_train_combined.reshape(-1, window_train_combined.shape[-1])
    ).reshape(window_train_combined.shape)

    window_test_combined = scaler_time.transform(
        window_test_combined.reshape(-1, window_test_combined.shape[-1])
    ).reshape(window_test_combined.shape)

    joblib.dump(scaler_time, SCALER_FILE_TIME)

    # --- Frequenz-Features ---
    scaler_freq = StandardScaler()
    freq_train_combined = scaler_freq.fit_transform(freq_train_combined)
    freq_test_combined = scaler_freq.transform(freq_test_combined)
    joblib.dump(scaler_freq, SCALER_FILE_FREQ)

else:  # infer
    # Zeit-Features
    scaler_time = joblib.load(SCALER_FILE_TIME)
    window_test_combined = scaler_time.transform(
        window_test_combined.reshape(-1, window_test_combined.shape[-1])
    ).reshape(window_test_combined.shape)

    # Frequenz-Features
    scaler_freq = joblib.load(SCALER_FILE_FREQ)
    freq_test_combined = scaler_freq.transform(freq_test_combined)

# Vor Modellerstellung
tf.config.optimizer.set_jit(True)

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ------------------------------------------------------------------
# Modell: Dilated Temporal Convolutional Network (Multi-Task)
#     – feste Hyperparameter nach Tuner-Ergebnis (filters 32, stacks 2, dropout 0.02)
# ------------------------------------------------------------------
def build_tcn_mtl(input_shape,
                  filters=32,  # optimaler Wert
                  stacks=2,  # optimaler Wert
                  dropout_rate=0.02):  # optimaler Wert
    inp = Input(shape=input_shape, name='emg_input')
    x = inp

    for s in range(stacks):
        dilation = 2 ** s  # 1, 2
        # -------- Residual Branch --------
        y = layers.Conv1D(filters, 3, padding='causal',
                          dilation_rate=dilation,
                          activation='relu')(x)
        y = layers.Conv1D(filters, 3, padding='causal',
                          dilation_rate=dilation,
                          activation='relu')(y)

        # -------- Shortcut Branch --------
        if x.shape[-1] != filters:  # Kanal-Match
            x = layers.Conv1D(filters, 1, padding='same')(x)

        x = layers.add([x, y])  # Residual-Add
        x = layers.Activation('relu')(x)

    # -------- Output-Head --------
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    out_e = layers.Dense(1, name='torque_elbow')(x)
    out_f = layers.Dense(1, name='torque_shoulder_front')(x)
    out_s = layers.Dense(1, name='torque_shoulder_side')(x)

    return models.Model(inp, [out_e, out_f, out_s], name='TCN_MTL')


def build_model(input_shape_time, input_shape_freq,
                mode="time+freq", filters=32, stacks=2, dropout_rate=0.2):

    inputs = []
    branches = []

    if mode in ["time", "time+freq"]:
        # Zeit-Pfad (TCN)
        inp_time = Input(shape=input_shape_time, name='emg_input')
        x = inp_time
        for s in range(stacks):
            dilation = 2 ** s
            y = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation)(x)
            y = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation)(y)
            if x.shape[-1] != filters:
                x = layers.Conv1D(filters, 1, padding='same')(x)
            x = layers.add([x, y])
        x = layers.GlobalAveragePooling1D()(x)
        inputs.append(inp_time)
        branches.append(x)

    if mode in ["freq", "time+freq"]:
        # Frequenz-Pfad (Dense)
        inp_freq = Input(shape=input_shape_freq, name='freq_input')
        f = layers.Dense(64, activation="relu")(inp_freq)
        f = layers.Dropout(dropout_rate)(f)
        inputs.append(inp_freq)
        branches.append(f)

    # Fusion (falls mehrere Zweige)
    if len(branches) > 1:
        combined = layers.concatenate(branches)
    else:
        combined = branches[0]

    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dropout(dropout_rate)(combined)

    out_e = layers.Dense(1, name='torque_elbow')(combined)
    out_f = layers.Dense(1, name='torque_shoulder_front')(combined)
    out_s = layers.Dense(1, name='torque_shoulder_side')(combined)

    safe_mode = mode.replace("+", "_")
    return models.Model(inputs, [out_e, out_f, out_s], name=f"MTL_{safe_mode}")


# ------------------------------------------------------------------
# Kompilieren & Trainieren
# ------------------------------------------------------------------
# Shapes
input_shape_time = (window_train_combined.shape[1], window_train_combined.shape[2])
input_shape_freq = (freq_train_combined.shape[1],)

if RUN_MODE == "train":
    model_mtl = build_model(input_shape_time, input_shape_freq, mode=INPUT_MODE)

    # Compile
    model_mtl.compile(optimizer=config_param['model_param']['optimizer'],
                      loss=config_param['model_param']['loss_fcn'])

    # Trainingsdaten auswählen
    if INPUT_MODE == "time":
        X_train, X_test = window_train_combined, window_test_combined
    elif INPUT_MODE == "freq":
        X_train, X_test = freq_train_combined, freq_test_combined
    else:  # "time+freq"
        X_train, X_test = [window_train_combined, freq_train_combined], [window_test_combined, freq_test_combined]

    # Fit
    history = model_mtl.fit(
        X_train,
        {
            'torque_elbow': y_e_train_combined,
            'torque_shoulder_front': y_f_train_combined,
            'torque_shoulder_side': y_s_train_combined
        },
        epochs=config_param['model_param']['n_epochs'],
        batch_size=config_param['model_param']['batch_size'],
        validation_split=config_param['model_param']['validation_split'],
        callbacks=[early_callback]
    )

    model_mtl.save(MODEL_DIR / "tcn_mtl.keras")

else:  # RUN_MODE == "infer"
    model_mtl = load_model(MODEL_DIR / "tcn_mtl.keras", compile=False)

# ------------------------------------------------------------------------------
# Vorhersagen erzeugen
# ------------------------------------------------------------------------------

# Prediction
# --- Vorhersagen erzeugen ---
if INPUT_MODE == "time":
    X_test = window_test_combined
elif INPUT_MODE == "freq":
    X_test = freq_test_combined
else:  # "time+freq"
    X_test = [window_test_combined, freq_test_combined]

predictions_e, predictions_f, predictions_s = model_mtl.predict(X_test)
predictions_e = predictions_e.flatten()
predictions_f = predictions_f.flatten()
predictions_s = predictions_s.flatten()

# ------------------------------------------------------------------------------
# Post-Processing
# ------------------------------------------------------------------------------

window_length = config_param['post_processing_param']['savitzky_window_length']
polyorder = config_param['post_processing_param']['savitzky_polyorder']

if config_param['post_processing_param']['savitzky_on']:
    from scipy.signal import savgol_filter

    predictions_e = savgol_filter(predictions_e, window_length, polyorder)
    predictions_f = savgol_filter(predictions_f, window_length, polyorder)
    predictions_s = savgol_filter(predictions_s, window_length, polyorder)


def moving_average(x, w):
    import numpy as np
    return np.convolve(x, np.ones(w), 'same') / w


filter_window_size = config_param['post_processing_param']['moving_av_window_size']
if config_param['post_processing_param']['moving_av_on']:
    predictions_e = moving_average(predictions_e, filter_window_size)
    predictions_f = moving_average(predictions_f, filter_window_size)
    predictions_s = moving_average(predictions_s, filter_window_size)

# ------------------------------------------------------------------------------
# RMSE-Berechnung (Test + Train) und Ausgabe
# ------------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error
import numpy as np

# Test-RMSE
rmse_e = np.sqrt(mean_squared_error(y_e_test_combined, predictions_e))
rmse_f = np.sqrt(mean_squared_error(y_f_test_combined, predictions_f))
rmse_s = np.sqrt(mean_squared_error(y_s_test_combined, predictions_s))

# Train-RMSE
# --- Train-RMSE ---
if INPUT_MODE == "time":
    X_train = window_train_combined
elif INPUT_MODE == "freq":
    X_train = freq_train_combined
else:  # "time+freq"
    X_train = [window_train_combined, freq_train_combined]

train_pred_e, train_pred_f, train_pred_s = model_mtl.predict(X_train)

rmse_e_train = np.sqrt(mean_squared_error(y_e_train_combined, train_pred_e.flatten()))
rmse_f_train = np.sqrt(mean_squared_error(y_f_train_combined, train_pred_f.flatten()))
rmse_s_train = np.sqrt(mean_squared_error(y_s_train_combined, train_pred_s.flatten()))

print('Ergebnisse (Multi-Task):')
print(f"Ellbogen      -> Test-RMSE: {rmse_e:.2f}  | Train-RMSE: {rmse_e_train:.2f}")
print(f"Schulter Front-> Test-RMSE: {rmse_f:.2f}  | Train-RMSE: {rmse_f_train:.2f}")
print(f"Schulter Side -> Test-RMSE: {rmse_s:.2f}  | Train-RMSE: {rmse_s_train:.2f}")

# --- Zeitachsen der Plots anpassen (bisher über window_step_size gelöst)  ------------------------------------------
fs = config_param['preprocess_param']['f_samp']  # Abtastfreq. [Hz]
step = config_param['preprocess_param']['window_step']  # Fenster-Schritt [Samples]
time_ax_e = np.arange(len(predictions_e)) * step / fs * 5  # Zeit [s]
time_ax_f = np.arange(len(predictions_f)) * step / fs * 5  # Zeit [s]
time_ax_s = np.arange(len(predictions_s)) * step / fs * 5  # Zeit [s]

# ------------------------------------------------------------------------------
# 6) Visualisierung
# ------------------------------------------------------------------------------

plotResults(y_e_test_combined, "real torque",
            predictions_e, "predicted torque",
            f"Elbow Joint Filtered; RMSE: {rmse_e:.4f} N-m",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(y_f_test_combined, "real torque",
            predictions_f, "predicted torque",
            f"Shoulder Front Joint Filtered; RMSE: {rmse_f:.4f} N-m",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(y_s_test_combined, "real torque",
            predictions_s, "predicted torque",
            f"Shoulder Side Joint Filtered; RMSE: {rmse_s:.4f} N-m",
            ylabel="Torque in N-m",
            is_grid_on=True)

plt.show()
