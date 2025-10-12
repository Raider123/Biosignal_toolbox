import warnings

import zmq
import time
import os
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, savgol_filter, medfilt
from biosignal_toolbox.emg_lib import OnlineEMG
from tensorflow.keras.models import load_model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

class LiveEstimation:

   def __init__(self):
       self.property = {}
       self.configure_properties()
       print("Loaded properties!") # ToDo Replace properties with the data in the config_file!

       config_filename = 'pipeline_mlp_to_cnn.yaml'
       self.cfg = loadConfig(filename=config_filename)
       print('Loaded the config file!')

       # EMG 8 channel names
       self.channel_names = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

       # Initialise HPF sos
       self.sos_hpf = butter(N=2, Wn=self.property["f_cutoff_hpf"], btype='highpass', analog=False, output='sos',
                             fs=self.property["f_samp"])
       self.sos_hpf_idx = 0

       # Initialise LPF sos
       self.sos_lpf = butter(N=2, Wn=self.property["f_cutoff_lpf"], btype='lowpass', analog=False, output='sos',
                             fs=self.property["f_samp"])
       self.sos_lpf_idx = 0

       # Create old_online EMG object
       self.EMG_live = OnlineEMG(stream_type="data",
                                 channel_names=self.channel_names, n_samples=self.property["buffer_size"],
                                 f_samp=self.property["f_samp"])

       print("Created EMG_live!!")

       self.EMG_live_freq = None
       self.features = None
       self.predictions = None

       # Variables for Visualization
       self.all_predictions = []

       self.fig = None
       self.axes = None
       self.lines = []
       self.current_time = 0
       self.elapsed_times = []  # speichert die Zeitachse

       self.setup_plot()

       #  TCN Model
       self.load_model(
           'F:/SMT_MASTERPROJEKT/biosignal_toolbox/src/JTE_Project/offline/saved_online_models/tcn_mtl.keras')

       print("Loaded TCN Model")

       emg_context = zmq.Context()
       self.emg_socket = emg_context.socket(zmq.SUB)
       self.emg_socket.connect("tcp://127.0.0.1:5555")
       print("EMG Subscriber is active")
       self.emg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

       print("Finished configuring EMG receiver")

       self.update_loop()

   def setup_plot(self):
       """Erstellt die Live-Visualisierung mit 3 Subplots."""
       print("🎨 Setup der Visualisierung...\n")

       self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
       self.fig.suptitle('Joint-Torque-Estimation (Realtime)', fontsize=14, fontweight='bold')

       joint_names = ['Elbow', 'Shoulder Front', 'Shoulder Side']
       colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

       for i, (ax, name, color) in enumerate(zip(self.axes, joint_names, colors)):
           ax.set_ylabel(f'{name} Torque', fontsize=10)
           ax.set_xlabel('Zeit (s)', fontsize=10)
           ax.grid(True, alpha=0.3)
           ax.set_xlim(0, 10)
           ax.set_ylim(-5, 5)
           line, = ax.plot([], [], color=color, linewidth=1.5, label=name)
           self.lines.append(line)
           ax.legend(loc='upper right')

       plt.tight_layout()

   def update_plot(self, elapsed_times):
       if not self.all_predictions or not elapsed_times:
           return

       # In (N, C) bringen – jedes Element flach machen und stapeln
       preds = [np.asarray(p).reshape(-1) for p in self.all_predictions]
       all_preds = np.vstack(preds)
       if all_preds.ndim == 1:
           all_preds = all_preds.reshape(-1, 1)

       times = np.asarray(elapsed_times, dtype=float)

       # Auf gleiche Länge kürzen
       n = min(len(times), all_preds.shape[0])
       if n == 0:
           return
       times, all_preds = times[:n], all_preds[:n, :]

       # Linien aktualisieren (nur so viele wie Spalten)
       cols = all_preds.shape[1]
       for i, line in enumerate(self.lines[:cols]):
           y = all_preds[:, i]
           if y.size == 0:
               line.set_data([], [])
               continue

           line.set_data(times, y)

           y_min, y_max = y.min(), y.max()
           margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
           self.axes[i].set_ylim(y_min - margin, y_max + margin)

           t_last = times[-1]
           self.axes[i].set_xlim(max(0.0, t_last - 10.0), max(10.0, t_last + 1.0))

       # Überzählige Linien leeren
       for j in range(cols, len(self.lines)):
           self.lines[j].set_data([], [])

       self.fig.canvas.draw_idle()
       self.fig.canvas.flush_events()

   def add_property(self, name, default_value):
       self.property[name] = default_value

   def configure_properties(self):
       self.add_property("buffer_size", 500)
       self.add_property("feature_size", 20)
       self.add_property("f_samp", 500)
       self.add_property("n_channels", 8)
       self.add_property("f_cutoff_hpf", 15)
       self.add_property("f_cutoff_lpf", 10)
       self.add_property("var_filter_width", 20)
       self.add_property("mvc", 2.7579163508176626e-06)
       self.add_property("delay", 25)
       self.add_property("beta1", 0.25)
       self.add_property("beta2", 0.05)
       self.add_property("gamma", 0.7)
       self.add_property("A", -1.5)

   def load_model(self, model_path=None):
       """
       Load the trained TCN model.

       Parameters
       ----------
       model_path : str or Path, optional
           Path to the saved model. If None, uses default path from config.
       """
       if model_path is None:
           save_model_path = (
                   Path(__file__).parent.parent / "saved_online_models"
           )
           model_path = save_model_path / "tcn_mtl.keras"

       print(f"Loading model from: {model_path}")
       self.model = load_model(model_path, compile=False)
       print("Model loaded successfully!\n")

       return True

   def start_device_stream(self):
        # Start EMG live stream (unused in Pseudoonline)
        self.EMG_live.startANTEegoStreaming(path_to_so_file=self.property["path_to_so_file"])

        return True

   def extract_features(self):
       """Extract all features from windowed EMG data."""
       print("Extracting features from windowed data...")

       # Timepoints feature extraction
       window_size_ms = (
               self.cfg.preprocess_param.window_size_x * 1000 / self.property["f_samp"]
       )
       feature_indices_windows_x = np.array([0, window_size_ms])

       self.EMG_live.featureExtractionFromWindows(
           feature_type="timepoints",
           feature_indices_windows=feature_indices_windows_x
       )

       # Time domain features
       n_channels = len(self.channel_names)

       # RMS (Root Mean Square)
       rms_feature = self.EMG_live.getRMSFeatures_windows(n_channels=n_channels)
       self.EMG_live.addFeatures(rms_feature)

       # Waveform Length
       wfl_feature = self.EMG_live.getWaveformLengthFeatures_windows(
           n_channels=n_channels
       )
       self.EMG_live.addFeatures(wfl_feature)

       # Slope Sign Change
       ssc_feature = self.EMG_live.getSlopeSignChangeFeatures_windows(
           n_channels=n_channels,
           threshold=0.02
       )
       self.EMG_live.addFeatures(ssc_feature)

       # Frequency domain features
       self.EMG_live_freq.featureExtractionFromWindows(
           feature_type="freqBandPower",
           psd_method="multitaper",
           freq_bands=[15, 50, 100, 150, 200, 245]
       )
       fbp_feature = self.EMG_live_freq.getFeatures()
       self.EMG_live.addFeatures(fbp_feature)

       # Time-frequency domain features (Morlet Wavelet)
       freqs = np.arange(start=50, stop=226, step=25)
       n_cycles = np.ones(len(freqs)) * 5
       n_cycles[0] = 3
       n_cycles[1] = 4

       mwc_feature = self.EMG_live_freq.getMorletWaveletCoeffFeatures_windows(
           freqs=freqs,
           n_cycles=n_cycles
       )
       self.EMG_live.addFeatures(mwc_feature)

       # Differential features (change between consecutive windows)
       peak_detection = np.diff(
           self.EMG_live.getFeatures(),
           axis=0,
           prepend=self.EMG_live.getFeatures()[0:1, :]
       )
       self.EMG_live.addFeatures(peak_detection)

       self.features = self.EMG_live.getFeatures()
       print(f"Total EMG features extracted: {self.features.shape}")
       print("Feature extraction completed!\n")

   def scale_features(self):
       """Scale features using StandardScaler and apply PCA."""
       print("Scaling features...")

       # Pre-PCA scaling
       scaler, scaled_features, _, _ = self.EMG_live.scaleFeatures_windows(
           train_data=self.features,
           test_data=self.features,
           val_data=self.features,
           method="StandardScaler"
       )

       self.features = scaled_features

       '''
       # Dimensionality reduction with PCA
       self.features = self.emg_data.reduceDimensions_windows(
           train_data=scaled_features,
           test_data=scaled_features,
           val_data=scaled_features,
           method="PCA",
           n_components=0.99
       )

       # Extract Array from Tuple
       self.features = self.features[0]
       '''
       print(f"Features after PCA: {self.features.shape}")
       print("Feature scaling completed!\n")

   def predict(self):
       """
       Process EMG file and predict joint torques.

       Parameters
       ----------
       data_array : Preloaded EMG Data (cf. EMGData class in emg_lib.py)

       Returns
       -------
       np.ndarray
           Predicted torques for [elbow, shoulder_front, shoulder_side]
       """
       # create local copy
       features = self.features

       # Reshape for CNN/TCN input: (samples, features, channels)
       n_features = features.shape[1]
       features_cnn = features.reshape((-1, n_features, 1))

       print(f"Input shape for model: {features_cnn.shape}")

       # Predict
       print("Running model prediction...")
       preds = self.model.predict(features_cnn)

       # Concatenate multi-task outputs: [elbow, front, side]
       self.predictions = np.concatenate(
           [preds[0], preds[1], preds[2]], axis=1
       )

       print(f"Predictions shape: {self.predictions.shape}")
       print("Model prediction completed!\n")

       return self.predictions

   def apply_post_filter(self):
       """
       Apply post-prediction filtering (median and/or Savitzky-Golay).

       Returns
       -------
       np.ndarray
           Filtered predictions
       """
       if self.predictions is None:
           warnings.warn("No predictions available. Run predict() first!")
           return None

       print("Applying post-prediction filters...")
       filtered_predictions = self.predictions.copy()

       # Median filter
       if self.cfg.post_train_param.filter_type == 'median':
           print(f"Applying median filter (kernel size: "
                 f"{self.cfg.post_train_param.filter_size})...")
           for i in range(3):
               filtered_predictions[:, i] = medfilt(
                   filtered_predictions[:, i],
                   kernel_size=self.cfg.post_train_param.filter_size
               )

       # Savitzky-Golay filter
       if getattr(self.cfg.post_train_param, 'savgol_window_len', None):
           print(f"Applying Savitzky-Golay filter (window: "
                 f"{self.cfg.post_train_param.savgol_window_len}, "
                 f"poly order: {self.cfg.post_train_param.savgol_poly_order})...")
           for i in range(3):
               filtered_predictions[:, i] = savgol_filter(
                   filtered_predictions[:, i],
                   self.cfg.post_train_param.savgol_window_len,
                   self.cfg.post_train_param.savgol_poly_order
               )

       self.predictions = filtered_predictions
       print("Post-prediction filtering completed!\n")

       return self.predictions

   def update_loop(self):
       while True:
           update_start_time = time.time()

           # Read emg data from the stream
           data_arr_str = self.emg_socket.recv_string()
           data_arr_np = np.array(data_arr_str)

           # Manually setting the data (otherwise use start_hook())
           self.EMG_live.setChunk(data_arr_np, chunk_type="numpy")

           # Update the internal ring buffer
           self.EMG_live.updateBuffer(num_channels = 8)

           # Temp - get data from the buffer
           #latest_data = self.EMG_live.getDataBuffer()
           #print("Buffer-Shape: ", latest_data.shape)

           self.EMG_live.ensureLoopFrequency(print_loop_time=False)

           # # high pass filter
           self.EMG_live.highPassFilter(cutoff_freq=self.property["f_cutoff_hpf"], order=2, fs=self.property["f_samp"], filter_type="butter", sos=self.sos_hpf, counter=self.sos_hpf_idx, mode='old_offline')
           self.sos_hpf_idx = 1

           # Create identical copy for frequency extraction
           self.EMG_live_freq = copy.deepcopy(self.EMG_live)

           # # variance filter
           self.EMG_live.applyVarianceFilter_data(mode = "old_online", ring_buffer=np.zeros(self.property['var_filter_width']), width=self.property['var_filter_width'], index=0)

           # # normalisation
           self.EMG_live.normalizeContinuousData(mvc=self.property["mvc"], mode = "old_online")

           # # low pass filter
           self.EMG_live.lowPassFilter(mode='old_offline', cutoff_freq=self.property["f_cutoff_lpf"], order=2, fs=self.property["f_samp"], filter_type="butter", sos=self.sos_lpf, counter=self.sos_lpf_idx)
           self.sos_lpf_idx = 1

           # # neural activation force
           self.EMG_live.calculateActivationForceFunction(mode='old_offline', d=self.property["delay"], b1=self.property["beta1"], b2=self.property["beta2"], g=self.property["gamma"], nonlinear_shape_factor=self.property["A"])

           # convert filtered data into window
           self.EMG_live.bufferToWindows(num_non_data_channels=0)
           self.EMG_live_freq.bufferToWindows(num_non_data_channels=0)

           ''' # REPLACED WITH OFFLINE FUNCTIONS
           # Window related Filtering
           self.EMG_live.filterWindows(sos=self.sos_bp, apply_method="zero_phase_sos", padtype="even")

           self.EMG_live.windows = np.abs(self.EMG_live.windows)
           self.EMG_live.filterWindows(sos=self.sos_lp, apply_method="zero_phase_sos", padtype="even")

           # get data buffer size to calculate feature window indices
           db = self.EMG_live.getDataBuffer()
           feature_indices_windows = np.arange(db.shape[2] - self.property["feature_size"], db.shape[2], step=1)

           # extract features from window
           self.EMG_live.featureExtractionFromWindows(feature_type="timepoints",
                                                      feature_indices_windows=feature_indices_windows)

           inp_emg = self.EMG_live.getFeatures()


           # calculate mean absolute value (MAV)
           inp_emg = self.EMG_live.calculateMAVFromFeatures(len(self.channel_names))

           print(inp_emg.shape)
           '''

           # Use the feature extraction/scaling/prediction/visualization from the offline case
           self.extract_features()
           self.scale_features() # ToDO PCA
           self.predict()
           #self.apply_post_filter() # ToDO Adjust Savgol Size

           # Save all single predictions
           self.all_predictions.append(self.predictions)

           # Plot
           update_time_step = time.time() - update_start_time
           self.current_time = self.current_time + update_time_step
           self.elapsed_times.append(self.current_time)

           self.update_plot(elapsed_times=self.elapsed_times)
           plt.pause(0.001)


if __name__ == "__main__":

 live_estimation_obj = LiveEstimation()
