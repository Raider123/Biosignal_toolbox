import warnings

import zmq
import time
import os
import copy
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, savgol_filter, medfilt
from biosignal_toolbox.emg_lib import OnlineEMG
from tensorflow.keras.models import load_model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

class LiveEstimation:

   def __init__(self):
       emg_context = zmq.Context()
       self.emg_socket = emg_context.socket(zmq.SUB)
       self.emg_socket.connect("tcp://127.0.0.1:5555")
       print("EMG Subscriber is active")
       self.emg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

       # Parameters for receiving batches (instead of single strings)
       self.batch_size = 50
       self.emg_buffer = []
       self.emg_array = None

       self.property = {}
       self.configure_properties()
       print("Loaded properties!") # ToDo Replace properties with the data in the config_file!

       config_filename = 'pipeline_mlp_to_cnn.yaml'
       self.cfg = loadConfig(filename=config_filename)
       print('Loaded the config file!')

       # pre-calculated channelwise mvc
       self.channelwise_mvc = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/channelwise_mvc.npy"))
       print("Loaded channelwise mvc file")

       #  TCN Model
       self.load_model(
           getAbsolutePath('src/JTE_Project/offline/saved_online_models/tcn_mtl_2.keras'))

       # Scaler File
       self.emg_scaler_file = joblib.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/emg_scaler.pkl"))
       print("Loaded EMG Scaler File")

       # Load Torque Values (ground truth, only in prediction plot)
       Y_e = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/e.npy"))
       Y_f = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/front.npy"))
       Y_s = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/side.npy"))
       Y_ref_raw = np.stack((Y_e, Y_f, Y_s), axis=1)

       y_ref_length = int (Y_ref_raw.shape[0] / self.batch_size)
       self.Y_ref = self.downsample_mean_bins(Y_ref_raw, y_ref_length)

       #print("Shape Yref ", Y_ref.shape)
       #print("Shape Yref downsampled ", Y_ref_ds.shape)
       #plt.plot(Y_ref_ds[:,0])
       #plt.show()

       # EMG 8 channel names
       self.channel_names = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']


       # Create old_online EMG object
       self.EMG_live = OnlineEMG(stream_type="data",
                                 channel_names=self.channel_names, buffer_size=self.property["buffer_size"],
                                 f_samp=self.property["f_samp"])
       print("Created EMG_live!!")

       self.var_buffer = np.zeros((1, len(self.channel_names), self.property["buffer_size"], 1))

       # Design bandpass filter
       self.sos_bp = self.EMG_live.designFilter(f_high=self.cfg.preprocess_param.f_cutoff_hpf,
                                                f_low=self.cfg.preprocess_param.f_cutoff_lpf,
                                                order=self.cfg.preprocess_param.filter_order,
                                                filter_type="scipy_butter",
                                                return_type="sos")


       # Design lowpass filter
       self.sos_lp = self.EMG_live.designFilter(f_low=self.cfg.preprocess_param.f_cutoff_sm_lpf,
                                      order=self.cfg.preprocess_param.sm_filter_order,
                                      filter_type="scipy_butter",
                                      return_type="sos")

       self.EMG_live_freq = None
       self.features = None
       self.predictions = None

       # Variables for Visualization
       self.all_predictions = []

       # Variable for the EMG values (for debugging)
       self.all_emg_vals = []

       # Plotting Predictions
       self.fig = None
       self.axes = None
       self.lines = []
       self.current_time = 0
       self.elapsed_times = []  # speichert die Zeitachse

       self.show_prediction_plot = False

       if self.show_prediction_plot:
           self.setup_plot()

       # Plotting EMG
       self.emg_plot = False

       # Emg plot
       if self.emg_plot:
        self.emg_fig, self.emg_ax, self.emg_lines = self.setup_emg_plot()

       print("Finished configuring EMG receiver")
       self.update_loop()

   def downsample_mean_bins(self, Y_ref, target_len=1000):
       n = Y_ref.shape[0]
       edges = np.linspace(0, n, target_len + 1, dtype=int)
       Y_ds = np.vstack([Y_ref[edges[i]:edges[i + 1]].mean(axis=0) for i in range(target_len)])
       return Y_ds

   def setup_plot(self):
       self.elapsed_times = getattr(self, "elapsed_times", [])
       self.all_predictions = getattr(self, "all_predictions", [])
       self.Y_ref = getattr(self, "Y_ref", [])
       self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
       self.fig.suptitle('Joint-Torque-Estimation (Realtime)', fontsize=14, fontweight='bold')
       names = ['Elbow', 'Shoulder Front', 'Shoulder Side']
       self.lines, self.gt_lines = [], []
       for ax, n in zip(self.axes, names):
           ax.set_ylabel(f'{n} Torque')
           ax.set_xlabel('Zeit (s)')
           ax.grid(True, alpha=0.3)
           ax.set_ylim(-1, 20)
           l_pred, = ax.plot([], [], linewidth=1.8, label=f'{n} (Pred)')
           l_gt, = ax.plot([], [], linestyle='--', linewidth=1.5, label=f'{n} (GT)')
           self.lines.append(l_pred)
           self.gt_lines.append(l_gt)
           ax.legend(loc='upper right')
       plt.tight_layout()

   def update_plot(self):
       t = np.asarray(getattr(self, "elapsed_times", []), float)
       P = np.asarray([np.asarray(p).reshape(-1)[:3] for p in getattr(self, "all_predictions", [])], float) if len(
           getattr(self, "all_predictions", [])) else np.zeros((0, 3))
       R = np.asarray(getattr(self, "Y_ref", []), float)
       if R.ndim == 1: R = R.reshape(1, -1)
       if t.size == 0: t = np.array([0.0])
       if P.size == 0: P = np.zeros((1, 3))
       if R.size == 0: R = np.zeros((1, 3))
       k = min(t.shape[0], P.shape[0], R.shape[0])  # Fortschritt
       x_all, Yp_all, Yr_all = t[:k], P[:k], R[:k]  # bis jetzt
       s = max(0, k - 1000)  # nur letzte 10
       x, Yp, Yr = x_all[s:], Yp_all[s:], Yr_all[s:]
       xlims = (x[0], x[0]+10+1) if x[-1] <= x[0]+10 else (float(x[-1]-10), float(x[-1]+1))
       for i in range(3):
           self.lines[i].set_data(x, Yp[:, i])
           self.gt_lines[i].set_data(x, Yr[:, i])
           self.axes[i].set_xlim(*xlims)
       plt.pause(0.001)

   def add_property(self, name, default_value):
       self.property[name] = default_value

   def configure_properties(self):
       self.add_property("buffer_size", 500)
       self.add_property("f_samp", 500)
       self.add_property("n_channels", 8)
       self.add_property("f_cutoff_hpf", 15)
       self.add_property("f_cutoff_lpf", 5)
       self.add_property("var_filter_width", 20)
       self.add_property("mvc", 3.941259927517915e-06)
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
       #print("Extracting features from windowed data...")

       # Timepoints feature extraction
       window_size_ms = (
               self.cfg.preprocess_param.window_size_x * 1000 / self.property["f_samp"]
       )
       feature_indices_windows_x = np.array([0, window_size_ms])

       self.EMG_live.featureExtractionFromWindows(
           feature_type="timepoints",
           feature_indices_windows=feature_indices_windows_x
       )
       '''
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
       '''

       self.features = self.EMG_live.getFeatures()
       #print(f"Total EMG features extracted: {self.features.shape}")
       #print("Feature extraction completed!\n")

   def scale_features(self):
       self.features = self.EMG_live.scaleEMG_windows(self.emg_scaler_file, self.features)


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

       #print(f"Input shape for model: {features_cnn.shape}")

       # Predict
       #print("Running model prediction...")
       preds = self.model.predict(features_cnn)

       # Concatenate multi-task outputs: [elbow, front, side]
       self.predictions = np.concatenate(
           [preds[0], preds[1], preds[2]], axis=1
       )

       #print(f"Predictions shape: {self.predictions.shape}")
       #print("Model prediction completed!\n")

       return self.predictions

   def read_emg_batch(self):
       while len(self.emg_buffer) != self.batch_size:
           try:
               # Nachricht empfangen (als String)
                message = self.emg_socket.recv_string()

               # In Zahlen (float) umwandeln
                values = np.fromstring(message, sep=" ")

                self.emg_buffer.append(values)

               # Wenn 500 Samples gesammelt → NumPy-Array bilden
                if len(self.emg_buffer) == self.batch_size:
                     self.emg_array = np.stack(self.emg_buffer)  # shape: (500, n_channels)

           except Exception as e:
               print(f"⚠️ Error: {e}")

       # Buffer leeren
       self.emg_buffer = []

   def setup_emg_plot(self, n_channels=8, n_samples=500):
       plt.ion()
       fig, ax = plt.subplots(figsize=(10, 5))
       x = np.arange(n_samples)
       lines = [ax.plot(x, np.zeros(n_samples))[0] for _ in range(n_channels)]
       ax.set_xlim(0, n_samples - 1)
       ax.set_ylim(-1, 5)
       ax.grid(True)
       return fig, ax, lines

   def update_emg_plot(self, lines, arr):
       """
       arr: (1,8,500,1) oder (8,500)
       """
       data = arr[0, :, :, 0] if arr.ndim == 4 else arr  # -> (8, 500)
       for i, ln in enumerate(lines):
           ln.set_ydata(data[i])
       plt.pause(0.01)

   def save_all_predictions(self, filename="all_predictions.npy"):
       all_preds = np.concatenate(self.all_predictions, axis=0)
       np.save(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_predictions.npy"), all_preds)

   def save_torques(self):
       np.save(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_torques.npy"), self.Y_ref)

   def save_elapsed_times(self):
       elapsed_time_np = np.array(self.elapsed_times)
       np.save(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_times.npy"), elapsed_time_np)

   def save_emg_vals(self):
       all_emgs = np.concatenate(self.all_emg_vals, axis=1)
       np.save(getAbsolutePath("src/JTE_Project/offline/filter_tests/lowpass_online.npy"), all_emgs)
       print("Save EMG Shape: ", all_emgs.shape)


   def update_loop(self):
       # len(self.elapsed_times) * self.batch_size < 500: # Iterate exactly over 500 samples
       # len(self.elapsed_times) < self.Y_ref.shape[0]: # One complete runthrough
       while len(self.elapsed_times) < self.Y_ref.shape[0]: # One complete runthrough
           update_start_time = time.perf_counter()

           # Read emg data from the stream
           self.read_emg_batch()

           # Manually setting the data (otherwise use start_hook())
           self.EMG_live.setChunk(self.emg_array, chunk_type="numpy")

           # Update the internal ring buffer
           self.EMG_live.updateBuffer(num_channels = 8)

           # * apply bandpass filter (previous highpass filter)
           self.EMG_live.filterBuffer_bandPass(sos=self.sos_bp)

           # # Store previous values for the variance filter
           self.var_buffer = np.roll(self.var_buffer, shift = int(-1*self.batch_size), axis = 2)
           self.var_buffer[0, :, int(-1*self.batch_size):, 0] = self.EMG_live.getDataBuffer()[0, :, int(-1*self.batch_size):, 0]

           # Create identical copy for frequency extraction
           self.EMG_live_freq = copy.deepcopy(self.EMG_live)
           
           # # variance filter
           self.EMG_live.applyVarianceFilter_data(width=20, mode="old_online", var_buffer=self.var_buffer)

           # # normalisation #
           self.EMG_live.normalizeContinuousData(mvc=self.channelwise_mvc, mode="old_online")

           # # low pass filter
           self.EMG_live.filterBuffer_lowPass(sos=self.sos_lp)

           # # neural activation force #
           self.EMG_live.calculateActivationForceFunction(mode='online', d=self.property["delay"], b1=self.property["beta1"], b2=self.property["beta2"], g=self.property["gamma"], nonlinear_shape_factor=self.property["A"])

           # convert filtered data into window
           self.EMG_live.bufferToWindows()
           #self.EMG_live_freq.bufferToWindows()

           if self.emg_plot:
               ## EMG - Window Extraction and Reshaping
               windows = self.EMG_live.getWindows()[0]  # (n_channels, n_samples, n_windows)
               windows = np.transpose(windows, (2, 1, 0))  # (n_windows, n_samples, n_channels)
               windows = windows[0].T #  (n_samples,n_channels)

               # # For saving emg values
               self.all_emg_vals.append(windows[:,-self.batch_size:])
               self.save_emg_vals()

               # For plotting emg in debug case
               self.update_emg_plot(self.emg_lines, windows)
               plt.pause(0.01)

           else:
               # ToDo Feature Extraction
               # ToDo Scaling - Standard and PCA look below
               # Pre-PCA scaling
               # Dimensionality reduction with PCA
               # Post-PCA scaling of input features
               # ToDo Model Prediction
               # ToDo PostProcessing

               self.extract_features()
               self.predict()

               # Printing the Timings
               update_time_step = time.perf_counter() - update_start_time
               self.current_time = self.current_time + update_time_step
               self.elapsed_times.append(self.current_time)
               print(len(self.elapsed_times))

               print(f"{update_time_step * 1000:.1f} ms")

               # Save all single predictions (and Plotting)
               self.all_predictions.append(self.predictions)
               if self.show_prediction_plot:
                   self.update_plot()

if __name__ == "__main__":

 live_estimation_obj = LiveEstimation()

 live_estimation_obj.save_all_predictions()

 live_estimation_obj.save_torques()

 live_estimation_obj.save_elapsed_times()
