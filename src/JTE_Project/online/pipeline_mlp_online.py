import warnings

import zmq
import time
import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, savgol_filter, medfilt
from sklearn.metrics import r2_score
from biosignal_toolbox.emg_lib import OnlineEMG
from tensorflow.keras.models import load_model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath
from scipy.ndimage import uniform_filter1d

# XLA-JIT einschalten (beschleunigt den Inferenz-Graph)
tf.config.optimizer.set_jit(True)

import os
# -1 bedeutet: Keine GPU sichtbar. TensorFlow nutzt automatisch die CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class LiveEstimation:

    def __init__(self):
        emg_context = zmq.Context()
        self.emg_socket = emg_context.socket(zmq.SUB)
        self.emg_socket.setsockopt(zmq.RCVTIMEO, 30000)
        self.emg_socket.connect("tcp://127.0.0.1:5555")
        print("EMG Subscriber is active")
        self.emg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Parameters for receiving batches (instead of single strings)
        self.buffer_size = 250 # 250 was good
        self.batch_size = 50
        self.emg_buffer = []
        self.emg_array = None

        config_filename = 'emg_torque_estimation_jte.yaml'
        self.cfg = loadConfig(filename=config_filename)
        print('Loaded the config file!')

        ################################################################################################################
        #################################################################################################################
        # Setup:
        # 1. Input Train EMG.TXT and corresponding 3 Reference Torque Files (.npy)
        # 2. Copy MVC File, ML-Model file, Yscaler File (Adjust used weight and movement)

        self.use_yscaler = True

        # Note that if enabled, the prediction is slower than in a real time scenario
        self.show_prediction_plot = False

        # Plotting EMG
        self.emg_plot = False

        ############################### Publisher File and Reference Torques ############################################

        current_weight = '1850g'
        current_move = 'grasp'
        set_num = '3'

        mvc_num1 = 0
        mvc_num2 = 0

        if current_weight == '0g':
            mvc_num1 = 0
        elif current_weight == '1100g':
            mvc_num1 = 1
        elif current_weight == '1850g':
            mvc_num1 = 2

        if current_move == 'grasp':
            mvc_num2 = 0
        elif current_move == 'complex':
            mvc_num2 = 1


        # Load the emg file (just for length of the file)
        print(f"Session Config: Weight={current_weight}, Move={current_move}")
        emg_filepath = f"data/jte/emg/BU62D/backup/24072025_BU62D_{current_weight}_{current_move}_{set_num}.txt"
        emg_file = getAbsolutePath(emg_filepath)
        df = pd.read_csv(emg_file, sep=" ", header=None)
        df = df.drop(df.columns[0], axis=1)
        emg_max_rows = df.shape[0] - 1
        print(f"Loaded {emg_filepath}")


       # Load Torque Values (ground truth, only in prediction plot)
        Y_e = np.load(str(getAbsolutePath(f"data/jte/quali/BU62D/torques_kartik/quali_torque_elbow_24_07_2025_BU62D_{current_weight}_{current_move}_{set_num}.npy")))
        Y_f = np.load(str(getAbsolutePath(f"data/jte/quali/BU62D/torques_kartik/quali_torque_shoulder_front_24_07_2025_BU62D_{current_weight}_{current_move}_{set_num}.npy")))
        Y_s = np.load(str(getAbsolutePath(f"data/jte/quali/BU62D/torques_kartik/quali_torque_shoulder_side_24_07_2025_BU62D_{current_weight}_{current_move}_{set_num}.npy")))
        self.Y_ref_raw = np.stack((Y_e, Y_f, Y_s), axis=1)

        ############################ COPY FROM OFFLINE TRAINING #############################################

        # pre-calculated channelwise mvc
        self.channelwise_mvc = np.load(str(getAbsolutePath("src/JTE_Project/online/resources/mvc/channelwise_mvc.npy")))[mvc_num1,mvc_num2]
        print("Loaded channelwise mvc file")

        # Loading the ML Model
        self.load_model(getAbsolutePath(
            'data/jte/ml_models/BU62D/mlp_2c2.keras'))

        self.model_type = 'mlp'
        self.subject_id = 'BU62D'
        self.current_weight = current_weight
        self.current_move = current_move
        self.set_num = set_num
        

        # Loading the Y-scaler
        if self.use_yscaler:
            y_scaler = joblib.load(getAbsolutePath("src/JTE_Project/online/resources/y_scaler/Y_scaler.pkl"))
            print("Loading Y_Scaler succesful.")
            # Select correct Scaler File
            self.scaler_file = y_scaler[current_weight][current_move]
            print(self.scaler_file)
            print(f"  Original Data Min (Nm): {self.scaler_file.data_min_}")  # [Elbow, Front, Side]
            print(f"  Original Data Max (Nm): {self.scaler_file.data_max_}")  # [Elbow, Front, Side]
            print(f"  Scale Factor:           {self.scaler_file.scale_}")
            print(f"  Min Parameter (Offset): {self.scaler_file.min_}")

        
        # Weight Vector: [0g, 1100g, 1850g]
        if current_weight == '0g':
            vec_w = [1, 0, 0]
        elif current_weight == '1100g':
            vec_w = [0, 1, 0]
        elif current_weight == '1850g':
            vec_w = [0, 0, 1]
        else:
            vec_w = [0, 0, 0]  # Fehlerfall

        # Move Vector: [grasp, complex]
        if current_move == 'grasp':
            vec_m = [1, 0]
        elif current_move == 'complex':
            vec_m = [0, 1]
        else:
            vec_m = [0, 0]

        self.static_ohe_vector = np.concatenate([vec_w, vec_m])
        print(f"Static OHE Vector: {self.static_ohe_vector}")

        # e.g. emg sample size: 41.000 and 50 samples every 100ms, means 820 predictions in total
        self.total_timepoints = int(emg_max_rows / self.batch_size)
        self.Y_ref = self.downsample_mean_bins(self.Y_ref_raw, self.total_timepoints)
        print("Total amount of Timepoints:: ", self.total_timepoints)

        # EMG 8 channel names
        self.channel_names = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']


        # Create old_online EMG object
        self.EMG_live = OnlineEMG(stream_type="data",
                                  channel_names=self.channel_names, buffer_size=self.buffer_size,
                                  f_samp=self.cfg.preprocess_param.f_samp)
        print("Created EMG_live!!")

        self.var_buffer = np.zeros((1, len(self.channel_names), self.buffer_size, 1))

        self.lp_mirror = np.zeros_like(self.EMG_live.getDataBuffer())

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
        self.peak_features = None
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
        self.timeout = False

        if self.show_prediction_plot:
            self.setup_plot()

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
            l_pred, = ax.plot([], [], linewidth=1.5, label=f'{n} (Pred)') # linewidth=1.8
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

        # return last values for rmse calculation
        return Yp[-1:], Yr[-1:]


    def load_model_old(self, model_path=None):
        """
        Load the trained TCN model.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to the saved model. If None, uses default path from config.
        """

        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path, compile=False)
        print("Model loaded successfully!\n")

        return True

    def load_model(self, model_path=None):
        """
        Load the trained TCN/MLP model and compile a dedicated inference graph.
        """
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path, compile=False)

        # --- OPTIMIERUNG: Erstellen einer kompilierten Graph-Funktion ---
        # jit_compile=True nutzt XLA (Accelerated Linear Algebra) für maximale Performance
        @tf.function(jit_compile=True)
        def inference_step(inputs):
            return self.model(inputs, training=False)

        self.inference_func = inference_step

        # --- WARM-UP: Der erste Aufruf ist immer langsam (Tracing) ---
        try:
            # Input-Shape ermitteln (Batch-Dimension ignorieren)
            input_shape = self.model.input_shape
            if input_shape[0] is None:
                # Dummy Batch size von 1, Rest aus Model-Info
                dummy_shape = (1,) + tuple(input_shape[1:])
            else:
                dummy_shape = input_shape

            print(f"Running Warm-up with shape {dummy_shape}...")
            dummy_input = tf.zeros(dummy_shape)
            self.inference_func(dummy_input)
            print("Model loaded and warmed up successfully!\n")
        except Exception as e:
            print(f"Warning during Warm-up: {e}. First prediction might be slower.")

        return True

    def start_device_stream(self):
            # Start EMG live stream (unused in Pseudoonline)
            self.EMG_live.startANTEegoStreaming(path_to_so_file="path_to_so_file")

            return True

    def extract_features(self):
        """Extract all features from windowed EMG data."""
        #print("Extracting features from windowed data...")

        # Timepoints feature extraction
        window_size_ms = (
                self.cfg.preprocess_param.window_size_x * 1000 / self.cfg.preprocess_param.f_samp
        )
        feature_indices_windows_x = np.array([0, window_size_ms])

        self.EMG_live.featureExtractionFromWindows(
            feature_type="timepoints",
            feature_indices_windows=feature_indices_windows_x
        )

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
        #print(f"Total EMG features extracted: {self.features.shape}")
        #print("Feature extraction completed!\n")

    def extract_sliding_features(self, n_windows=5):
        """
        Manually extracts 'n_windows' overlapping windows from the end of the buffer.
        This creates a batch of data covering the last 100ms with higher resolution.
        """
        raw_data = self.EMG_live.getDataBuffer()[0, :, :, 0]

        n_channels, n_samples = raw_data.shape

        model_window_len = self.batch_size

        stride = int(self.batch_size / n_windows)  # e.g., 50 / 5 = 10 samples step

        batch_windows = []

        current_idx = n_samples

        for i in range(n_windows):
            # Calculate end index for this slice
            # i=0 -> offset=40, i=4 -> offset=0
            offset = (n_windows - 1 - i) * stride
            end_idx = current_idx - offset
            start_idx = end_idx - model_window_len

            # Safety check
            if start_idx < 0:
                # Pad with zeros or duplicate if buffer isn't full yet
                window = np.zeros((n_channels, model_window_len))
                available = raw_data[:, :end_idx]
                window[:, -available.shape[1]:] = available
            else:
                window = raw_data[:, start_idx:end_idx]

            batch_windows.append(window.T)  # Transpose to (Time, Channels)

        # Stack into a batch: (Batch_Size, Timepoints, Channels)
        # e.g., (5, 50, 8)
        self.features = np.stack(batch_windows, axis=0)

        return self.features

    def apply_scaler(self):
        "HOWEVER wrong, not use for EMG DATA"
        arr = self.features.flatten()
        elements_to_keep = (arr.shape[0] // 3) * 3
        arr_trimmed = arr[:elements_to_keep]
        arr_reshaped = arr_trimmed.reshape(-1, 3)
        print(arr_reshaped.shape)
        scaled_feats = self.EMG_live.scaleEMG_windows(scaler_file=None, test_data=arr_reshaped)
        self.features = scaled_feats.reshape(1, -1)

    def inverse_transform_scaler(self):
        self.predictions = self.scaler_file.inverse_transform(self.predictions)

    def predict_old(self):
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
        mlp_in = self.features

        # One-Hot Vector (Batch, 5)
        batch_size = mlp_in.shape[0]
        one_hot_batch = np.tile(self.static_ohe_vector, (batch_size, 1))

        # Reshape for CNN/TCN input: (samples, features, channels)
        mlp_in = np.concatenate([mlp_in, one_hot_batch], axis=1)

        #print(f"Input shape for model: {features_cnn.shape}")
        # Predict
        x = tf.convert_to_tensor(mlp_in)
        preds = self.model(x, training=False) # using a direct model call vs self.model.predict() cuts time expense in half
        # Concatenate multi-task outputs: [elbow, front, side]
        self.predictions = np.asarray(preds)
        #self.inverse_transform_scaler()
        #print(f"Predictions shape: {self.predictions.shape}")
        #print("Model prediction completed!\n")

        return self.predictions

    def predict(self):
        """
        Process EMG file and predict joint torques using the compiled graph.
        """
        # Feature Referenz holen
        mlp_in = self.features  # Shape: (Batch, N_Features)

        batch_size = mlp_in.shape[0]

        one_hot_batch = np.tile(self.static_ohe_vector, (batch_size, 1))
        final_input = np.concatenate([mlp_in, one_hot_batch], axis=1)

        # Umwandlung in Tensor
        # Wir rufen direkt die kompilierte Funktion auf
        preds_tensor = self.inference_func(tf.convert_to_tensor(final_input, dtype=tf.float32))

        # Rückumwandlung in Numpy (.numpy() ist hier notwendig, da Tensor zurückkommt)
        self.predictions = preds_tensor.numpy()

        self.inverse_transform_scaler()
        print(f"Predictions shape: {self.predictions.shape}")
        print("Model prediction completed!\n")

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

            except zmq.error.Again:
                # recv_string timed out (no data available)
                print("⚠️ Socket timeout - no EMG data received")
                self.timeout = True
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")

        # Buffer leeren
        self.emg_buffer = []

    def apply_median_savitzky(self, sav_filter_size = 5, poly_order = 2, mean_filter_size = 1):
        if len(self.all_predictions) >= sav_filter_size - 1:
            # get the previous predictions
            all_preds = np.vstack(self.all_predictions[-(sav_filter_size - 1):])

            # median_temp (3,10) -> Result (3,) -> sav_temp -> Result -> (3,) -> Append to predictions

            # apply the mean filter
            mean_temp = np.vstack((all_preds, self.predictions))

            if mean_temp.shape[0] >= mean_filter_size:
                window = mean_temp[-mean_filter_size:, :]
                mean_sample = np.mean(window, axis=0)
            else:
                mean_sample = mean_temp[-1, :]


            # apply the savitzky filter
            sav_temp = np.vstack((all_preds, mean_sample))

            preds_sav = np.array([
                savgol_filter(sav_temp[:, i], window_length=sav_filter_size, polyorder=poly_order)[-1]
                for i in range(sav_temp.shape[1])
            ])

            filtered = np.maximum(preds_sav, 0)

            self.all_predictions.append(filtered)
        else:
            # append raw data if self.all_predictions does not match size
            self.all_predictions.append(self.predictions)


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
        list_of_2d_arrays = [np.atleast_2d(arr) for arr in self.all_predictions]
        all_preds = np.concatenate(list_of_2d_arrays, axis=0)
        np.save(str(getAbsolutePath(
            f"src/JTE_Project/online/online_results/{self.model_type}/{self.subject_id}/all_predictions_{self.current_weight}_{self.current_move}_{self.set_num}.npy")),
                all_preds)

    def save_torques(self):
        np.save(str(getAbsolutePath(
            f"src/JTE_Project/online/online_results/{self.model_type}/{self.subject_id}/all_torques_{self.current_weight}_{self.current_move}_{self.set_num}.npy")),
                self.Y_ref_raw)

    def save_elapsed_times(self):
        elapsed_time_np = np.array(self.elapsed_times)
        np.save(str(getAbsolutePath(
            f"src/JTE_Project/online/online_results/{self.model_type}/{self.subject_id}/all_times_{self.current_weight}_{self.current_move}_{self.set_num}.npy")),
                elapsed_time_np)

    def save_emg_vals(self):
        all_emgs = np.concatenate(self.all_emg_vals, axis=1)
        np.save(str(getAbsolutePath(
            f"src/JTE_Project/online/online_results/{self.model_type}/{self.subject_id}/bandpass_online_{self.current_weight}_{self.current_move}_{self.set_num}.npy")),
                all_emgs)
        print("Save EMG Shape: ", all_emgs.shape)


    def update_loop(self):

        feat_times = []
        pred_times = []
        update_times = []
        
        while len(self.elapsed_times) < self.total_timepoints:
            update_read_time = time.perf_counter()
            # Read emg data from the stream
            self.read_emg_batch()
            if self.timeout:
                print("Exiting update loop due to timeout.")
                break
            end_read_time = time.perf_counter() - update_read_time
            #print(f"EMG BATCH: {end_read_time * 1000:.1f} ms")

            # Start the time measurement (self.read_emg_batch waits for 50 new samples, approx 65ms duration)
            update_start_time = time.perf_counter()

            # Manually setting the data (otherwise use start_hook())
            self.EMG_live.setChunk(self.emg_array, chunk_type="numpy")

            # Update the internal ring buffer
            self.EMG_live.updateBuffer(num_channels=8)

            # * apply bandpass filter (previous highpass filter)
            self.EMG_live.filterBuffer_bandPass(sos=self.sos_bp)

            # # Store previous values for the variance filter
            self.var_buffer = np.roll(self.var_buffer, shift=int(-1 * self.batch_size), axis=2)
            self.var_buffer[0, :, int(-1 * self.batch_size):, 0] = self.EMG_live.getDataBuffer()[0, :,
                                                                   int(-1 * self.batch_size):, 0]


            # Create identical copy for frequency extraction
            # Only the Bandpass-Filter is used to reduce distortions introduced by the variance filter and maintain frequency components (for freq)
            self.EMG_live_freq = copy.deepcopy(self.EMG_live)


            # # variance filter
            self.EMG_live.applyVarianceFilter_data(width=20, mode="old_online", var_buffer=self.var_buffer)

            # # normalisation #
            self.EMG_live.normalizeContinuousData(mvc=self.channelwise_mvc, mode="old_online")

            # # low pass filter
            self.EMG_live.filterBuffer_lowPass(sos=self.sos_lp)

            # # neural activation force #
            if self.cfg.preprocess_param.use_activation_fncn:
                current_lp_batch = self.EMG_live.getDataBuffer()[:, :, -self.batch_size:, :]
                self.lp_mirror = np.roll(self.lp_mirror, shift=int(-1 * self.batch_size), axis=2)
                self.lp_mirror[:, :, -self.batch_size:, :] = current_lp_batch
                af_history_backup = self.EMG_live.getDataBuffer()[:, :, :-self.batch_size, :].copy()
                self.EMG_live.data_buffer[...] = self.lp_mirror[...]

                self.EMG_live.calculateActivationForceFunction(
                    mode='online',
                    d=self.cfg.preprocess_param.act_delay,
                    b1=self.cfg.preprocess_param.act_beta1,
                    b2=self.cfg.preprocess_param.act_beta2,
                    g=self.cfg.preprocess_param.act_gamma,
                    nonlinear_shape_factor=self.cfg.preprocess_param.act_A
                )

                self.EMG_live.data_buffer[:, :, :-self.batch_size, :] = af_history_backup

            # convert filtered data into window
            self.EMG_live.bufferToWindows()
            self.EMG_live_freq.bufferToWindows()

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
                # Regular prediction pipeline
                self.extract_features()
                #self.scale_features()

                pred_timer = time.perf_counter()
                self.predict()
                pred_end_timer = time.perf_counter() - pred_timer
                print(f"Prediction Time Step: {pred_end_timer * 1000:.1f} ms")

                self.apply_median_savitzky(sav_filter_size=9, poly_order=2, mean_filter_size = 1)

                # Printing the Timings
                update_time_step = time.perf_counter() - update_start_time
                print(f"{update_time_step * 1000:.1f} ms")

                self.current_time = self.current_time + update_time_step
                self.elapsed_times.append(self.current_time)
                print("Elapsed Times " , len(self.elapsed_times))

                if update_time_step < 0.03:  # Only consider time steps that are less than 100ms (i.e., real-time performance)
                    update_times.append(update_time_step)
                    feat_times.append(update_time_step-pred_end_timer)
                    pred_times.append(pred_end_timer)


                if self.show_prediction_plot:
                    _, _ = self.update_plot()

        print(f"Average Feature Extraction Time (ms): {np.mean(np.array(feat_times)) * 1000:.3f} +- {np.std(np.array(feat_times)) * 1000:.3f}")
        print(f"Average Prediction Time (ms): {np.mean(np.array(pred_times)) * 1000:.3f} +- {np.std(np.array(pred_times)) * 1000:.3f}")
        print(f"Average Update Loop Time (ms): {np.mean(np.array(update_times)) * 1000:.3f} +- {np.std(np.array(update_times)) * 1000:.3f}")
        print(f"Total number of predictions: {len(self.all_predictions)}")


if __name__ == "__main__":

    live_estimation_obj = LiveEstimation()

    live_estimation_obj.save_all_predictions()

    live_estimation_obj.save_torques()

    live_estimation_obj.save_elapsed_times()
