import warnings

import zmq
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy.signal import butter, savgol_filter, medfilt
from biosignal_toolbox.emg_lib import OnlineEMG
from tensorflow.keras.models import load_model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath
from scipy.ndimage import uniform_filter1d

import os
# -1 bedeutet: Keine GPU sichtbar. TensorFlow nutzt automatisch die CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class LiveEstimation:

    def __init__(self):
        emg_context = zmq.Context()
        self.emg_socket = emg_context.socket(zmq.SUB)
        self.emg_socket.connect("tcp://127.0.0.1:5555")
        print("EMG Subscriber is active")
        self.emg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Parameters for receiving batches (instead of single strings)
        self.buffer_size = 250  # 250 was good
        self.batch_size = 50
        self.emg_buffer = []
        self.emg_array = None

        config_filename = 'pipeline_jte_bu62d.yaml'
        self.cfg = loadConfig(filename=config_filename)
        print('Loaded the config file!')

        ################################################################################################################
        #################################################################################################################
        # Setup:
        # 1. Input Train EMG.TXT and corresponding 3 Reference Torque Files (.npy)
        # 2. Copy MVC File, ML-Model file, Yscaler File (Adjust used weight and movement)

        self.use_yscaler = False

        # Note that if enabled, the prediction is slower than in a real time scenario
        self.show_prediction_plot = False

        # Plotting EMG
        self.emg_plot = False

        ############################### Publisher File and Reference Torques ############################################

        current_weight = '1100g'
        current_move = 'complex'
        set_num = '2'

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
        self.channelwise_mvc = np.load(str(getAbsolutePath("src/JTE_Project/online/resources/mvc/channelwise_mvc_pd.npy")))
        print("Loaded channelwise mvc file")

        # Loading the ML Model
        self.load_model(getAbsolutePath(
            'src/JTE_Project/online/resources/trained_models/tcn_model_pd.keras'))

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

        ###########################################################################################################
        ###########################################################################################################

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
            l_pred, = ax.plot([], [], linewidth=1.5, label=f'{n} (Pred)')  # linewidth=1.8
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
        xlims = (x[0], x[0] + 10 + 1) if x[-1] <= x[0] + 10 else (float(x[-1] - 10), float(x[-1] + 1))
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

        self.model = load_model(model_path, compile=False)
        print("Model loaded successfully!")

        return True

    def load_model(self, model_path=None):
        """
        Lädt das Modell und kompiliert es zu einer optimierten Concrete Function (self.inference_func).
        """
        # 1. Das normale Keras Modell laden (langsam, Python-Overhead)
        self.model = load_model(model_path, compile=False)
        print("Keras Model loaded.")

        # ---------------------------------------------------------
        # GPU Optimierung: XLA + Concrete Function erstellen
        # ---------------------------------------------------------

        # WICHTIG: Berechne hier die exakte Größe deines Static-Inputs
        # Peak Features (400) + One-Hot Vector (5) = 405
        static_input_dim = 405

        # Definition der Input-Typen für den Compiler (Signaturen)
        # shape=(None, ...) bedeutet: Batch-Size ist flexibel (1 oder 50 ist egal)
        sig_cnn = tf.TensorSpec(shape=(None, 50, 8), dtype=tf.float32, name="cnn_input")
        sig_static = tf.TensorSpec(shape=(None, static_input_dim), dtype=tf.float32, name="static_input")

        # Wir bauen eine Wrapper-Funktion mit XLA (jit_compile=True)
        @tf.function(jit_compile=True)
        def fast_predict_wrapper(cnn_tensor, static_tensor):
            return self.model([cnn_tensor, static_tensor], training=False)

        # Hier entsteht 'self.inference_func'!
        self.inference_func = fast_predict_wrapper.get_concrete_function(sig_cnn, sig_static)

        # ---------------------------------------------------------
        # WARMUP
        # ---------------------------------------------------------
        print("Starting CPU or GPU Warmup...")
        try:
            # Dummy Daten durchjagen
            dummy_cnn = tf.zeros((1, 50, 8), dtype=tf.float32)
            dummy_static = tf.zeros((1, static_input_dim), dtype=tf.float32)
            _ = self.inference_func(dummy_cnn, dummy_static)
            print("🚀 CPU or GPU Inference Graph successfully built and warmed up!")
        except Exception as e:
            print(f"⚠️ Warmup Warning: {e}")
            print("Überprüfe bitte, ob 'static_input_dim' (hier 405) wirklich zu deinen Daten passt!")

        return True

    def start_device_stream(self):
        # Start EMG live stream (unused in Pseudoonline)
        self.EMG_live.startANTEegoStreaming(path_to_so_file="path_to_so_file")

        return True

    def extract_features_old(self):
        """Extract all features from windowed EMG data."""
        # print("Extracting features from windowed data...")

        # Timepoints feature extraction
        window_size_ms = (
                self.cfg.preprocess_param.window_size_x * 1000 / self.cfg.preprocess_param.f_samp
        )
        feature_indices_windows_x = np.array([0, window_size_ms])

        self.EMG_live.featureExtractionFromWindows(
            feature_type="timepoints",
            feature_indices_windows=feature_indices_windows_x
        )
        advanced_feature_extraction = False
        if advanced_feature_extraction:
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

        # print("Feature extraction completed!\n")
        self.features = self.EMG_live.getFeatures()

        #print("Self.features pre reshaping!: ", self.features.shape)

        '''
        windows = self.EMG_live.getWindows()[0]  # (n_channels, n_samples, n_windows)
        windows = np.transpose(windows, (2, 1, 0))  # (n_windows, n_samples, n_channels)
        n_channels = 8
        self.features = windows.reshape(-1, self.batch_size, n_channels)
        '''

        # Reshape for CNN/TCN input: (samples, features, channels)
        n_features = self.features.shape[1]
        n_channels = 8
        n_timepoints = int(n_features / n_channels)
        self.features = self.features.reshape((-1, n_timepoints, n_channels))

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

    def extract_sliding_features_pd(self, n_windows=5):
        """
        Extracts raw windows for TCN input AND calculates the difference
        to previous windows for the Static input (Peak Detection).
        """
        raw_data = self.EMG_live.getDataBuffer()[0, :, :, 0]
        n_channels, n_samples = raw_data.shape

        model_window_len = self.batch_size
        # Stride bestimmt, wie weit das "vorherige Fenster" weg ist
        stride = int(self.batch_size / n_windows)

        batch_windows_raw = []  # Für TCN (Input 1)
        batch_windows_diff = []  # Für Static (Input 2)

        current_idx = n_samples

        for i in range(n_windows):
            # Indizes für das aktuelle Fenster berechnen
            offset = (n_windows - 1 - i) * stride
            end_idx = current_idx - offset
            start_idx = end_idx - model_window_len

            # --- 1. Raw Window (TCN Input) ---
            if start_idx < 0:
                # Padding falls am Anfang
                window_curr = np.zeros((n_channels, model_window_len))
                available = raw_data[:, :end_idx]
                window_curr[:, -available.shape[1]:] = available
            else:
                window_curr = raw_data[:, start_idx:end_idx]

            batch_windows_raw.append(window_curr.T)
            # --- 2. Peak Detection (Static Input) ---
            # Wir brauchen das "vorherige" Fenster, um die Differenz zu bilden.
            # Im Offline-Skript ist das np.diff() auf aufeinanderfolgende Fenster.
            # Wir nehmen hier einfach das Fenster, das um 1 'stride' verschoben ist
            # (oder 1 Sample, je nach Offline-Einstellung. Stride ist sicherer für "Fenster-Diff").

            prev_end_idx = end_idx - stride  # Oder -1, wenn offline sample-weise diff gemacht wurde
            prev_start_idx = start_idx - stride

            if prev_start_idx < 0:
                window_prev = np.zeros((n_channels, model_window_len))
                # Falls Daten verfügbar, füllen (vereinfacht)
                if prev_end_idx > 0:
                    avail_prev = raw_data[:, :prev_end_idx]
                    window_prev[:, -avail_prev.shape[1]:] = avail_prev
            else:
                window_prev = raw_data[:, prev_start_idx:prev_end_idx]

            diff_vec = window_curr.flatten() - window_prev.flatten()

            batch_windows_diff.append(diff_vec)

        # Stack TCN Input: (Batch, Time, Channels)
        self.features = np.stack(batch_windows_raw, axis=0)

        # Stack Peak-Features: (Batch, Features_Size)
        self.peak_features = np.stack(batch_windows_diff, axis=0)

        return self.features, self.peak_features

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
        # TCN Input (Raw Time Series)
        # Shape: (Batch, 50, 8)
        cnn_in = self.features

        # ---------------------------------------------------------
        # Static Input
        # ---------------------------------------------------------
        # Peak Detection Features (Batch, 400)
        peak_feats = self.peak_features

        # One-Hot Vector (Batch, 5)
        batch_size = cnn_in.shape[0]
        one_hot_batch = np.tile(self.static_ohe_vector, (batch_size, 1))

        static_in = np.concatenate([peak_feats, one_hot_batch], axis=1)

        # ---------------------------------------------------------
        x = tf.convert_to_tensor(cnn_in)
        s = tf.convert_to_tensor(static_in)
        # Predict
        preds = self.model([x, s], training=False)

        # Concatenate outputs
        self.predictions = np.concatenate(
            [preds[0], preds[1], preds[2]], axis=1
        )

        return self.predictions

    def predict(self):
        """
        Führt die Vorhersage mit der optimierten self.inference_func aus.
        """
        # 1. Daten als Float32 vorbereiten (Numpy standard ist oft float64 -> langsam auf GPU)
        cnn_in_np = self.features.astype(np.float32)  # (Batch, 50, 8)
        batch_size = cnn_in_np.shape[0]

        # 2. One-Hot Caching (Vermeidet np.tile Overhead)
        if not hasattr(self, 'cached_ohe') or self.cached_ohe.shape[0] != batch_size:
            # Erstelle Array nur neu, wenn sich Batch-Size ändert
            self.cached_ohe = np.tile(self.static_ohe_vector, (batch_size, 1)).astype(np.float32)

        # 3. Static Input zusammenbauen
        # peak_features müssen auch float32 sein!
        peak_feats_32 = self.peak_features.astype(np.float32)
        static_in_np = np.concatenate([peak_feats_32, self.cached_ohe], axis=1)

        # 4. Umwandlung in Tensoren (schiebt Daten auf GPU)
        cnn_tensor = tf.convert_to_tensor(cnn_in_np)
        static_tensor = tf.convert_to_tensor(static_in_np)

        # 5. Aufruf der optimierten Funktion (DAS ist der schnelle Teil)
        # self.inference_func wurde in load_model erstellt
        preds_tensors = self.inference_func(cnn_tensor, static_tensor)

        # 6. Ergebnisse zurückholen (Numpy)
        # preds_tensors ist eine Liste [Tensor(Elbow), Tensor(Front), Tensor(Side)]
        res_e = preds_tensors[0].numpy()
        res_f = preds_tensors[1].numpy()
        res_s = preds_tensors[2].numpy()

        self.predictions = np.concatenate([res_e, res_f, res_s], axis=1)

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

    def apply_median_savitzky(self, sav_filter_size=5, poly_order=2, mean_filter_size=1):
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
        np.save(str(getAbsolutePath("src/JTE_Project/online/online_results/all_predictions.npy")), all_preds)

    def save_torques(self):
        np.save(str(getAbsolutePath("src/JTE_Project/online/online_results/all_torques.npy")), self.Y_ref_raw)

    def save_elapsed_times(self):
        elapsed_time_np = np.array(self.elapsed_times)
        np.save(str(getAbsolutePath("src/JTE_Project/online/online_results/all_times.npy")), elapsed_time_np)

    def save_emg_vals(self):
        all_emgs = np.concatenate(self.all_emg_vals, axis=1)
        np.save(str(getAbsolutePath("src/JTE_Project/online/online_results/bandpass_online.npy")), all_emgs)
        print("Save EMG Shape: ", all_emgs.shape)

    def update_loop(self):
        while len(self.elapsed_times) < self.total_timepoints:
            update_read_time = time.perf_counter()
            # Read emg data from the stream
            self.read_emg_batch()
            end_read_time = time.perf_counter() - update_read_time

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

            '''
            if self.advanced_feature_extraction:
                # Create identical copy for frequency extraction
                # Only the Bandpass-Filter is used to reduce distortions introduced by the variance filter and maintain frequency components (for freq)
                self.EMG_live_freq = copy.deepcopy(self.EMG_live)
            '''

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
            '''
            if self.advanced_feature_extraction:
                self.EMG_live_freq.bufferToWindows()
            '''
            if self.emg_plot:
                ## EMG - Window Extraction and Reshaping
                windows = self.EMG_live.getWindows()[0]  # (n_channels, n_samples, n_windows)
                windows = np.transpose(windows, (2, 1, 0))  # (n_windows, n_samples, n_channels)
                windows = windows[0].T  # (n_samples,n_channels)

                # # For saving emg values
                self.all_emg_vals.append(windows[:, -self.batch_size:])
                self.save_emg_vals()

                # For plotting emg in debug case
                self.update_emg_plot(self.emg_lines, windows)
                plt.pause(0.01)

            else:
                # Regular prediction pipeline
                self.extract_sliding_features_pd(n_windows=2)

                pred_timer = time.perf_counter()
                self.predict()
                pred_end_timer = time.perf_counter() - pred_timer

                if self.use_yscaler:
                    self.inverse_transform_scaler() # Applying the inverse transform of the y scaler

                #self.apply_median_savitzky(sav_filter_size=9, poly_order=2, mean_filter_size=1)
                self.all_predictions.extend(self.predictions)

                update_time_step = time.perf_counter() - update_start_time
                self.current_time = self.current_time + update_time_step
                self.elapsed_times.append(self.current_time)

                # Printing the Timings
                print(f"EMG BATCH: {end_read_time * 1000:.1f} ms")
                print(f"Prediction Time Step: {pred_end_timer * 1000:.1f} ms")
                print(f"Pipeline Time Step: {update_time_step * 1000:.1f} ms")
                print("Elapsed Times ", len(self.elapsed_times))

                if self.show_prediction_plot:
                    _, _ = self.update_plot()


if __name__ == "__main__":
    live_estimation_obj = LiveEstimation()

    live_estimation_obj.save_all_predictions()

    live_estimation_obj.save_torques()

    live_estimation_obj.save_elapsed_times()
