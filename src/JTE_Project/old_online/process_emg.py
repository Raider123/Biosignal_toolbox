"""
Online EMG to Torque Prediction System
---------------------------------------
This script processes a single EMG.TXT file for old_online joint torque estimation.
The preprocessing is separated into a dedicated class for modularity.
"""

# ! ************************************************
# ! Imports
# ! ************************************************

import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np

# Own libs
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

warnings.formatwarning = customWarningFormat

# Model loading
from tensorflow.keras.models import load_model

# Post-filtering
from scipy.signal import medfilt, savgol_filter


# ! ************************************************
# ! EMG Preprocessing Class
# ! ************************************************

class EMGPreprocessor:
    """
    Handles all preprocessing steps for EMG data including filtering,
    normalization, windowing, and feature extraction.
    """

    def __init__(self, config):
        """
        Initialize the preprocessor with configuration parameters.

        Parameters
        ----------
        config : Config object
            Configuration loaded from YAML file
        """
        self.cfg = config
        self.emg_data = None
        self.emg_data_freq = None
        self.channel_names = None
        self.channelwise_mvc = None
        self.features = None

    def load_emg_file(self, data_array):
        """
        Load a single EMG file.

        Parameters
        ----------
        data_array cf. EMGData class in emg_lib.py
        """

        self.emg_data = EMGData(
            data_arr = data_array,
            format="ANTmini",
            filenames=None,
            data_path=self.cfg.filepath.data_path,
            f_samp=self.cfg.preprocess_param.f_samp,
            channel_names=self.cfg.preprocess_param.channel_names_emg
        )

        self.channel_names = self.emg_data.getChannelNames()
        print(f"Channel Names: {self.channel_names}")
        print(f"Channel Length: {len(self.channel_names)}\n")

    def apply_bandpass_filter(self):
        """Apply highpass and lowpass bandpass filtering to EMG data."""
        print("Applying bandpass filter...")

        # Design bandpass filter
        sos_hp = self.emg_data.designFilter(
            f_high=self.cfg.preprocess_param.f_cutoff_hpf,
            f_low=self.cfg.preprocess_param.f_cutoff_lpf,
            order=self.cfg.preprocess_param.filter_order,
            filter_type="scipy_butter",
            return_type="sos"
        )

        # Apply bandpass filter
        self.emg_data.filterData_offline(
            filter_method=self.cfg.preprocess_param.filter_method,
            sos=sos_hp
        )

        # Create a copy for frequency-domain processing
        self.emg_data_freq = deepcopy(self.emg_data)
        print("Bandpass filter applied!\n")

    def apply_variance_filter(self):
        """Apply variance filter to EMG data."""
        print("Applying variance filter...")

        width = self.cfg.preprocess_param.var_filter_width
        ring_buffer = np.zeros(width)
        index = 0

        self.emg_data.applyVarianceFilter_data(
            ring_buffer=ring_buffer,
            width=width,
            index=index
        )
        print("Variance filter applied!\n")

    def normalize_data(self):
        """Normalize EMG data using MVC (Maximum Voluntary Contraction)."""
        print("Calculating channel-wise MVC for EMG...")

        self.channelwise_mvc = np.max(
            np.abs(self.emg_data.data), axis=1
        ).reshape(-1, 1)

        print("Performing input normalization with MVC...")

        if self.cfg.preprocess_param.normalisation_method == 'overall_mvc':
            self.emg_data.normalizeContinuousData(mvc=np.max(self.channelwise_mvc))
        elif self.cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
            self.emg_data.normalizeContinuousData(mvc=self.channelwise_mvc)
        else:
            warnings.warn("Normalization method not implemented! Omitting!")

        print("Input normalization with MVC performed!\n")

    def apply_smoothing_filter(self):
        """Apply lowpass smoothing filter to EMG data."""
        print("Applying smoothing lowpass filter...")

        # Design lowpass filter
        sos_lp = self.emg_data.designFilter(
            f_low=self.cfg.preprocess_param.f_cutoff_sm_lpf,
            order=self.cfg.preprocess_param.sm_filter_order,
            filter_type="scipy_butter",
            return_type="sos"
        )

        # Apply lowpass filter
        self.emg_data.filterData_offline(
            filter_method=self.cfg.preprocess_param.filter_method,
            sos=sos_lp
        )
        print("Smoothing lowpass filter applied!\n")

    def apply_activation_function(self):
        """Calculate neural activation force if enabled."""
        if self.cfg.preprocess_param.use_activation_fncn:
            print("Calculating neural activation force...")

            self.emg_data.calculateActivationForceFunction(
                d=self.cfg.preprocess_param.act_delay,
                b1=self.cfg.preprocess_param.act_beta1,
                b2=self.cfg.preprocess_param.act_beta2,
                g=self.cfg.preprocess_param.act_gamma,
                nonlinear_shape_factor=self.cfg.preprocess_param.act_A
            )
            print("Neural activation force calculated!\n")

    def window_data(self):
        """Window the continuous EMG data."""
        print("Windowing EMG data...")

        emg_window_boundary_idx, _ = self.emg_data.windowContinuousData(
            startmarkernumber=1,
            stopmarkernumber=2,
            window_size=self.cfg.preprocess_param.window_size_x,
            window_step=self.cfg.preprocess_param.window_step,
            start_index_offset=0,
            start_channel_pick=0,
            end_channel_pick=8,
            return_window_end_indices=True
        )

        _, _ = self.emg_data_freq.windowContinuousData(
            startmarkernumber=1,
            stopmarkernumber=2,
            window_size=self.cfg.preprocess_param.window_size_x,
            window_step=self.cfg.preprocess_param.window_step,
            start_index_offset=0,
            start_channel_pick=0,
            end_channel_pick=8,
            return_window_end_indices=True
        )
        print("Data windowed!\n")

        # Output the windows.shape
        print("TEST   ", self.emg_data.windows.shape)

    def extract_features(self):
        """Extract all features from windowed EMG data."""
        print("Extracting features from windowed data...")

        # Timepoints feature extraction
        window_size_ms = (
            self.cfg.preprocess_param.window_size_x * 1000 / self.emg_data.f_samp
        )
        feature_indices_windows_x = np.array([0, window_size_ms])

        self.emg_data.featureExtractionFromWindows(
            feature_type="timepoints",
            feature_indices_windows=feature_indices_windows_x
        )
        '''
        # Time domain features
        n_channels = len(self.channel_names)

        # RMS (Root Mean Square)
        rms_feature = self.emg_data.getRMSFeatures_windows(n_channels=n_channels)
        self.emg_data.addFeatures(rms_feature)

        # Waveform Length
        wfl_feature = self.emg_data.getWaveformLengthFeatures_windows(
            n_channels=n_channels
        )
        self.emg_data.addFeatures(wfl_feature)

        # Slope Sign Change
        ssc_feature = self.emg_data.getSlopeSignChangeFeatures_windows(
            n_channels=n_channels,
            threshold=0.02
        )
        self.emg_data.addFeatures(ssc_feature)

        # Frequency domain features
        self.emg_data_freq.featureExtractionFromWindows(
            feature_type="freqBandPower",
            psd_method="multitaper",
            freq_bands=[15, 50, 100, 150, 200, 245]
        )
        fbp_feature = self.emg_data_freq.getFeatures()
        self.emg_data.addFeatures(fbp_feature)

        # Time-frequency domain features (Morlet Wavelet)
        freqs = np.arange(start=50, stop=226, step=25)
        n_cycles = np.ones(len(freqs)) * 5
        n_cycles[0] = 3
        n_cycles[1] = 4

        mwc_feature = self.emg_data_freq.getMorletWaveletCoeffFeatures_windows(
            freqs=freqs,
            n_cycles=n_cycles
        )
        self.emg_data.addFeatures(mwc_feature)

        # Differential features (change between consecutive windows)
        peak_detection = np.diff(
            self.emg_data.getFeatures(),
            axis=0,
            prepend=self.emg_data.getFeatures()[0:1, :]
        )
        self.emg_data.addFeatures(peak_detection)
        '''

        self.features = self.emg_data.getFeatures()
        print(f"Total EMG features extracted: {self.features.shape}")
        print("Feature extraction completed!\n")

    def scale_features(self):
        """Scale features using StandardScaler and apply PCA."""
        print("Scaling features...")

        # Pre-PCA scaling
        scaler, scaled_features, _, _ = self.emg_data.scaleFeatures_windows(
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


    def preprocess(self, data_array):
        """
        Run the complete preprocessing pipeline.

        Parameters
        ----------
        data_array: cf. EMG_Data class in emg_lib.py

        Returns
        -------
        np.ndarray
            Preprocessed features ready for model input
        """
        self.load_emg_file(data_array)
        self.apply_bandpass_filter()
        self.apply_variance_filter()
        self.normalize_data()
        self.apply_smoothing_filter()
        self.apply_activation_function()
        self.window_data()
        self.extract_features()
        #self.scale_features()

        return self.features


# ! ************************************************
# ! Online EMG Predictor Class
# ! ************************************************

class OnlineEMGPredictor:
    """
    Main class for old_online EMG-to-torque prediction.
    Handles model loading, preprocessing orchestration, and prediction.
    """

    def __init__(self, config_filename='pipeline_mlp_to_cnn.yaml'):
        """
        Initialize the old_online predictor.

        Parameters
        ----------
        config_filename : str
            Name of the YAML configuration file
        """
        print("=" * 60)
        print("Online EMG to Torque Prediction System")
        print("=" * 60 + "\n")

        # Load configuration
        self.cfg = loadConfig(filename=config_filename)

        # Initialize preprocessor
        self.preprocessor = EMGPreprocessor(self.cfg)

        # Model and predictions
        self.model = None
        self.predictions = None

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

    def predict(self, data_array):
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
        # Preprocess the EMG data
        features = self.preprocessor.preprocess(data_array)

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

        # Inverse Transform of the predictions (Yscaler)
        from joblib import load

        data = load('F:/SMT_MASTERPROJEKT/biosignal_toolbox/src/JTE_Project/offline/saved_online_models/scaler_data.joblib')
        Y_scaler_dict = data['dict']
        Y_scaler_info = data['info']
        rescaled_predictions = []
        '''
        for wgt, mov, start_idx, end_idx in Y_scaler_info:
            # Select the corresponding predictions to the Y_scaler
            predictions_select = self.predictions[start_idx:end_idx]

            # Load the correct Y_scaler
            scaler = Y_scaler_dict[wgt][mov]

            # Inverse transformation
            rescaled_predictions.append(scaler.inverse_transform(predictions_select))

        rescaled_predictions = np.concatenate(rescaled_predictions, axis=0)
        '''

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
        '''
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
        '''

        return self.predictions


    def run(self, data_arr):
        """
        Complete pipeline: preprocess, predict, and filter.

        Parameters
        ----------
        data_array : Preloaded data (look into EMG_Data in emg_lib)

        Returns
        -------
        np.ndarray
            Final filtered torque predictions
        """
        # Load model if not already loaded (Hardcoded Model)
        if self.model is None:
            self.load_model('F:/SMT_MASTERPROJEKT/biosignal_toolbox/src/JTE_Project/offline/saved_online_models/tcn_mtl.keras')

        # Predict
        self.predict(data_arr)

        # Apply post-filtering
        final_predictions = self.apply_post_filter()

        windows = self.preprocessor.emg_data.getWindows()[0]
        print("SHAPE WIN: ", windows.shape)
        windows = windows.reshape(8, 50*18)
        windows = windows[:, :100]

        return final_predictions, windows
