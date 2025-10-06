from process_emg import OnlineEMGPredictor
import numpy as np

# Initialize the online predictor
predictor = OnlineEMGPredictor(config_filename='pipeline_mlp_to_cnn.yaml')

def loadMiniANTEMGData(file_str, f_samp):
    """
    This function loads the EMG data recorded from the ANT EMG system (as .txt file, recorded via SDK)

    Parameters
    ----------
    file_str : str
        The file to load
    f_samp : int
        The sampling frequency in Hz

    Returns
    -------
    tuple
        emg_data : numpy array
            The EMG data with shape (n_samples, n_channel)
        time_axis : numpy array
            The time axis of the EMG data with shape (n_samples,)
    """

    # seperate between data, meta and channel names
    emg_data_raw = np.loadtxt(emg_file)  # at least th
    emg_data = emg_data_raw[:-1, :-2]
    time_axis = np.arange(0, (len(emg_data) / f_samp), step=1 / f_samp)

    emg_data = emg_data.T  # transpose for timeseries data format

    return emg_data, time_axis

# Path to single EMG file
emg_file = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/data/jte/emg/BU62D/test500.txt"

emg_tuple = loadMiniANTEMGData(file_str=emg_file, f_samp=500)

# Run the complete pipeline
try:
    predictions = predictor.run(emg_tuple)

    # Display results
    print("=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Shape: {predictions.shape}")

except Exception as e:
    print(f"Error during prediction: {e}")
    raise

