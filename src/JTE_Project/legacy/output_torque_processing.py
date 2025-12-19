import numpy as np
from copy import deepcopy
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.utils import loadConfig, getAbsolutePath

# Process in the same way as in the offline script
'''
For three Quali Objects (in offline script)
1- Creating Quali Objects
2- Lowpass-Filtering the Quali Objects
3- Windowing the Quali Objects
4- Feature Extraction from the Quali Objects
Or 
Downsampling Function

'''

def process_joint_torques(filename_elbow, filename_front, filename_side, cfg):
    """
    Processes torque data for 3 joints (Elbow, Shoulder Front, Shoulder Side)
    from specific filenames using the settings provided in cfg.

    Returns:
        np.array: Combined target features (n_windows, 3)
    """

    # ---------------------------------------------------------
    # 1. Initialization (Create Quali Objects)
    # ---------------------------------------------------------
    print("Creating Quali Data objects...")

    # Elbow
    Quali_Data_Elbow = EEGData(
        format="NumpyQualisys",
        filenames=[filename_elbow],
        data_path=cfg.filepath.data_path,
        f_samp=cfg.preprocess_param.f_samp,
        channel_names=cfg.preprocess_param.channel_names_quali,
        add_marker_channel=True
    )

    # Shoulder Front
    Quali_Data_Front = EEGData(
        format="NumpyQualisys",
        filenames=[filename_front],
        data_path=cfg.filepath.data_path,
        f_samp=cfg.preprocess_param.f_samp,
        channel_names=cfg.preprocess_param.channel_names_quali,
        add_marker_channel=True
    )

    # Shoulder Side
    Quali_Data_Side = EEGData(
        format="NumpyQualisys",
        filenames=[filename_side],
        data_path=cfg.filepath.data_path,
        f_samp=cfg.preprocess_param.f_samp,
        channel_names=cfg.preprocess_param.channel_names_quali,
        add_marker_channel=True
    )

    # ---------------------------------------------------------
    # 2. Low Pass Filtering (Smoothing)
    # ---------------------------------------------------------
    print("Designing and applying Low Pass Filter...")

    # Design the filter (Scipy Butterworth)
    sos_lp = Quali_Data_Elbow.designFilter(
        f_low=cfg.preprocess_param.f_cutoff_sm_lpf,
        order=cfg.preprocess_param.sm_filter_order,
        filter_type="scipy_butter",
        return_type="sos"
    )

    # Apply filter to all three objects
    Quali_Data_Elbow.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)
    Quali_Data_Front.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)
    Quali_Data_Side.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)

    # ---------------------------------------------------------
    # 3. Windowing
    # ---------------------------------------------------------
    print("Windowing the continuous data...")

    # Window Elbow
    Quali_Data_Elbow.windowContinuousData(
        startmarkernumber=1,
        stopmarkernumber=2,
        window_size=cfg.preprocess_param.window_size_y,
        window_step=cfg.preprocess_param.window_step,
        start_index_offset=0,
        start_channel_pick=0,
        end_channel_pick=1,
        return_window_end_indices=True
    )

    # Window Front
    Quali_Data_Front.windowContinuousData(
        startmarkernumber=1,
        stopmarkernumber=2,
        window_size=cfg.preprocess_param.window_size_y,
        window_step=cfg.preprocess_param.window_step,
        start_index_offset=0,
        start_channel_pick=0,
        end_channel_pick=1,
        return_window_end_indices=True
    )

    # Window Side
    Quali_Data_Side.windowContinuousData(
        startmarkernumber=1,
        stopmarkernumber=2,
        window_size=cfg.preprocess_param.window_size_y,
        window_step=cfg.preprocess_param.window_step,
        start_index_offset=0,
        start_channel_pick=0,
        end_channel_pick=1,
        return_window_end_indices=True
    )

    # Note: If necessary, you can add the logic here to trim windows if lengths mismatch
    # (mirrored from the original script's "Window mismatch" section)

    # ---------------------------------------------------------
    # 4. Feature Extraction
    # ---------------------------------------------------------
    print("Extracting features (Timepoints)...")

    window_size_ms = cfg.preprocess_param.window_size_y * 1000 / 500

    # Determine indices based on config
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
        raise ValueError(f"Unknown target_feature_select: {cfg.preprocess_param.target_feature_select}")

    # Extract features for all three
    Quali_Data_Elbow.featureExtractionFromWindows(
        feature_type="timepoints",
        feature_indices_windows=feature_indices_windows_y,
        use_mean=use_mean_bool
    )

    Quali_Data_Front.featureExtractionFromWindows(
        feature_type="timepoints",
        feature_indices_windows=feature_indices_windows_y,
        use_mean=use_mean_bool
    )

    Quali_Data_Side.featureExtractionFromWindows(
        feature_type="timepoints",
        feature_indices_windows=feature_indices_windows_y,
        use_mean=use_mean_bool
    )

    # ---------------------------------------------------------
    # 5. Merging
    # ---------------------------------------------------------
    target_features = np.concatenate([
        Quali_Data_Elbow.getFeatures(),
        Quali_Data_Front.getFeatures(),
        Quali_Data_Side.getFeatures()
    ], axis=1)

    print(f"Processing complete. Final target feature shape: {target_features.shape}")

    return target_features


# --- Usage Example ---
if __name__ == "__main__":
    # Define your specific filenames here
    file_elbow = getAbsolutePath("src/JTE_Project/online/resources/reference_torques/e.npy")
    file_front = getAbsolutePath("src/JTE_Project/online/resources/reference_torques/front.npy")
    file_side = getAbsolutePath("src/JTE_Project/online/resources/reference_torques/side.npy")

    # Make sure cfg is loaded
    config_filename = 'pipeline_jte_bu62d.yaml'
    cfg = loadConfig(filename=config_filename)

    #processed_targets = process_joint_torques(file_elbow, file_front, file_side, cfg)

    import pandas as pd

    emg_file = getAbsolutePath("src/JTE_Project/online/resources/publisher/24072025_BU62D_1100g_complex_2.txt")

    df = pd.read_csv(emg_file, sep=" ", header=None)
    df = df.drop(df.columns[0], axis=1)
    max_rows = df.shape[0] - 1

    print(max_rows)

    path_to_mvc = str(getAbsolutePath("src/JTE_Project/offline/saved_offline_models/channelwise_mvc.npy"))
    channel_normalizer = np.load(path_to_mvc)

    print(channel_normalizer.shape)