
import copy
import numpy as np

def firstStageOnlinePreprocessing(EEG_data, window_size, window_step, windows_selected): 
    
    # window EEG epochs 
    EEG_data.windowEEGEpochs(window_size, window_step)
    
    # window selection 
    if not (windows_selected[0] == "all"): 
        EEG_data.windowSelection(windows_selected)

    #print("windows type: ", EEG_data.windows.dtype)
    print("windows shape first stage", EEG_data.windows.shape)

    return EEG_data

def offlinePreprocessingAndFiltering(EEG_data, f_highpass, f_lowpass, marker_number, error_number, channel_list, t1, t2, windows_selected, window_size, window_step, split_train_test_epochs = None, n_epochs = None):

    # do rereferencing and epoching 
    EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=True, f_highpass = f_highpass, f_lowpass= f_lowpass, inverse_keep_channel = True, t1 = t1, t2= t2) 
    
    # window EEG epochs (separate in validation and test if required)
    if (split_train_test_epochs): 
        EEG_data_val, EEG_data_test = EEG_data.splitTrainTestEpochs(n_test_epochs=n_epochs)

        EEG_data_val.windowEEGEpochs(window_size, window_step)
        EEG_data_test.windowEEGEpochs(window_size, window_step)

        # window selection 
        if not (windows_selected[0] == "all"): 
            EEG_data_val.windowSelection(windows_selected)
            EEG_data_test.windowSelection(windows_selected)
        return EEG_data_val, EEG_data_test
    
    else: # if not split just process further 
        EEG_data.windowEEGEpochs(window_size, window_step)
    
        # window selection 
        if not (windows_selected[0] == "all"): 
            EEG_data.windowSelection(windows_selected)

        return EEG_data


def classicFirstStagePreprocessing(EEG_data, window_size, window_step, windows_selected, xd, xd_components): 
    
    # window EEG epochs 
    EEG_data.windowEEGEpochs(window_size, window_step)

    # spatial filter 
    EEG_data.applyxDAWNToWindows(xd, n_components = xd_components)
    
    # window selection 
    if not (windows_selected[0] == "all"): 
        EEG_data.windowSelection(windows_selected)

    #print("windows type: ", EEG_data.windows.dtype)
    print("windows shape first stage", EEG_data.windows.shape)

    return EEG_data

def classicLRPpreprocessing(EEG, window_labels, feature_indices_windows): 

    # ******** preprocessing *******************
    
    #sos = EEG.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")
    #EEG.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)

    #EEG.filterWindows(sos = sos, apply_method = "forward_sos_filter") # forward filter only 
    #EEG.cutWindows(n_samples_start = 25, n_samples_end = 25) # try this for reducing artifacts 

    print(f"window shape after processing:{EEG.getWindows().shape}")
    
    # standardization
    EEG.windowStandardization()
    
    # # specify the window labels (not needed)
    EEG.setWindowLabels(window_labels)

    # time domain features 
    EEG.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)

    # input features network 
    x = EEG.getFeatures()
    y = EEG.getLabels()
    
    return x, y


    
def MLPProcessingOnline(EEG_MLP, EEG_freq_MLP, window_labels, feature_indices_windows): 

    # ******** MLP processing *******************
    
    # bandpass filter data 

    sos = EEG_MLP.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")
    EEG_MLP.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)
    #EEG_MLP.cutWindows(n_samples_start = 25, n_samples_end = 25) # try this for reducing artifacts 
    

    # # specify the window labels (not needed)
    EEG_MLP.setWindowLabels(window_labels)

    
    EEG_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts
    EEG_freq_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts

    # time domain features (MLP)
    EEG_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")
    
    # feauture combination 
    x_freq = EEG_freq_MLP.getFeatures() # get features of freq
    EEG_MLP.addFeatures(x_freq) # add frequency domain features 

    # input features network 
    x_MLP = EEG_MLP.getFeatures()
    y_MLP = EEG_MLP.getLabels()
    
    return x_MLP, y_MLP

def MLPProcessingOnlineFilterNet(EEG_MLP, EEG_freq_MLP, window_labels, feature_indices_windows, filter_model): 

    # ******** MLP processing *******************
    
    # # specify the window labels
    EEG_MLP.setWindowLabels(window_labels)
    
    # reshape for filter Net 
    EEG_MLP.reshapeWindowsForCNNnets() # reshape for CNN net
    EEG_freq_MLP.reshapeWindowsForCNNnets() # reshape for CNN net

    # apply fitler model 
    filter_model.predict(data = EEG_MLP.getWindows(), classification = False, show_pred_time = True)

    filtered_windows = filter_model.getPredictionScores() # has now shape (None, Channels, Sampels, 1)
    print(f"type after filtering: ", type(filtered_windows))
    print(f"filtered windows shape: {filtered_windows.shape}")

    EEG_MLP.setWindows(filtered_windows) # set filtered windows again 

    # cut windows to remove artifacts 
    EEG_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts 
    EEG_freq_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts 

    print(f"window shape after processing {EEG_freq_MLP.getWindows().shape}")

    # time domain features (MLP)
    EEG_MLP.featureExtractionReshapedWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_freq_MLP.featureExtractionReshapedWindows(feature_type = "freqBandPower")

    # feauture combination 
    x_freq = EEG_freq_MLP.getFeatures() # get features of freq
    EEG_MLP.addFeatures(x_freq) # add frequency domain features 

    # input features network 
    x_MLP = EEG_freq_MLP.getFeatures()
    y_MLP = EEG_MLP.getLabels()

    print("x shape", x_MLP.shape)
    print("y shape:", y_MLP.shape)
    
    return x_MLP, y_MLP

def MLPProcessingOffline(EEG_MLP, EEG_freq_MLP, window_labels, feature_indices_windows): 

    # ******** MLP processing *******************
    
    # depends on comparison but not required 
    # # specify the window labels (not needed)
    EEG_MLP.setWindowLabels(window_labels)

    # reshape for filter Net 
    EEG_MLP.reshapeWindowsForCNNnets() # reshape for CNN net
    EEG_freq_MLP.reshapeWindowsForCNNnets() # reshape for CNN net

    EEG_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts 
    EEG_freq_MLP.cutWindows(n_samples_start = 75, n_samples_end = 25)

    
    # time domain features (MLP)
    EEG_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")
    

    # feauture combination 
    x_freq = EEG_freq_MLP.getFeatures() # get features of freq
    EEG_MLP.addFeatures(x_freq) # add frequency domain features 

    # input features network 
    x_MLP = EEG_MLP.getFeatures()
    y_MLP = EEG_MLP.getLabels()
    
    return x_MLP, y_MLP

    
def EEGNetProcessingOnline(EEG_EEGNet, window_labels, num_classes): 

    # *********** EEGNet processing *******************

    sos = EEG_EEGNet.designFilter(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")
    EEG_EEGNet.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)

    #EEG_EEGNet.cutWindows(n_samples_start = 25, n_samples_end = 25) # try this for reducing artifacts 

    EEG_EEGNet.setWindowLabels(window_labels)
    # reshape windows for net
    
    EEG_EEGNet.reshapeWindowsForCNNnets()
    EEG_EEGNet.labelsToCategorical(num_classes = num_classes) 

    # cut windows accordingly 
    EEG_EEGNet.cutWindows(n_samples_start = 75, n_samples_end = 25)
    
    # get train windows 
    x_EEGNet = EEG_EEGNet.getWindows()
    y_EEGNet = EEG_EEGNet.getLabels()

    return x_EEGNet, y_EEGNet


def EEGNetProcessingOnlineFilterNet(EEG_EEGNet, window_labels, num_classes, filter_model): 
    
    EEG_EEGNet.setWindowLabels(window_labels)
    # reshape windows for net
    
    EEG_EEGNet.reshapeWindowsForCNNnets() # reshape for CNN net 
    print(f"shape of windows: {EEG_EEGNet.getWindows().shape}")

    filter_model.predict(data = EEG_EEGNet.getWindows(), classification = False, show_pred_time = True)

    filtered_windows = filter_model.getPredictionScores() # has now shape (None, Channels, Sampels, 1)
    print(f"type after filtering: ", type(filtered_windows))
    print(f"filtered windows shape: {filtered_windows.shape}")

    EEG_EEGNet.setWindows(filtered_windows) # set filtered windows again 
    EEG_EEGNet.cutWindows(n_samples_start = 75, n_samples_end = 25)

    EEG_EEGNet.labelsToCategorical(num_classes = num_classes) 
    
    # get train windows 
    x_EEGNet = EEG_EEGNet.getWindows()
    y_EEGNet = EEG_EEGNet.getLabels()


    return x_EEGNet, y_EEGNet


def EEGNetProcessingOffline(EEG_EEGNet, window_labels, num_classes): 

    # *********** EEGNet processing *******************

    # depends on comparison but not required 
    EEG_EEGNet.cutWindows(n_samples_start = 75, n_samples_end = 25) # try this for reducing artifacts 

    EEG_EEGNet.setWindowLabels(window_labels)
    # reshape windows for net
    
    EEG_EEGNet.reshapeWindowsForCNNnets()
    EEG_EEGNet.labelsToCategorical(num_classes = num_classes) 
    
    # get train windows 
    x_EEGNet = EEG_EEGNet.getWindows()
    y_EEGNet = EEG_EEGNet.getLabels()


    return x_EEGNet, y_EEGNet