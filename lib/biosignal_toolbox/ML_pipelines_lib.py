def firstStagePreprocessing(EEG_data, window_size, window_step, windows_selected): 
    
    # window EEG epochs 
    EEG_data.windowEEGEpochs(window_size, window_step)

    # window selection 
    if not (windows_selected[0] == "all"): 
        EEG_data.windowSelection(windows_selected)

    #print("windows type: ", EEG_data.windows.dtype)
    print("windows shape", EEG_data.windows.shape)

    return EEG_data

def MLPProcessing(EEG_MLP, EEG_freq_MLP, window_labels, feature_indices_windows): 

    # ******** MLP processing *******************

    # bandpass filter data 
    #EEG_MLP.WindowMedianCorrection(ratio_len = 0.1)
    EEG_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")
    
    #EEG_MLP.windowStandardization()
    
    # # specify the window labels (not needed)
    EEG_MLP.setWindowLabels(window_labels)

    # time domain features (MLP)
    EEG_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")

    # feauture combination 
    x_freq = EEG_freq_MLP.getFeatures() # get features of freq
    EEG_MLP.addFeatures(x_freq) # add frequency domain features 

    # input features network 
    x_MLP = EEG_MLP.getFeatures()
    y_MLP = EEG_MLP.getTrainLabels()

    return x_MLP, y_MLP

def EEGNetProcessing(EEG_EEGNet, window_labels, num_classes): 

    # *********** EEGNet processing *******************

    #EEG_EEGNet.FilterWindows(filter_type = "dc_removal", alpha = 0.8)
    #EEG_EEGNet.WindowMedianCorrection(ratio_len = 0.1)
    EEG_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")

    EEG_EEGNet.setWindowLabels(window_labels)
    # reshape windows for net
    
    EEG_EEGNet.reshapeWindowsForCNNnets()
    EEG_EEGNet.labelsToCategorical(num_classes = num_classes) 

    # get train windows 
    x_EEGNet = EEG_EEGNet.getWindows()
    y_EEGNet = EEG_EEGNet.getTrainLabels()


    return x_EEGNet, y_EEGNet

