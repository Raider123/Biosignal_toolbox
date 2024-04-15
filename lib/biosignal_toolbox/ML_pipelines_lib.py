import scipy.signal as sig
def firstStagePreprocessing(EEG_data, window_size, window_step, windows_selected): 
    
    # window EEG epochs 
    EEG_data.windowEEGEpochs(window_size, window_step)
    
    # window selection 
    if not (windows_selected[0] == "all"): 
        EEG_data.windowSelection(windows_selected)

    #print("windows type: ", EEG_data.windows.dtype)
    print("windows shape first stage", EEG_data.windows.shape)

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
    
def MLPProcessing(EEG_MLP, EEG_freq_MLP, window_labels, feature_indices_windows): 

    # ******** MLP processing *******************
    
    # bandpass filter data 
    
    # wind = sig.windows.kaiser_bessel_derived(M=1000, beta = 600, sym=True)
    # wind_band = wind[225:-225]
    # EEG_MLP.windows[0, 0, :, 0] = EEG_MLP.windows[0, 0, :, 0] *wind_band

    sos = EEG_MLP.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")
    EEG_MLP.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)
    EEG_MLP.cutWindows(n_samples_start = 25, n_samples_end = 25) # try this for reducing artifacts 
    
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
    y_MLP = EEG_MLP.getLabels()
    
    return x_MLP, y_MLP

    
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

def EEGNetProcessing(EEG_EEGNet, window_labels, num_classes): 

    # *********** EEGNet processing *******************

    #wind = sig.windows.kaiser_bessel_derived(M=1000, beta = 600, sym=True)
    #wind_band = wind[225:-225]
    #EEG_EEGNet.windows[0, 0, :, 0] = EEG_EEGNet.windows[0, 0, :, 0] *wind_band

    sos = EEG_EEGNet.designFilter(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")
    EEG_EEGNet.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)

    EEG_EEGNet.cutWindows(n_samples_start = 25, n_samples_end = 25) # try this for reducing artifacts 

    EEG_EEGNet.setWindowLabels(window_labels)
    # reshape windows for net
    
    EEG_EEGNet.reshapeWindowsForCNNnets()
    EEG_EEGNet.labelsToCategorical(num_classes = num_classes) 

    # get train windows 
    x_EEGNet = EEG_EEGNet.getWindows()
    y_EEGNet = EEG_EEGNet.getLabels()


    return x_EEGNet, y_EEGNet

