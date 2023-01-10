# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.utils import shuffle


# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

def loadBrainproductsData(dataset_list): 

    """
    This function can be used for loading one or more datasets in brainproducts format.
    Arguments:
        dataset_list: A list of strings with filenames of the datasets to be loaded.

    Returns:
        raw: An mne object with the loaded (concatenated) dataset(s). 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """
    
    if (len(dataset_list) > 1): 
        raw_list = []
        for dataset in dataset_list: 
            raw1 = mne.io.read_raw_brainvision(dataset, preload = True, verbose = False)
            raw_list.append(raw1)
        raw = mne.concatenate_raws(raw_list)
    else: 
        raw = mne.io.read_raw_brainvision(dataset_list[0], preload = True, verbose = False)

    return raw

def create_acticap_montage(plot_montage): 

    """
    This function can be used to create an acticap montage (used by e.g. LiveAmp64). The montage was created based on the acticap manual and an easycap template provided by mne.
    Arguments:
        plot_montage: A boolean flag if the montage info should be shown or not. if set to True, the montage will be shown. 

    Returns:
        montage: An mne montage object, that was created for the acticap layout. 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """
    
    montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095) 
    #show easy cap Montage  

    montage.rename_channels({'Cz' : 'CZ','Pz' : 'PZ','Fz' : 'FZ','CPz' : 'CPZ', 'Fp1': 'FP1','Fp2': 'FP2','Oz': 'OZ','POz': 'POZ'}, allow_duplicates=False) # maybe change this ? 
    exclude_list = np.array(['O10','Fpz', 'Iz', 'F9', 'F10', 'P9', 'P10', 'O9', 'FCz', 'AFz']) # may change this ? 
    
    # get params of structure 
    easy_cap_ch_names = montage.ch_names
    easy_cap_dig = montage.dig
    dev_head = montage.dev_head_t

    # find indizes to remove channels not in ActiCap 
    indizes_to_removing_channel = []
    for index in range(0, len(exclude_list)):
        for index1 in range(0, len(easy_cap_ch_names)):
            a = exclude_list[index]
            b = easy_cap_ch_names[index1]

            if(a == b):
                indizes_to_removing_channel.append(index1)

    # seperate Standard digs and EEG digs 
    easy_cap_dig_standard = easy_cap_dig[0:3]
    easy_cap_dig_eeg = easy_cap_dig[3:]

    #delete Channels not there for acticap
    indizes_to_removing_channel_sorted = sorted(indizes_to_removing_channel, reverse= True)
    for indizes in indizes_to_removing_channel_sorted:
        #print('Indizes', indizes-count)
        del easy_cap_ch_names[indizes]
        del easy_cap_dig_eeg[indizes]

    # Adapted channel names and digitazation for acticap montage 
    easy_cap_dig_adapted = easy_cap_dig_standard.copy() # only head digits etc. 
    easy_cap_dig_adapted.extend(easy_cap_dig_eeg) # append selected channel digits 
    easy_cap_ch_names_adapted = easy_cap_ch_names.copy()

    #create Montage 
    #PlotMontage = True
    acti_cap_montage = mne.channels.DigMontage(dev_head_t=dev_head ,dig=easy_cap_dig_adapted, ch_names=easy_cap_ch_names_adapted)
    if(plot_montage == True): 
        acti_cap_montage.plot()
        
    return acti_cap_montage


def topoplot(mean_epochs, time_axis_eeg_epoch, mne_obj, start_time, step_time, title_str, min_val, max_val, f_samp_eeg): 

    """
    This function creates and showes an topoplot at different points in time. 
    Arguments:
        mean_epochs: The average epochs over all trials in numpy format. Shape should be (channel, sampels). 
        time_axis_eeg_epoch: The time axis of the EEG-epochs as one dimensional numpy array. 
        mne_obj: The mne object of the dataset from which the information is used for the plot (e.g. channel names). 
        start_time: The timepoint of the first topoplot in ms. 
        step_time: The timestep between each generated topoplots, i.e the time resolution of the plots. 
        title_str: The title of the plot as string. 
        min_val: The minimum Voltage in the color scale. 
        max_val: The maximum Voltage in the color scale. 
        f_samp_eeg: Sampling rate of the EEG-data in Hz. 

    Returns:
        -
    
    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """

    
    #topoplot at different times 
    n,m = mean_epochs.shape
    start_idx = (start_time/1000)*f_samp_eeg
    step_idx = (step_time/1000)* f_samp_eeg

    indices_of_topoplot = np.arange(start_idx, m, step = step_idx).astype(int) # 22 er steps 
    mean_epochs.astype(float)

    count = 0
    fig, ax = plt.subplots(nrows=len(indices_of_topoplot), figsize=(8, 20), gridspec_kw=dict(top=0.9),sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5)
    for index in indices_of_topoplot: 
        cmap = 'bwr'
        im, cn = mne.viz.plot_topomap(mean_epochs[:,index], mne_obj.info, cmap = cmap, axes = ax[count], show = False, image_interp = 'bicubic',extrapolate='local',vmin = min_val, vmax = max_val)
        str_time = str(round(time_axis_eeg_epoch[index],3))
        ax[count].set_title(title_str+str_time+" s", color='black', fontsize=12)
        cbar =fig.colorbar(im, ax = ax[count], orientation="vertical", pad = 0.15)
        cbar.set_label("in uV")
        count = count+1
    plt.show()


def getKerasPredictionResultsLRP(model, epochs, n_samp_features): 

    """
    This function creates and showes an topoplot at different points in time. 
    Arguments:
        model: The keras model object. 
        epochs: The EEG-epochs as numpy array with shape: (n_epochs, n_channels, n_sampels)
        n_samp_features: Number of sampels that are used as features for both classes, currently the last n_samp_features datapoints are consideres as erp class and the remaining are from noerp class. 


    Returns:
        predicted_labels: The predicted labels of the classifier as float values (0.0 noerp or 1.0 erp)
        true_labels: The true labels in respect to the number of n_samp_features as erp labels (-n_samp_features to time 0 as erp labelled points)
        trial_prediction: The prediction scores of each classified datapoint. 
    
    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """

    true_labels = []
    predicted_labels = []
    prediction_scores = []

    for trials in epochs: 
        # true labels single trial 
        erp_labels = np.zeros((trials.shape[1]))
        erp_labels[-n_samp_features:] = 1.0
        true_labels.append(erp_labels[:,])
        # single trial predictions 
        trial_prediction = model.predict(trials.T)
        trial_label = [0 if score <0.5 else 1 for score in trial_prediction]
        trial_label = np.array(trial_label)
        predicted_labels.append(trial_label)
        prediction_scores.append(trial_prediction)
        
    #flatten the trial labels
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    prediction_scores = np.array(prediction_scores)

    return predicted_labels, true_labels, prediction_scores


def rereferencingEpoching(raw, marker_number, error_number,channel_list, inverse_keep_channel, reref_channels, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline): 
    
    """
    Apply rereferencing and epoching with given parameters and filters to an raw mne instance. 

    Arguments:
        raw: The created mne object. 
        marker_number: The markernumber of the used event for creating the epochs. 
        error_number: The markernumber of trials with an error that are excluded from the evaluation. 
        channel_list: A list of EEG-channels that are either kept or dropped from evaluation depending on the "inverse_keep_channel" flag. 
        inverse_keep_channel: If False, all channel in "channel_list" are kept, otherwise the specified channel are dropped. 
        reref_channels: A list of channels that are used for rereferencing. If the list is empty, the original ref-channel is used, if ["average"] is passed an average reference is applied.
        apply_filter: Boolean flag that should be True if a filter should be applied. 
        f_highpass: The highpass cutoff frequency in Hz (only used when apply_filter is True). 
        f_lowpass: The lowpass cutoff frequency in Hz (only used when apply_filter is True). 
        event_id_used: The created mne event id (e.g. {"movement_onset": marker_number}). 
        t1: Start time of the epochs in ms.  
        t2: End time of the epochs in ms. 
        f_samp_eeg: Sampling rate of the EEG-signals in Hz. 
        apply_baseline_correction: Boolean flag that is set to True if baseline correction should be applied (mean value between t0_baseline and t1_baseline is used as correction). 
        t0_baseline: Specified start time for the baseline correction. 
        t1_baseline: Specified end time for the baseline correction. 


    Returns:
        erp_epochs: The epochs of the erp analysis as numpy array with shape: (n_epochs, n_channel, n_samples)
        erp_epoch_obj: The erp epochs object created by mne. 
        time_axis_eeg_batch: The created time axis as one dimensional numpy array. 
        remaining_eeg_channel_names: A list of EEG-channels that are included in the analysis (in the erp_epochs array). 
        filtered_eeg_rereferenced: Manipulated instance of an mne raw object (after filtering and channel selection). 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """

    
    # rereferencing 
    rereferenced_eeg_raw = raw.copy()
    
    if not (reref_channels):
        print("no reref channels specified, using original ref")
        print("")
    else: 
        if (reref_channels[0] == "average"):
            print("using average reference over all electrodes")
            print("")
            rereferenced_eeg_raw, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw, ref_channels='average' ,copy=True)
        else:
            print("using custom electrodes for rereferencing")
            rereferenced_eeg_raw, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw, ref_channels=reref_channels ,copy=True)
    
    raw_eeg_rereferenced = rereferenced_eeg_raw.copy()
    #drop channel

    #apply filter 
    if (apply_filter): 
        filtered_eeg_rereferenced = raw_eeg_rereferenced.filter(f_highpass,f_lowpass)
    else: 
        filtered_eeg_rereferenced = raw_eeg_rereferenced
    
    #extract events 
    plot_events, plot_event_dict = mne.events_from_annotations(filtered_eeg_rereferenced)
    plot_onset_indices = np.where(plot_events[:,2] == marker_number)[0] # S100 marker is leaving plate 
    exclude_indices = np.where(plot_events[:,2] == error_number)[0] # S3 marker should be excluded 
    
    plot_onset_indices = plot_onset_indices[0:-1] # cut of last movement, might be after experiment
    include_mask = np.ones(plot_onset_indices.shape)

    #search for correct indizes without S3 errors 
    for i in range(0, len(plot_onset_indices)): 
        for j in range(0, len(exclude_indices)):
            if ((exclude_indices[j]-1) == plot_onset_indices[i] or (exclude_indices[j]-2) == plot_onset_indices[i]):
                include_mask[i] = 0

    #Epochs with an S3 error are excluded 
    plot_onset_indices_correct = plot_onset_indices[include_mask.astype(bool)]

    #create new event matrix 
    used_plot_events = np.zeros((len(plot_onset_indices_correct),3))
    used_plot_events = plot_events[plot_onset_indices_correct,:]

    #eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
    if not(channel_list): 
        if(apply_baseline_correction): 
            eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used, tmin=t1, baseline=(t0_baseline, t1_baseline), tmax=t2, preload=True, reject_by_annotation = True)
        else: 
            eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used, tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
            
    else: #drop specified channels if False 

        # keep or drop specified channels 
        if(inverse_keep_channel == True): 
            filtered_eeg_rereferenced.drop_channels(channel_list)
        else: 
            filtered_eeg_rereferenced.pick_channels(channel_list)

        if(apply_baseline_correction): 
            eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, tmax=t2, baseline=(t0_baseline, t1_baseline), preload=True, reject_by_annotation = True)
        else: 
            eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, tmax=t2, baseline=None, preload=True, reject_by_annotation = True)
    
    # Get remaining channel names  
    remaining_eeg_channel_names = filtered_eeg_rereferenced.ch_names
    
    erp_epoch_obj = eeg_epochs.copy() # the object of epochs from mne 

    #get data out as numpy array for further processing 
    erp_epochs = eeg_epochs.get_data() 
    
    #generate a time axis for the epochs 
    time_axis_eeg_batch = np.arange(t1,t2+1/f_samp_eeg, step = 1/f_samp_eeg) #build time axis (epoch)

    #return everything needed for further processing 
    return erp_epochs, erp_epoch_obj, time_axis_eeg_batch, remaining_eeg_channel_names, filtered_eeg_rereferenced # shape of epochs: (epochs, channel, samples)


def windowEEGEpochs(epochs, f_samp_eeg, window_size = 1000, window_step = 50, with_channel_dim = True): 

    """
    This function cuts (overlapping) windows from continues EEG-signals (currently only for postprocessing without channel dimension). 

    Arguments:
        epochs: The EEG-epochs as numpy array with shape: (n_epochs, n_channel, n_samples) or for postprocessing (with_channel_dim = False) with shape:(n_trials, n_sampels). 
        window_size: The size of the windows in ms to be cutout (default: 1000). 
        window_step: The stepsize of the sliding window (sliding step) in ms (default: 50)

    Returns:
        wind_arr: Numpy array with windowed EEG-data with shape (n_trials, n_channel, n_sampels, n_windows) if with_channel_dim = True or (n_trials, n_sampels, n_windows) for postprocessing (no channel dim)
        num_of_windows: The total number of windows that are created. 
        wind_names: A list of the window names according to the pySPACE naming of window definitions.  
    
    Meta information: 
        Author: Niklas Kueper 
        Last changed: 10.01.2023 (by Niklas Kueper)
    """

    window_size_samp = int((window_size/1000) * f_samp_eeg) 
    window_step_samp = int((window_step/1000) * f_samp_eeg) 

    if (with_channel_dim == False): 

        epochs_arr_cut = epochs[:, 1:]
        num_of_windows =  int((epochs_arr_cut.shape[1]-window_size_samp)/window_step_samp)+1

        #init window arrays with shape (trials, sampel of window, windownumber)
        wind_arr = np.zeros((epochs_arr_cut.shape[0], window_size_samp, num_of_windows))
        wind_names = []

        for win_nr in range(0, num_of_windows): 
            wind_start_idx = win_nr*window_step_samp
            wind_end_idx = window_size_samp+wind_start_idx
            wind_name = "bis"+str(int((((epochs_arr_cut.shape[1]-wind_end_idx)*-1)/f_samp_eeg) *1000))
            
            wind_names.append(wind_name) # a list of all window names
            #create arrays with cutted 
            wind_arr[:, :, win_nr] = win_nr
            wind_arr[:, :, win_nr] = epochs_arr_cut[:, wind_start_idx:wind_end_idx]

    else: # with channel dimension  (n_epochs, n_channel, n_samples)
        
        epochs_arr_cut = epochs[:, :, 1:]
        num_of_windows =  int((epochs_arr_cut.shape[2]-window_size_samp)/window_step_samp)+1

        #init window arrays with shape (trials, channel sampels of window, windownumber)
        wind_arr = np.zeros((epochs_arr_cut.shape[0], epochs_arr_cut.shape[1], window_size_samp, num_of_windows))
        wind_names = []


        for win_nr in range(0, num_of_windows): 

            wind_start_idx = win_nr*window_step_samp
            wind_end_idx = window_size_samp+wind_start_idx
            wind_name = "bis"+str(int((((epochs_arr_cut.shape[2]-wind_end_idx)*-1)/f_samp_eeg) *1000))
            
            wind_names.append(wind_name) # a list of all window names
            #create arrays with cutted 
            wind_arr[:, :, :, win_nr] = win_nr
            wind_arr[:, :, :, win_nr] = epochs_arr_cut[:, :, wind_start_idx:wind_end_idx]

    return wind_arr, num_of_windows, wind_names


def calcTestAccAndRates(prediction_labels, true_labels):

    """
    Get metrics from classification output of the test data. Currently the accuracy, balanced accuracy,  tnr and tpr are calculated. 

    Arguments:
        prediction_labels: The predicted labels as one dimensional numpy array (flatten the array if it has more dimensions).
        true_labels: The true labels as one dimensional numpy array (flatten the array if it has more dimensions). 

    Returns:
        tnr: True negative rate 
        tpr: True positive rate
        acc: Accuracy
        ba: Balanced accuracy
    
    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """

    prediction_labels = prediction_labels.astype(int)
    true_labels = true_labels.astype(int)
    
    negatives = np.sum(np.abs(true_labels-1))
    positives = np.sum(true_labels)
    
    n = len(true_labels)
    tps= 0 
    tns = 0
    for index in range(0, n): 
        if (prediction_labels[index] == 0 and true_labels[index] == 0):
            tns = tns+1
        elif (prediction_labels[index] == 1 and true_labels[index] == 1): 
            tps = tps+1
    tnr = tns/negatives
    tpr = tps/positives
    acc = ((tns+tps)/n) #in percent
    ba = (tnr+tpr)/2

    return tnr, tpr, acc, ba

def applyRelabelling(predicted_labels, determine_labels, searching_bounds):

    """
    Apply the relabelling method to the classification output in order to get the "true ground truth" labels. This function should be carefully used since it creates new ground truth labels for the evaluation of the classifier! 

    Arguments:
        prediction_labels: The predicted labels as one dimensional numpy array (flatten the array if it has more dimensions).
        determine_labels: The amount of negative classes that are counted from the right side (end of each epoch/trial) to specify the "label change point". 
        searching_bounds: A list with boundaries ([lower bound, upper bound]) in which the label change point for the relabelling is searched (e.g. for LRP the numbers of the windows for -1000 ms and 0 ms). 

    Returns:
        new_true_labels: A numpy array (shape: (n_trials, n_sampels)) containing the new ground truth labels (0.0 neg class; 1.0 pos class). 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 28.11.2022 (by Niklas Kueper)
    """ 


    new_true_labels = np.zeros(predicted_labels.shape)
    new_true_labels[:, -1] = 1.0 # last label has to be positive class 

    #predicted_labels_cut = predicted_labels[:, searching_bounds[0]:searching_bounds[1]] # cut to searching bounds where the label change point is searched 
    
    trial = 0 

    for predicted_trial_label in predicted_labels: 
        # reset for every trial 
        neg_class_counter = 0
        index =  searching_bounds[1]-1 # do this like that for now but change afterwards,  upper bound could lay in the middle ! 

        while index > searching_bounds[0]: # go from the back to front until the minimum point 
            # count number of neg classes 
            if (predicted_trial_label[index] == 0): 
                neg_class_counter = neg_class_counter+1
            else: 
                neg_class_counter = 0
            
            if(neg_class_counter >= determine_labels): 
                new_true_labels[trial, index+determine_labels:] = 1.0 # set true labels after label change point
                break # stop when change point was found
            
            index = index -1

        # no consecutive determine label numbers of neg class found 
        if(index <= searching_bounds[0] and neg_class_counter < determine_labels): 
            new_true_labels[trial, searching_bounds[0]:] = 1.0

        trial = trial +1

    return new_true_labels


def calcWindowMetrics(wind_arr, evaluation_time_per_window, window_step, f_samp_eeg, n_samp_features, use_relabelling, determine_labels, searching_bounds): 

    """
    Calculate Metrics of a window wise classification output. 

    Arguments:
        wind_arr: The windowed class predictions as numpy array with shape (n_trials, n_sampels, n_windows). 
        evaluation_time_per_window: The time in ms at the end of each window for which the window metric is calculated (-evaluation_time_per_window to 0 ms are used). 
        window_step: The sliding step size of the windows in ms (standard 50 ms). 
        f_samp_eeg: The sampling rate of the EEG-data in Hz. 
        n_samp_features: The number of sampels that are used as features (-n_samp_features:0 of each epoch). 
        use_relabelling: If this flag is set to True, the relabelling method is applied to calculate the metrics. This method should be treated with care since it effects the classification performance!
        determine_labels: The number of consecutive negative classes that are counted when estimating the label change point of both classes. 
        searching_bounds: A list of the lower and upper bound of window numbers ([lower bound, upper bound]) where the label change point is searched. 

    Returns:
        tnr: True negative rate 
        tpr: True positive rate 
        acc: Accuracy 
        ba: Balanced accuracy 
        window_predictions: The prediction values (0 - 1) for all windows and trials 
        window_eval_true_labels: The true labels that are specified or the new true labels when relabelling is used. 

    Meta information: 
        Author: Niklas Kueper 
        Last changed: 29.11.2022 (by Niklas Kueper)
    """ 

    window_step_samp = int((window_step/1000) * f_samp_eeg) 
    evaluation_samp_per_window = int((evaluation_time_per_window/1000) * f_samp_eeg) 
    window_predictions_samp = np.sum(wind_arr[:, -evaluation_samp_per_window:, :], axis = 1)

    # convert the label of each sampels to windowwise labels 
    window_predictions = (window_predictions_samp > evaluation_samp_per_window/2).astype(float)

    window_eval_true_labels = np.zeros(window_predictions.shape)
    num_erp_windows = int(np.round(n_samp_features/window_step_samp)) 
    trial_erp_label = window_eval_true_labels[0, :] 
    trial_erp_label[-num_erp_windows:] = 1.0

    for trial_nr in range(0, window_eval_true_labels.shape[0]): 
        window_eval_true_labels[trial_nr, :] = trial_erp_label

    #get metrics for window evaluation 
    if (use_relabelling == False): 
        tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), window_eval_true_labels.flatten())
    else: 
        relabelled_true_labels = applyRelabelling(window_predictions, determine_labels, searching_bounds)
        tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), relabelled_true_labels.flatten())
        window_eval_true_labels = relabelled_true_labels # return the relabelled true labels instead 

    return tnr, tpr, acc, ba, window_predictions, window_eval_true_labels


def timeDomainFeaturesFromEpochs(erp_epochs, time_axis_eeg_batch, n_samp_features, use_continues_sampels, shuffle_data, t1_time, t2_time): 

    """
    This function 
    Arguments:
        erp_epochs: The epochs of the erp analysis as numpy array with shape: (n_epochs, n_channel, n_samples). 
        time_axis_eeg_batch: The time axis of the EEG-epochs as one dimensional numpy array. 
        n_samp_features: The number of sampels that are used as features (-n_samp_features:0 of each epoch). 
        use_continues_sampels: If boolean flag is set to True, continous sampels with a stepsize are extracted from the specified negative class (t1_time to t2_time, balanced). If False, all sampels between t1_time and t2_time are used as features for the negative class. 
        shuffle_data: If boolean flag is set to True, the features are randomly shuffled. 
        t1_time: The start time where sampels of the negative class are extracted as features from EEG epochs. 
        t2_time: The end time where sampels of the negative class are extracted as features from EEG epochs. 
        
    Returns:
        x: The time domain features as a numpy array with shape (n_sampels, n_channel/features). 
        y: The encoded labels of both classes (binary classification) as numpy array with shape (n_sampels, )
    
    Meta information: 
        Author: Niklas Kueper 
        Last changed: 29.11.2022 (by Niklas Kueper)
    """
    tensor_shape = erp_epochs.shape # shape is (trials, channel, sampels)

    n_start = n_samp_features # number of samples to use as features 
    n_end = 1
    erp_idx_1 = tensor_shape[2]-n_start-1
    erp_idx_2 = tensor_shape[2]-n_end
    no_erp_idx_1 = n_start

    # convert from ms to seconds 
    t1_time = t1_time/1000
    t2_time = t2_time/1000

    if (t1_time < 0): 
        for index in range(0, len(time_axis_eeg_batch)): 
            if(time_axis_eeg_batch[index] <= t1_time and time_axis_eeg_batch[index+1] >= t1_time): 
                print("t1 set")
                t1 = index 
                break

    if (t2_time < 0): 
        for index in range(0, len(time_axis_eeg_batch)): 
            if(time_axis_eeg_batch[index] <= t2_time and time_axis_eeg_batch[index+1] >= t2_time): 
                t2 = index 
                break


    print("t1: ", t1)
    print("t2: ", t2)

    # print("erp indizes are: ", erp_idx_1, erp_idx_2)
    # print("no_erp indizes are: ", t1, no_erp_idx_1+t1)
    
    #seperate the samples of both classes 
    if(use_continues_sampels): # here no erp sampels are cut out at even spaced steps to erp class
        erp_sampels = erp_epochs[:, :, erp_idx_1:erp_idx_2] # shape (trials, channel, sampels)
        num_erp_samps = erp_sampels.shape[2]
        #print((erp_idx_1/num_erp_samps))
        stepsize_no_erp = int(((t2-t1)/num_erp_samps))
        no_erp_ind = np.arange(t1, t2, step = stepsize_no_erp) 
        
        no_erp_indices = no_erp_ind[no_erp_ind.shape[0]-num_erp_samps:] # make the balance !
        no_erp_sampels = erp_epochs[:, :, no_erp_indices]
    else: 
        erp_sampels = erp_epochs[:, :, erp_idx_1:erp_idx_2] # shape (trials, channel, sampels)
        no_erp_sampels = erp_epochs[:, :, t1:no_erp_idx_1+t1]
        
    #output the time values of the cutted slices: 
    
    erp_times = np.array([time_axis_eeg_batch[erp_idx_1] *1000,time_axis_eeg_batch[erp_idx_2]*1000])
    if(use_continues_sampels):
        no_erp_times = np.array([time_axis_eeg_batch[no_erp_indices[0]]*1000,time_axis_eeg_batch[no_erp_indices[-1]]*1000])
    else: 
        no_erp_times = np.array([time_axis_eeg_batch[t1]*1000,time_axis_eeg_batch[no_erp_idx_1+t1]*1000])

    print("erp times in ms: ", (erp_times).astype(dtype=np.int32))
    print("no_erp times in ms: ", (no_erp_times.astype(dtype=np.int32)))

    if(use_continues_sampels): 
        print("stepsize for no erp is (samples): ", stepsize_no_erp)

    # init arrays for both classes 
    erp_shaped = np.zeros((erp_sampels.shape[0]*erp_sampels.shape[2], erp_sampels.shape[1]))
    no_erp_shaped = np.zeros((no_erp_sampels.shape[0]*no_erp_sampels.shape[2], no_erp_sampels.shape[1]))

    # flatten sampels and trials to one dim 
    for channel in range(0, tensor_shape[1]): 
        erp_shaped[:, channel] = erp_sampels[:,channel,:].flatten()
        no_erp_shaped[:, channel] = no_erp_sampels[:,channel,:].flatten()

    #merge data together 
    x = np.concatenate((erp_shaped, no_erp_shaped), axis=0).astype(dtype = np.float64)

    #create label encoding 
    y = np.zeros((x.shape[0],1)).astype(dtype=np.float64)
    y[0:erp_shaped.shape[0]] = 1.0
    
    #concatenate data for shuffling 
    if (shuffle_data): 
        x_y_concat = np.concatenate((x, y), axis=1)
        x_y_concat_shuffle = shuffle(x_y_concat)
        x = x_y_concat_shuffle[:, 0:x_y_concat_shuffle.shape[1]-1]
        y = x_y_concat_shuffle[:, x_y_concat_shuffle.shape[1]-1]
    
    return x, y
