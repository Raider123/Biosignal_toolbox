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


def getPredictionResults(model, epochs, n_samp_features): 
    true_labels = []
    predicted_labels = []

    for trials in epochs: 
        # true labels single trial 
        lrp_labels = np.zeros((trials.shape[1]))
        lrp_labels[-n_samp_features:] = 1.0
        true_labels.append(lrp_labels[:,])
        # single trial predictions 
        trial_prediction = model.predict(trials.T)
        trial_label = [0 if score <0.5 else 1 for score in trial_prediction]
        trial_label = np.array(trial_label)
        predicted_labels.append(trial_label)
        
    #flatten the trial labels
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    return predicted_labels, true_labels


def rereferencingEpoching(raw, onset_number, error_number,channel_list, inverse_keep_channel, reref_channels, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline): 
    # rereferencing 
    rereferenced_eeg_raw = raw.copy()
    rereferenced_eeg_raw = rereferenced_eeg_raw.drop_channels(['x_dir', 'y_dir', 'z_dir'])
    
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
            rereferenced_EEG_raw, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw, ref_channels=reref_channels ,copy=True)
    
    raw_eeg_rereferenced = rereferenced_eeg_raw.copy()
    #drop channel

    #apply filter 
    if (apply_filter): 
        filtered_eeg_rereferenced = raw_eeg_rereferenced.filter(f_highpass,f_lowpass)
    else: 
        filtered_eeg_rereferenced = raw_eeg_rereferenced
    
    #extract events 
    plot_events, plot_event_dict = mne.events_from_annotations(filtered_eeg_rereferenced)
    plot_onset_indices = np.where(plot_events[:,2] == onset_number)[0] # S100 marker is leaving plate 
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
    
    lrp_epoch_obj = eeg_epochs.copy() # the object of epochs from mne 

    #get data out as numpy array for further processing 
    lrp_epochs = eeg_epochs.get_data() 
    
    #generate a time axis for the epochs 
    time_axis_eeg_batch = np.arange(t1,t2+1/f_samp_eeg, step = 1/f_samp_eeg) #build time axis (epoch)

    #return everything needed for further processing 
    return lrp_epochs, lrp_epoch_obj, time_axis_eeg_batch, remaining_eeg_channel_names, filtered_eeg_rereferenced # shape of epochs: (epochs, channel, samples)


def windowEEGEpochs(epochs, f_samp_eeg, window_size, window_step, with_channel_dim): 

    if (with_channel_dim == False): 
        window_size_samp = int((window_size/1000) * f_samp_eeg) 
        window_step_samp = int((window_step/1000) * f_samp_eeg) 
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

    return wind_arr, num_of_windows, wind_names


def calcTestAccAndRates(prediction_labels, true_labels):
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

    window_step_samp = int((window_step/1000) * f_samp_eeg) 
    evaluation_samp_per_window = int((evaluation_time_per_window/1000) * f_samp_eeg) 
    window_predictions_samp = np.sum(wind_arr[:, -evaluation_samp_per_window:, :], axis = 1)
    # convert the label of each sampels to windowwise labels 
    window_predictions = (window_predictions_samp > 50.0).astype(float)

    window_eval_true_labels = np.zeros(window_predictions.shape)
    num_lrp_windows = int(np.round(n_samp_features/window_step_samp)) 
    trial_lrp_label = window_eval_true_labels[0, :] 
    trial_lrp_label[-num_lrp_windows:] = 1.0

    for trial_nr in range(0, window_eval_true_labels.shape[0]): 
        window_eval_true_labels[trial_nr, :] = trial_lrp_label

    #get metrics for window evaluation 
    if (use_relabelling == False): 
        tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), window_eval_true_labels.flatten())
    else: 
        relabelled_true_labels = applyRelabelling(window_predictions, determine_labels, searching_bounds)
        tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), relabelled_true_labels.flatten())
        window_eval_true_labels = relabelled_true_labels # return the relabelled true labels instead 

    return tnr, tpr, acc, ba, window_predictions, window_eval_true_labels


def prepareEpochsForNetwork(lrp_epochs, time_axis_eeg_batch, n_samp_features, continues_prediction, shuffle_data, t1_time, t2_time): 
    tensor_shape = lrp_epochs.shape # shape is (trials, channel, sampels)

    n_start = n_samp_features # number of samples to use as features 
    n_end = 1
    lrp_idx_1 = tensor_shape[2]-n_start-1
    lrp_idx_2 = tensor_shape[2]-n_end
    no_lrp_idx_1 = n_start

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

    # print("lrp indizes are: ", lrp_idx_1, lrp_idx_2)
    # print("no_lrp indizes are: ", t1, no_lrp_idx_1+t1)
    
    #seperate the samples of both classes 
    if(continues_prediction): # here no lrp sampels are cut out at even spaced steps to LRP class
        lrp_sampels = lrp_epochs[:, :, lrp_idx_1:lrp_idx_2] # shape (trials, channel, sampels)
        num_lrp_samps = lrp_sampels.shape[2]
        #print((lrp_idx_1/num_lrp_samps))
        stepsize_no_lrp = int(((t2-t1)/num_lrp_samps))
        no_lrp_ind = np.arange(t1, t2, step = stepsize_no_lrp) 
        
        no_lrp_indices = no_lrp_ind[no_lrp_ind.shape[0]-num_lrp_samps:] # make the balance !
        no_lrp_sampels = lrp_epochs[:, :, no_lrp_indices]
    else: 
        lrp_sampels = lrp_epochs[:, :, lrp_idx_1:lrp_idx_2] # shape (trials, channel, sampels)
        no_lrp_sampels = lrp_epochs[:, :, t1:no_lrp_idx_1+t1]
        
    #output the time values of the cutted slices: 
    
    lrp_times = np.array([time_axis_eeg_batch[lrp_idx_1] *1000,time_axis_eeg_batch[lrp_idx_2]*1000])
    if(continues_prediction):
        no_lrp_times = np.array([time_axis_eeg_batch[no_lrp_indices[0]]*1000,time_axis_eeg_batch[no_lrp_indices[-1]]*1000])
    else: 
        no_lrp_times = np.array([time_axis_eeg_batch[t1]*1000,time_axis_eeg_batch[no_lrp_idx_1+t1]*1000])

    print("lrp times in ms: ", (lrp_times).astype(dtype=np.int32))
    print("no_lrp times in ms: ", (no_lrp_times.astype(dtype=np.int32)))

    if(continues_prediction): 
        print("stepsize for no lrp is (samples): ", stepsize_no_lrp)

    # init arrays for both classes 
    lrp_shaped = np.zeros((lrp_sampels.shape[0]*lrp_sampels.shape[2], lrp_sampels.shape[1]))
    no_lrp_shaped = np.zeros((no_lrp_sampels.shape[0]*no_lrp_sampels.shape[2], no_lrp_sampels.shape[1]))

    # flatten sampels and trials to one dim 
    for channel in range(0, tensor_shape[1]): 
        lrp_shaped[:, channel] = lrp_sampels[:,channel,:].flatten()
        no_lrp_shaped[:, channel] = no_lrp_sampels[:,channel,:].flatten()

    #merge data together 
    x = np.concatenate((lrp_shaped, no_lrp_shaped), axis=0).astype(dtype = np.float64)

    #create label encoding 
    y = np.zeros((x.shape[0],1)).astype(dtype=np.float64)
    y[0:lrp_shaped.shape[0]] = 1.0
    
    #concatenate data for shuffling 
    if (shuffle_data): 
        x_y_concat = np.concatenate((x, y), axis=1)
        x_y_concat_shuffle = shuffle(x_y_concat)
        x = x_y_concat_shuffle[:, 0:x_y_concat_shuffle.shape[1]-1]
        y = x_y_concat_shuffle[:, x_y_concat_shuffle.shape[1]-1]
    
    return x, y
