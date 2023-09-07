
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import mne
import sys 

# own libs 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder 
import eeg_lib


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"

# Give the path of the datasets that are available 
data_str_uni_JV43 = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set2.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set3.vhdr"])
data_str_uni_RA12 = np.array([data_path+"20211216_r_RA12_intentional_unilateral_set1.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set2.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set3.vhdr"])
data_str_uni_AV82 = np.array([data_path+"20211220_r_AV82_intentional_unilateral_set1.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set2.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set3.vhdr"])
data_str_uni_UP28 = np.array([data_path+"20211223_r_UP28_intentional_unilateral_set1.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set2.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set3.vhdr"])
data_str_uni_XP01 = np.array([data_path+"20211222_r_XP01_intentional_unilateral_set1.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set2.vhdr",data_path+"20211222_r_XP01_intentional_unilateral_set3.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set4.vhdr"])
data_str_uni_ZS27 = np.array([data_path+"20220104_r_ZS27_intentional_unilateral_set1.vhdr", data_path+"20220104_r_ZS27_intentional_unilateral_set2.vhdr",data_path+"20220104_r_ZS27_intentional_unilateral_set3.vhdr"])
data_str_uni_JD68 = np.array([data_path+"20220105_r_JD68_intentional_unilateral_set1.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set2.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set3.vhdr"])
data_str_uni_QS70 = np.array([data_path+"20220107_r_QS70_intentional_unilateral_set1.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set2.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set3.vhdr"])


#choose a dataset of a subject 
dataset = data_str_uni_UP28

dataset_pool = np.concatenate((data_str_uni_JD68, data_str_uni_XP01, data_str_uni_JV43, data_str_uni_RA12, data_str_uni_QS70, data_str_uni_AV82, data_str_uni_ZS27))
include_test_sub_in_train = False


# Which sets are used 
set_nums = [0, 1, 2]
validation_rate = 0.2 # rate to split test and validation data 

# name pattern of current subject and paradigm 

# Channelnumbers with EEG Data from Dataset
eeg_channel_start_number = 0 
eeg_channel_end_number   = 67 # 64 eeg and 3 axis accelerometer (automatically removed laterl on )

# name pattern of current subject and paradigm
subject_paradigm_name = dataset[0].split("_r_")[1].split("_set")[0]+"_34ch_05_40Hz_pool"


# Filtering Params for EEG data 
f_highpass = 0.5 #0.5 # in Hz 
f_lowpass = 40.0 # 4.0 in Hz 

apply_filter = True # setting to False will ignore the 

#rereferencing (["average"] or [] for no reref (otherwise specify channel names))
reref_channel = []

# should baseline correction be applied ? (standard -1.5 to -1 seconds)
apply_baseline_correction = False 
t0_baseline = 0 # not used
t1_baseline = 0

f_samp_eeg = 500 #sample Frequency of eeg
marker_number = 100 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 3 # number of the error marker 

# specifying the movement onset marker 
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -5.0 # time in seconds (start epoch)
epoching_time_after_onset = 0.0 # time in seconds (0 = movement onset)


# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True
channel_list = ["x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]


# just remap the parameters (need to be adapted)
t1 = epoching_time_before_onset
t2 = epoching_time_after_onset

# *********************************************************************************
# ***************** Load and merge datasets and split to train, test, val *********
# *********************************************************************************

# create 3 permutations with different train test and validation set combinations
for iterations in set_nums:  # change here later on 
    if(iterations == 0): 
        
        if (include_test_sub_in_train): 
            train_list = [dataset[0], dataset[1]]+ list(dataset_pool)
        else: 
            train_list = list(dataset_pool)
            train_list_tuning = [dataset[0], dataset[1]]
            raw_train_tuning = eeg_lib.loadBrainproductsData(train_list_tuning)

        test_list = [dataset[2]]

        #test_list = [dataset[3], dataset[2]] # use for XP01 

        raw_train= eeg_lib.loadBrainproductsData(train_list) # read data in brainproducts format
        raw_test_val = eeg_lib.loadBrainproductsData(test_list) # read data in brainproducts format 

    elif(iterations == 1): 
        
        if (include_test_sub_in_train): 
            train_list = [dataset[1], dataset[2]] + list(dataset_pool)
        else: 
            train_list = list(dataset_pool) 
            train_list_tuning = [dataset[1], dataset[2]] 
            #train_list_tuning = [dataset[1], dataset[2], dataset[3]] # use for XP01
            raw_train_tuning = eeg_lib.loadBrainproductsData(train_list_tuning)
        

        #train_list = [dataset[1], dataset[2], dataset[3]] +list(dataset_pool) # use for XP01 
        test_list = [dataset[0]]


        raw_train= eeg_lib.loadBrainproductsData(train_list) # read data in brainproducts format
        raw_test_val = eeg_lib.loadBrainproductsData(test_list) # read data in brainproducts format 

    else: 

        if (include_test_sub_in_train): 
            train_list = [dataset[0], dataset[2]] + list(dataset_pool)

        else: 
            train_list = list(dataset_pool) 
            train_list_tuning = [dataset[0], dataset[2]]
            #train_list_tuning = [dataset[0], dataset[2], dataset[3]] # use for XP01 
            raw_train_tuning = eeg_lib.loadBrainproductsData(train_list_tuning)
        
        #train_list = [dataset[0], dataset[2], dataset[3]] + list(dataset_pool)# use for XP01 
        test_list = [dataset[1]]


        raw_train= eeg_lib.loadBrainproductsData(train_list) # read data in brainproducts format
        raw_test_val = eeg_lib.loadBrainproductsData(test_list) # read data in brainproducts format 
        

    # *********************************************************************************
    # *********** EEG preprocessing, epoching and train, test split *******************
    # *********************************************************************************
    
    print(raw_train.ch_names)

    # epoch the eeg data to trial length (for merged sets)
    lrp_epochs_train, lrp_epochs_train_obj, time_axis_eeg_batch, remaining_eeg_channel_names, raw_train_obj = eeg_lib.rereferencingEpoching(raw_train, onset_number, error_number,channel_list,inverse_keep_channel, reref_channel, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline)
    lrp_epochs_test_val, lrp_epochs_test_val_obj, time_axis_eeg_batch, remaining_eeg_channel_names, raw_test_val_obj = eeg_lib.rereferencingEpoching(raw_test_val, onset_number, error_number,channel_list, inverse_keep_channel, reref_channel, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline)


    # fit scaler only on train data 
    # train test permutation 1 
    scaler = mne.decoding.Scaler(info=raw_train_obj.info, scalings='mean', with_mean=True, with_std=True)  #(n_epochs, n_channels, n_times) scaler requires this shape
    scaler.fit(lrp_epochs_train) # fit to epochs data
    
    # scaler for fine tuning 
    if(include_test_sub_in_train == False): 
        lrp_epochs_train_tune, lrp_epochs_train_obj_tune, time_axis_eeg_batch, remaining_eeg_channel_names, raw_train_obj_tune = eeg_lib.rereferencingEpoching(raw_train_tuning, onset_number, error_number,channel_list,inverse_keep_channel, reref_channel, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline)
        scaler_tune = mne.decoding.Scaler(info=raw_train_obj_tune.info, scalings='mean', with_mean=True, with_std=True)  #(n_epochs, n_channels, n_times) scaler requires this shape
        scaler_tune.fit(lrp_epochs_train_tune) # fit to epochs data
        lrp_epochs_train_tune_scaled= scaler_tune.transform(lrp_epochs_train_tune) # transform train data (unit variance and zero mean)


    if(include_test_sub_in_train): 
        lrp_epochs_train_scaled= scaler.transform(lrp_epochs_train) # transform train data (unit variance and zero mean)
        lrp_epochs_test_val_scaled= scaler.transform(lrp_epochs_test_val) # transform test val data (unit variance and zero mean)
    else: 
        # scale train and test data with actual subjects data (adapt pooling data)
        lrp_epochs_train_scaled= scaler.transform(lrp_epochs_train) # transform train data (unit variance and zero mean)
        lrp_epochs_test_val_scaled= scaler_tune.transform(lrp_epochs_test_val) # transform test val data (unit variance and zero mean)


    # lrp_epochs_train_scaled = lrp_epochs_train_scaled -np.min(lrp_epochs_train_scaled)
    # lrp_epochs_train_scaled = lrp_epochs_train_scaled/np.max(lrp_epochs_train_scaled)
    # lrp_epochs_test_val_scaled = -np.min(lrp_epochs_train_scaled)
    # lrp_epochs_test_val_scaled = lrp_epochs_test_val_scaled/np.max(lrp_epochs_test_val_scaled)
    

    # seperate test and validation
    test_val_idx = int(lrp_epochs_test_val_scaled.shape[0]*validation_rate) 
    lrp_epochs_val_scaled = lrp_epochs_test_val_scaled[0:test_val_idx, :, :]
    lrp_epochs_test_scaled = lrp_epochs_test_val_scaled[test_val_idx:, :, :]
    

    # *********************************************************************************
    # *********************** Save preprocessed EEG data ******************************
    # *********************************************************************************

    # save preprocessed data with shape (trials, channel, sampel) for every permutation 
    np.save(data_path+subject_paradigm_name+"_train_"+str(iterations), lrp_epochs_train_scaled)
    np.save(data_path+subject_paradigm_name+"_test_"+str(iterations), lrp_epochs_test_scaled)
    np.save(data_path+subject_paradigm_name+"_val_"+str(iterations), lrp_epochs_val_scaled)
    np.save(data_path+"time_axis_eeg_epochs", time_axis_eeg_batch)
    np.save(data_path+"remaining_eeg_channel_names", remaining_eeg_channel_names)

    # save tuning data seperately 
    if(include_test_sub_in_train == False): 
        np.save(data_path+subject_paradigm_name+"_trainTune_"+str(iterations), lrp_epochs_train_tune_scaled)


    print("shape train data: ", lrp_epochs_train_scaled.shape)
    print("shape train tune data: ", lrp_epochs_train_tune_scaled.shape)

    print("")
    print("Done storing preprocessed data")

