
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt


# own libs 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder 
import emg_lib
import eeg_lib
import mne 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"

# Create an array with dataset file names 
#data_str_arr = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.txt", data_path+"20211210_r_JV43_intentional_unilateral_set2.txt"])
data_str_uni_JV43_EMG = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.txt", data_path+"20211210_r_JV43_intentional_unilateral_set2.txt", data_path+"20211210_r_JV43_intentional_unilateral_set3.txt"])
data_str_uni_JV43_EEG = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set2.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set3.vhdr"])
data_str_uni_RA12_EMG = np.array([data_path+"20211216_r_RA12_intentional_unilateral_set1.txt", data_path+"20211216_r_RA12_intentional_unilateral_set2.txt", data_path+"20211216_r_RA12_intentional_unilateral_set3.txt"])
data_str_uni_RA12_EEG = np.array([data_path+"20211216_r_RA12_intentional_unilateral_set1.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set2.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set3.vhdr"])
data_str_uni_AV82_EMG = np.array([data_path+"20211220_r_AV82_intentional_unilateral_set1.txt", data_path+"20211220_r_AV82_intentional_unilateral_set2.txt", data_path+"20211220_r_AV82_intentional_unilateral_set3.txt"])
data_str_uni_AV82_EEG = np.array([data_path+"20211220_r_AV82_intentional_unilateral_set1.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set2.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set3.vhdr"])
data_str_uni_UP28_EMG = np.array([data_path+"20211223_r_UP28_intentional_unilateral_set1.txt", data_path+"20211223_r_UP28_intentional_unilateral_set2.txt", data_path+"20211223_r_UP28_intentional_unilateral_set3.txt"])
data_str_uni_UP28_EEG = np.array([data_path+"20211223_r_UP28_intentional_unilateral_set1.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set2.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set3.vhdr"])
data_str_uni_XP01_EMG = np.array([data_path+"20211222_r_XP01_intentional_unilateral_set1.txt", data_path+"20211222_r_XP01_intentional_unilateral_set2.txt", data_path+"20211222_r_XP01_intentional_unilateral_set3.txt", data_path+"20211222_r_XP01_intentional_unilateral_set4.txt"])
data_str_uni_XP01_EEG = np.array([data_path+"20211222_r_XP01_intentional_unilateral_set1.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set2.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set3.vhdr",  data_path+"20211222_r_XP01_intentional_unilateral_set4.vhdr"])
data_str_uni_ZS27_EMG = np.array([data_path+"20220104_r_ZS27_intentional_unilateral_set1.txt", data_path+"20220104_r_ZS27_intentional_unilateral_set2.txt", data_path+"20220104_r_ZS27_intentional_unilateral_set3.txt"])
data_str_uni_ZS27_EEG = np.array([data_path+"20220104_r_ZS27_intentional_unilateral_set1.vhdr", data_path+"20220104_r_ZS27_intentional_unilateral_set2.vhdr", data_path+"20220104_r_ZS27_intentional_unilateral_set3.vhdr"])
data_str_uni_JD68_EMG = np.array([data_path+"20220105_r_JD68_intentional_unilateral_set1.txt", data_path+"20220105_r_JD68_intentional_unilateral_set2.txt", data_path+"20220105_r_JD68_intentional_unilateral_set3.txt"])
data_str_uni_JD68_EEG = np.array([data_path+"20220105_r_JD68_intentional_unilateral_set1.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set2.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set3.vhdr"])
data_str_uni_QS70_EMG = np.array([data_path+"20220107_r_QS70_intentional_unilateral_set1.txt", data_path+"20220107_r_QS70_intentional_unilateral_set2.txt", data_path+"20220107_r_QS70_intentional_unilateral_set3.txt"])
data_str_uni_QS70_EEG = np.array([data_path+"20220107_r_QS70_intentional_unilateral_set1.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set2.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set3.vhdr"])


dataset_EMG = data_str_uni_JV43_EMG
dataset_EEG = data_str_uni_JV43_EEG

# *** channel selection params ***
selected_channels = ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
inverse = False

# Which sets are used 
set_nums = [0, 1, 2]
validation_rate = 0.5 # rate to split test and validation data 

# name pattern of current subject and paradigm 
subject_paradigm_name = dataset_EMG[0].split("_r_")[1].split("_set")[0]+"_emg"

# EMG sampling rate 
fsamp_emg = 2000 # in Hz

# markernumbers of EEG system 
onset_marker = 100
start_marker = 1

# *** processing parameters *** 
target_frequency = 500 # tdata_str_uni_UP28_EEGarget frequency after downsampling 
n_var = int(680/4) # length of the variance filter 

# epoching 
t_start = -5.0 # in ms 
t_stop = 0 # in ms 

# *********************************************************************************
# ************************* Load EMG data and split to train and test  ************
# *********************************************************************************


#load EEG data 

def EMGProcessingPipeline(dataset, dataset_eeg):

    print("")
    print(dataset_eeg)
    print("")
    eeg_raw = eeg_lib.loadBrainproductsData(dataset_eeg)
    events, event_id = mne.events_from_annotations(eeg_raw)

    # *********************************************************************************
    # ************************* Process EMG data  *************************************
    # *********************************************************************************

    # loading EMG data 
    emg_data, emg_time_axis, emg_ch_names = emg_lib.loadCometaEMGData(dataset)

    # # channel selection, if inverse = False all channels specified are kept 
    emg_data_select, emg_ch_names_select = emg_lib.channelSelection(emg_data, emg_ch_names, selected_channels, inverse)

    # # downsampling to target frequency 
    emg_data_down, time_axis_down = emg_lib.decimateEMGData(emg_data_select, emg_time_axis, target_frequency, fsamp_emg)

    # #emg_lib.showEMGData(emg_data_down, time_axis_down, emg_ch_names_select)

    emg_data_filtered = emg_lib.applyVarianceFilter(emg_data_down, n_var)

    # emg_lib.showEMGData(emg_data_filtered, time_axis_down, emg_ch_names_select)

    # calc onsets for synchronization and epoching 
    align_index = events[np.where(events[:, 2] == start_marker)[0][0], 0]
    onset_indices = events[np.where(events[:, 2] == onset_marker)[0], 0]

    # correction of onset indices for synchronization 
    onset_indices_corrected = onset_indices-align_index # fit indices to emg data (correction of synchronization)
    print("Number of onset indices: ", len(onset_indices_corrected))
    #print(onset_indices_corrected)

    # get emg epochs based on markers 
    emg_epochs_processed = emg_lib.epocheEMGData(emg_data_filtered, onset_indices_corrected, target_frequency, t_start, t_stop)

    return emg_epochs_processed, time_axis_down, emg_ch_names_select


# train and test set processing 
emg_set1_processed, time_axis_down1, emg_ch_names_select1 = EMGProcessingPipeline(dataset_EMG[0], [dataset_EEG[0]])
emg_set2_processed, time_axis_down2, emg_ch_names_select2 = EMGProcessingPipeline(dataset_EMG[1], [dataset_EEG[1]])
emg_set3_processed, time_axis_down3, emg_ch_names_select3 = EMGProcessingPipeline(dataset_EMG[2], [dataset_EEG[2]])

# for XP01 
#emg_set4_processed, time_axis_down4, emg_ch_names_select4 = EMGProcessingPipeline(dataset_EMG[3], [dataset_EEG[3]])

print("len1: ", emg_set1_processed.shape)
print("len2: ", emg_set2_processed.shape)
print("len3: ", emg_set3_processed.shape)

#create 3 permutations with different train test and validation set combinations
for iterations in set_nums:  # change here later on 
    if(iterations == 0): 
        
        #train_list = [dataset[0], dataset[1]]
        #test_list = [dataset[2]]
        #test_list = [dataset[3], dataset[2]] # use for XP01 

        emg_train = np.concatenate((emg_set1_processed, emg_set2_processed), axis=0)
        emg_test  = emg_set3_processed
        #emg_test = np.concatenate((emg_set4_processed, emg_set3_processed), axis=0) # XP01


    elif(iterations == 1): 
        
        emg_train = np.concatenate((emg_set2_processed, emg_set3_processed), axis=0)
        #emg_train = np.concatenate((emg_set2_processed, emg_set3_processed, emg_set4_processed), axis=0) # XP01
        emg_test  = emg_set1_processed

    else: 
    
        emg_train = np.concatenate((emg_set1_processed, emg_set3_processed), axis=0)
        #emg_train = np.concatenate((emg_set1_processed, emg_set3_processed, emg_set4_processed), axis=0) # XP01
        emg_test  = emg_set2_processed


    scaler = mne.decoding.Scaler(info=None, scalings='mean', with_mean=True, with_std=True)  #(n_epochs, n_channels, n_times) scaler requires this shape
    scaler.fit(emg_train) # fit to epochs data

    # channel wise normalization 
    emg_epochs_train_scaled= scaler.transform(emg_train) # transform train data (unit variance and zero mean)
    emg_epochs_test_val_scaled= scaler.transform(emg_test) # transform test val data (unit variance and zero mean)


    # seperate test and validation
    test_val_idx = int(emg_epochs_test_val_scaled.shape[0]*validation_rate) 
    emg_epochs_val_scaled = emg_epochs_test_val_scaled[0:test_val_idx, :, :]
    emg_epochs_test_scaled = emg_epochs_test_val_scaled[test_val_idx:, :, :]


    # make same shape as for EEG ! 
    emg_epochs_val_scaled = np.insert(emg_epochs_val_scaled, 0, 0, axis=2)
    emg_epochs_test_scaled = np.insert(emg_epochs_test_scaled, 0, 0, axis=2)
    emg_epochs_train_scaled = np.insert(emg_epochs_train_scaled, 0, 0, axis=2)

    # *********************************************************************************
    # *********************** Save preprocessed EMG data ******************************
    # *********************************************************************************

    # save preprocessed data with shape (trials, channel, sampel) for every permutation 
    np.save(data_path+subject_paradigm_name+"_train_"+str(iterations), emg_epochs_train_scaled) # shape (8, 3, 2501)
    np.save(data_path+subject_paradigm_name+"_test_"+str(iterations), emg_epochs_test_scaled)
    np.save(data_path+subject_paradigm_name+"_val_"+str(iterations), emg_epochs_val_scaled)
    np.save(data_path+"time_axis_emg_epochs", time_axis_down1)
    np.save(data_path+"remaining_emg_channel_names", emg_ch_names_select1)


    print("len train data: ", emg_epochs_train_scaled.shape)

    print("")
    print("Done storing preprocessed data")
