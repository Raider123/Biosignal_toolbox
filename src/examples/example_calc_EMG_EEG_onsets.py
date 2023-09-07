
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

from xml.etree.ElementPath import xpath_tokenizer_re
import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 

import mne 

# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 


# own libs 
import eeg_lib
import emg_lib

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


#filenames 
# filenames_EEG = [["20230424_AC17D_orthosisErrorIjcai_multi_set1"],
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set7"], 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set8"] , 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set9"] ,
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_set10"], 
#                 ["20230424_AC17D_orthosisErrorIjcai_multi_baseline_set1"], 

#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set1"],
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set7"] , 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set9"] ,
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set10"], 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_set11"], 
#                 ["20230427_AA56D_orthosisErrorIjcai_multi_baseline_set1"],

#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set1"],
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set7"] , 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set8"] ,
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set9"], 
#                 ["20230426_AJ05D_orthosisErrorIjcai_multi_set10"], 

#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set1"],
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set7"] , 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set8"] ,
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set9"], 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_set10"], 
#                 ["20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1"],

filenames_EEG = [["20230425_AW59D_orthosisErrorIjcai_multi_set1"],
                ["20230425_AW59D_orthosisErrorIjcai_multi_set2"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set3"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set4"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set5"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set6"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set7"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set8"] ,
                ["20230425_AW59D_orthosisErrorIjcai_multi_set9"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set10"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2"]]

# filenames_EEG = [["20230425_AY63D_orthosisErrorIjcai_multi_set1"],
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set7"] , 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set8"] ,
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set9"], 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_set10"], 
#                 ["20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2"]]

# filenames_EEG = [["20230426_BS34D_orthosisErrorIjcai_multi_set1"],
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set2"] , 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set3"], 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set4"] , 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set5"], 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set6"] , 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set7"] , 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set8"] ,
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set9"], 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_set10"], 
#                 ["20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1"]
#                 ]
# filenames_EMG = ["20230424_AC17D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set8.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set9.txt" , 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_set10.txt", 
#                  "20230424_AC17D_orthosisErrorIjcai_multi_baseline_set1.txt",

#                  "20230426_AA56D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set9.txt", 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set10.txt", 
#                  "20230426_AA56D_orthosisErrorIjcai_multi_set11.txt",
#                  "20230426_AA56D_orthosisErrorIjcai_multi_baseline_set1.txt", 

#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set8.txt", 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set9.txt", 
#                  "20230426_AJ05D_orthosisErrorIjcai_multi_set10.txt", 

#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set8.txt", 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set9.txt", 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_set10.txt", 
#                  "20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.txt",

filenames_EMG = ["20230425_AW59D_orthosisErrorIjcai_multi_set1.txt",
                 "20230425_AW59D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230425_AW59D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.txt"]

# filenames_EMG = ["20230425_AY63D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set8.txt", 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set9.txt", 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_set10.txt", 
#                  "20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.txt"]
# +
# filenames_EMG = ["20230426_BS34D_orthosisErrorIjcai_multi_set1.txt",
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set2.txt" , 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set3.txt" ,
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set4.txt" , 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set5.txt" , 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set6.txt" , 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set7.txt" , 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set8.txt", 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set9.txt", 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_set10.txt", 
#                  "20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.txt"
#                  ]


EMG_marker_number = 1
Button_press_number = 96
f_samp_emg = 1000 

# for ANT Systems 
emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]


sample_diffs = []
measurement_times = []
emg_label_list = []

for i in range(0, len(filenames_EMG)): 
    
    filename_EMG = filenames_EMG[i]
    filename_EEG = filenames_EEG[i]

    print(filename_EEG)
    # *********************************************************************************
    # ***************** Load and concatenate a dataset *********
    # *********************************************************************************

    #create numpy array with file names 
    data_str_arr = []
    for files_str in filename_EEG: 
        data_str_arr.append(os.path.join(data_path, files_str)) 

    # EEG data 
    raw= eeg_lib.loadBrainproductsData(data_str_arr) # read data in brainproducts format
    f_samp_eeg = raw.info['sfreq'] # get sampling rate 

    # get events and extract the marker indizes 
    events, event_id = mne.events_from_annotations(raw)

    EMG_indices = np.where(events[:, 2] == EMG_marker_number)[0]
    EMG_start_end_indices = events[EMG_indices, 0]

    # get button presses 
    button_presses = np.where(events[:, 2] == Button_press_number)[0]
    button_press_indices = events[button_presses, 0]


    # EMG data loading and processing 
    file_str_emg = os.path.join(data_path, filename_EMG)
    emg_data, emg_time_axis = emg_lib.loadMiniANTEMGData(file_str_emg, f_samp_emg)

    # channel selection, if inverse = False all channels specified are kept 
    emg_data_select, emg_ch_names_select = emg_lib.channelSelection(emg_data, emg_ch_names, ["EMG2","EMG1"], False)


    # apply bandpass filter 
    emg_data_filtered = emg_lib.applyBPFilterRectifying(f_samp_emg, 20, 450, emg_data_select)

    # show all EMG channels
    #emg_lib.showEMGData(emg_data_filtered, emg_time_axis, emg_ch_names_select)
    sampling_factor = f_samp_emg/f_samp_eeg

    button_press_indices_in_emg = ((button_press_indices-EMG_start_end_indices[0])*sampling_factor).astype(int)
    
    y_before = -1
    y_after = 1
    emg_epochs = emg_lib.epocheEMGData(emg_data_filtered, button_press_indices_in_emg, f_samp_emg, y_before, y_after)
    print(emg_epochs.shape)
    emg_epochs_button_press = emg_epochs[:, 1,:] # only first channel due to bug 


    for i in range(0, emg_epochs_button_press.shape[0]): 
        emg_slice = emg_epochs_button_press[i, :]

        button_press_ind = button_press_indices_in_emg[i]

        print("Button press ind: ", button_press_ind)

        fig = plt.figure()
        plt.plot(emg_slice)
        print(y_after*f_samp_emg)
        # plt.axvline(x = 1000, color = 'g', label = 'axvline - full height', ymin = np.min(emg_slice), ymax = np.max(emg_slice))

        # input = plt.ginput(1)

        plt.show()

#         x = input[0]
#         x_val = x[0]

#         EMG_onset_ind = int(x_val)+button_press_ind+int(y_before*f_samp_emg)
        
#         print("EMG onset", EMG_onset_ind)

#         print("Diff EMG onset, button press: ", (EMG_onset_ind -button_press_ind)/f_samp_emg)
#         # print("EMG Onset", int(input[0])+i)
#         # label_list.append(input[0])
        

#         emg_label_list.append((EMG_onset_ind- button_press_ind)/f_samp_emg)

#         print("EMG label list: ",emg_label_list)

# print("Mean vals: ",np.mean(emg_label_list))
# print("STD vals: ",np.std(emg_label_list))

# np.save("AW59D", emg_label_list)


