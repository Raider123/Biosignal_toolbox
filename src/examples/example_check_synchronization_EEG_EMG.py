
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

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
filenames_EEG = [["20230424_AC17D_orthosisErrorIjcai_multi_set1"],
                ["20230424_AC17D_orthosisErrorIjcai_multi_set2"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set3"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set4"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set5"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set6"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set7"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set8"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set9"] ,
                ["20230424_AC17D_orthosisErrorIjcai_multi_set10"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_baseline_set1"], 

                ["20230427_AA56D_orthosisErrorIjcai_multi_set1"],
                ["20230427_AA56D_orthosisErrorIjcai_multi_set2"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set3"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set4"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set5"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set6"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set7"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set9"] ,
                ["20230427_AA56D_orthosisErrorIjcai_multi_set10"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set11"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_baseline_set1"],

                ["20230426_AJ05D_orthosisErrorIjcai_multi_set1"],
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set2"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set3"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set4"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set5"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set6"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set7"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set8"] ,
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set9"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set10"], 

                ["20230421_AQ59D_orthosisErrorIjcai_multi_set1"],
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set2"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set3"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set4"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set5"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set6"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set7"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set8"] ,
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set9"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set10"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1"],

                ["20230425_AW59D_orthosisErrorIjcai_multi_set1"],
                ["20230425_AW59D_orthosisErrorIjcai_multi_set2"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set3"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set4"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set5"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set6"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set7"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set8"] ,
                ["20230425_AW59D_orthosisErrorIjcai_multi_set9"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set10"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2"],

                ["20230425_AY63D_orthosisErrorIjcai_multi_set1"],
                ["20230425_AY63D_orthosisErrorIjcai_multi_set2"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set3"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set4"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set5"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set6"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set7"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set8"] ,
                ["20230425_AY63D_orthosisErrorIjcai_multi_set9"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set10"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2"],

                ["20230426_BS34D_orthosisErrorIjcai_multi_set1"],
                ["20230426_BS34D_orthosisErrorIjcai_multi_set2"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set3"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set4"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set5"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set6"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set7"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set8"] ,
                ["20230426_BS34D_orthosisErrorIjcai_multi_set9"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set10"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1"]
                ]

filenames_EMG = ["20230424_AC17D_orthosisErrorIjcai_multi_set1.txt",
                 "20230424_AC17D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230424_AC17D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set8.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set9.txt" , 
                 "20230424_AC17D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230424_AC17D_orthosisErrorIjcai_multi_baseline_set1.txt",

                 "20230426_AA56D_orthosisErrorIjcai_multi_set1.txt",
                 "20230426_AA56D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230426_AA56D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230426_AA56D_orthosisErrorIjcai_multi_set11.txt",
                 "20230426_AA56D_orthosisErrorIjcai_multi_baseline_set1.txt", 

                 "20230426_AJ05D_orthosisErrorIjcai_multi_set1.txt",
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230426_AJ05D_orthosisErrorIjcai_multi_set10.txt", 

                 "20230421_AQ59D_orthosisErrorIjcai_multi_set1.txt",
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.txt",

                 "20230425_AW59D_orthosisErrorIjcai_multi_set1.txt",
                 "20230425_AW59D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230425_AW59D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.txt",

                 "20230425_AY63D_orthosisErrorIjcai_multi_set1.txt",
                 "20230425_AY63D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230425_AY63D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230425_AY63D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.txt",
# +
                 "20230426_BS34D_orthosisErrorIjcai_multi_set1.txt",
                 "20230426_BS34D_orthosisErrorIjcai_multi_set2.txt" , 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set3.txt" ,
                 "20230426_BS34D_orthosisErrorIjcai_multi_set4.txt" , 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set5.txt" , 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set6.txt" , 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set7.txt" , 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set8.txt", 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set9.txt", 
                 "20230426_BS34D_orthosisErrorIjcai_multi_set10.txt", 
                 "20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.txt"
                 ]


EMG_marker_number = 1
f_samp_emg = 1000 


sample_diffs = []
measurement_times = []

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

    # EMG data 

    file_str_emg = os.path.join(data_path, filename_EMG)
    emg_data, emg_time_axis = emg_lib.loadMiniANTEMGData(file_str_emg, f_samp_emg)

    sampling_factor = f_samp_emg/f_samp_eeg

    l_emg = len(emg_data)
    #print("Data points (amount) EMG: ", l_emg)
    EEG_compare = (EMG_start_end_indices[1]-EMG_start_end_indices[0])*sampling_factor
    #print("Data points (amount) EEG: ", EEG_compare)

    #print("Sample diff: ", l_emg -EEG_compare)
    #print("Time diff in ms: ", ((l_emg -EEG_compare)*f_samp_emg)/1000) # 1000 Hz = 1 ms 

    measurement_times.append(EEG_compare/f_samp_emg)
    sample_diffs.append(l_emg -EEG_compare)


print("Sample diffs: ", sample_diffs)
print("Measurement times (s): ", measurement_times) 

sample_diffs = np.array(sample_diffs)
measurement_times = np.array(measurement_times)

fig = plt.figure()
plt.scatter(measurement_times, sample_diffs)
plt.xlabel("Measurement time in s")
plt.ylabel("Sample diff (EMG/EEG) in ms")
plt.title("All subjects synchro")
plt.show()


print(np.mean(np.abs(sample_diffs)))

fig.savefig("synchro.svg")
