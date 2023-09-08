
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 
# own libs 
# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
# # own libs 
from biosignal_toolbox.emg_lib import EMGData
import mne 

# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


#filenames 
filenames_EEG = [["20230424_AC17D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230424_AC17D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set7.vhdr"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set8.vhdr"] , 
                ["20230424_AC17D_orthosisErrorIjcai_multi_set9.vhdr"] ,
                ["20230424_AC17D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230424_AC17D_orthosisErrorIjcai_multi_baseline_set1.vhdr"], 

                ["20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230427_AA56D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set9.vhdr"] ,
                ["20230427_AA56D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_set11.vhdr"], 
                ["20230427_AA56D_orthosisErrorIjcai_multi_baseline_set1.vhdr"],

                ["20230426_AJ05D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set8.vhdr"] ,
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set9.vhdr"], 
                ["20230426_AJ05D_orthosisErrorIjcai_multi_set10.vhdr"], 

                ["20230421_AQ59D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set8.vhdr"] ,
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set9.vhdr"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.vhdr"],

                ["20230425_AW59D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230425_AW59D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set8.vhdr"] ,
                ["20230425_AW59D_orthosisErrorIjcai_multi_set9.vhdr"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.vhdr"],

                ["20230425_AY63D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230425_AY63D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set8.vhdr"] ,
                ["20230425_AY63D_orthosisErrorIjcai_multi_set9.vhdr"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.vhdr"],

                ["20230426_BS34D_orthosisErrorIjcai_multi_set1.vhdr"],
                ["20230426_BS34D_orthosisErrorIjcai_multi_set2.vhdr"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set3.vhdr"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set4.vhdr"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set5.vhdr"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set6.vhdr"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set7.vhdr"] , 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set8.vhdr"] ,
                ["20230426_BS34D_orthosisErrorIjcai_multi_set9.vhdr"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_set10.vhdr"], 
                ["20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.vhdr"]
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

 

    # EEG data 
    EEG = EEGData(format = "Brainvision", filenames = filename_EEG, data_path = data_path)

    # get events and extract the marker indizes 
    events = EEG.getEvents()

    EMG_indices = np.where(events[:, 2] == EMG_marker_number)[0]
    EMG_start_end_indices = events[EMG_indices, 0]

    # EMG data 

    EMG = EMGData(format="ANTmini",data_path = data_path, filename = filename_EMG, f_samp = f_samp_emg)

    emg_data, emg_time_axis = EMG.getEMGData()

    sampling_factor = f_samp_emg/EEG.getSamplingRate()

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
