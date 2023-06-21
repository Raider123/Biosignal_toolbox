
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


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"

# Create an string with dataset file name
file_str = data_path+"14032023_AU12D_unilateral_set3.txt"

# ANT data
#file_str = data_path+"01092022_BO42D_B_S0_M_1.txt"


# *** channel selection params ***
selected_channels =  []#['\tL.Brachiorad(uV)'] # ['\tEmg_4(uV)']['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
#inverse = True

# for ANT Systems 
#emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
#selected_channels = ["EMG2"]


# EMG sampling rate (Cometa)
fsamp_emg = 2000 # in Hz

#fsamp_emg = 500 # for ANT

# *** processing parameters *** 
target_frequency = 500 # target frequency after downsampling 

# variance filter length 
n_var = 170 # calc length of variance filter (default 170, tested for 500 Hz sampling rate)


# *********************************************************************************
# ************************* Load EMG data  ****************************************
# *********************************************************************************

# loading emg data (Cometa)
emg_data, emg_time_axis, emg_ch_names = emg_lib.loadCometaEMGData(file_str)
# (data, channels)

# load EMG data ANT
#emg_data, emg_time_axis = emg_lib.loadMiniANTEMGData(file_str, fsamp_emg)

# *********************************************************************************
# ************************* Process EMG data  *************************************
# *********************************************************************************


# plt.figure()
# plt.plot(emg_time_axis, emg_data[:, 0])
# plt.show()


# channel selection, if inverse = False all channels specified are kept 
#emg_data_select, emg_ch_names_select = emg_lib.channelSelection(emg_data, emg_ch_names, selected_channels, inverse)

# emg_data_select = emg_data_select[0:100, ]
# emg_time_axis = emg_time_axis[0:100,]
# apply bandpass filter 
#emg_data_filtered = emg_lib.applyBPFilterRectifying(fsamp_emg, 20, 200, emg_data_select)


#emg_lib.showEMGData(emg_data, emg_time_axis, emg_ch_names)
#emg_lib.showEMGData(emg_data_filtered, emg_time_axis, emg_ch_names_select)

was_same_before = False
i = 0

save_percent_values = []
for emg_data_ch_wise in emg_data.T: 
    i = i+1
    samp_count = [] 
    samp_value = []
    
    count = 0
    old_val = emg_data_ch_wise[0]
    for value in emg_data_ch_wise[:]: 
        if (value == old_val): 
            count = count+1
            samp_value.append(1)
            was_same_before = True
        else: 
            samp_value.append(0)

            if(was_same_before):
                samp_count.append(count)
                was_same_before = False
                count = 0 
        
        old_val = value
        
    plt.figure()
    plt.scatter(np.arange(0, len(samp_count)), samp_count)
    plt.title("Amount of sampels with same values in a row")
    plt.xlabel("samples (>1 same samples)")
    plt.ylabel("counted values")
    plt.show()

    # plt.figure()
    # plt.plot(samp_value)
    # plt.show()

    print("Channel: ", i)
    print("Percentage error: ", ((np.sum(samp_value)/len(emg_data_ch_wise))*100))
    save_percent_values.append((np.sum(samp_value)/len(emg_data_ch_wise))*100)
    print("Max count: ",np.max(samp_count))
    print("")

np.savetxt("values.csv", np.array(save_percent_values), delimiter=",")

# downsampling to target frequency 
#emg_data_down, time_axis_down = emg_lib.decimateEMGData(emg_data_select, emg_time_axis, target_frequency, fsamp_emg)

#emg_lib.showEMGData(emg_data_down, time_axis_down, emg_ch_names_select)

#apply variance filter to EMG signals  
#emg_data_filtered = emg_lib.applyVarianceFilter(emg_data_down, n_var)

# show EMG signals in plots 
#emg_lib.showEMGData(emg_data_filtered, time_axis_down, emg_ch_names_select)


