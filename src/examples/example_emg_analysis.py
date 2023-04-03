
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

# Create an string with dataset file name (ANT and Cometa)
file_str = data_path+"example.txt"


# *** channel selection params ***
selected_channels =  []#['\tL.Brachiorad(uV)'] # ['\tEmg_4(uV)']['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
inverse = True

# for ANT Systems 
#emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
#selected_channels = ["EMG2"]


# EMG sampling rate (Cometa default)
fsamp_emg = 2000 # in Hz

#fsamp_emg = 500 # for ANT

# *** processing parameters *** 
target_frequency = 500 # target frequency after downsampling (Cometa)

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

# *** Just comment in what should be used for processing *** 

# channel selection, if inverse = False all channels specified are kept 
#emg_data_select, emg_ch_names_select = emg_lib.channelSelection(emg_data, emg_ch_names, selected_channels, inverse)

# apply bandpass filter 
#emg_data_filtered = emg_lib.applyBPFilterRectifying(fsamp_emg, 20, 200, emg_data_select)

# show all EMG channels
#emg_lib.showEMGData(emg_data, emg_time_axis, emg_ch_names)

# show selected EMG channels
#emg_lib.showEMGData(emg_data_filtered, emg_time_axis, emg_ch_names_select)

# downsampling to target frequency 
#emg_data_down, time_axis_down = emg_lib.decimateEMGData(emg_data_select, emg_time_axis, target_frequency, fsamp_emg)


#emg_lib.showEMGData(emg_data_down, time_axis_down, emg_ch_names_select)

#apply variance filter to EMG signals  
#emg_data_filtered = emg_lib.applyVarianceFilter(emg_data_down, n_var)

# show EMG signals in plots 
#emg_lib.showEMGData(emg_data_filtered, time_axis_down, emg_ch_names_select)

