
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

# Create an array with dataset file names 
#data_str_arr = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.txt", data_path+"20211210_r_JV43_intentional_unilateral_set2.txt"])
file_str = data_path+"20211210_r_JV43_intentional_unilateral_set1.txt"

# *** channel selection params ***
selected_channels = ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
inverse = False

# EMG sampling rate 
fsamp_emg = 2000 # in Hz

# *** processing parameters *** 
target_frequency = 500 # target frequency after downsampling 


# *********************************************************************************
# ************************* Load EMG data  ****************************************
# *********************************************************************************

# loading emg data 
emg_data, emg_time_axis, emg_ch_names = emg_lib.loadCometaEMGData(file_str)


# *********************************************************************************
# ************************* Process EMG data  *************************************
# *********************************************************************************


# channel selection, if inverse = False all channels specified are kept 
emg_data_select, emg_ch_names_select = emg_lib.channelSelection(emg_data, emg_ch_names, selected_channels, inverse)

#emg_lib.showEMGData(emg_data_select, emg_time_axis, emg_ch_names_select)

# downsampling to target frequency 
emg_data_down, time_axis_down = emg_lib.decimateEMGData(emg_data_select, emg_time_axis, target_frequency, fsamp_emg)

#emg_lib.showEMGData(emg_data_down, time_axis_down, emg_ch_names_select)

n_var = int(680/4)
emg_data_filtered = emg_lib.applyVarianceFilter(emg_data_down, n_var)

emg_lib.showEMGData(emg_data_filtered, time_axis_down, emg_ch_names_select)

