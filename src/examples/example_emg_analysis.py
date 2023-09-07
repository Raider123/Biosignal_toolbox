
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 

# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__)) # project path 
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 


# own libs / classes 
from emg_lib import EMGData

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


# Create an string with dataset file name (ANT and Cometa)
file_str = "21032023_AJ80D_curls_with_pause_set1.txt"


# *** channel selection params (comment in for channel selection) ***
#selected_channels =  []# ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
inverse = False # define True for channel selection


# for ANT Systems 
emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
selected_channels = []


# EMG sampling rate (Cometa default)
#fsamp_emg = 2000 # in Hz

fsamp_emg = 500 # for ANT

# *** processing parameters (if used comment in) *** 
target_frequency = 500 # target frequency after downsampling (Cometa)

# variance filter length 
n_var = 170 # calc length of variance filter (default 170, tested for 500 Hz sampling rate)

# *********************************************************************************
# ************************* Load and process EMG data  ****************************
# *********************************************************************************

# define the processing params
 
load_data = 1 # 0 for Cometa, 1 vor ANT
apply_var_filter = 0 # 0 -> no, 1 -> yes
apply_bp_filter = 1 # 0 -> no, 1 -> yes
plot_emg = 2 # 0 -> no plot, 1 -> plot raw data, 2 -> plot filtered data

# create EMGData Object

emg_data_obj = EMGData(load_data, data_path, file_str, fsamp_emg, emg_ch_names, apply_var_filter, apply_bp_filter, plot_emg, selected_channels, inverse, target_frequency, n_var)

