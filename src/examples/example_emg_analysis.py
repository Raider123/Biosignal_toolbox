
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import os 

# project path settings 
#proj path 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = os.path.join(proj_path, 'data/') # path where the data lays 


# # own libs 
from biosignal_toolbox.emg_lib import EMGData


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


# Create an string with dataset file name (ANT and Cometa)
file_str = ["20211223_r_UP28_intentional_unilateral_set2.txt"]


# *** channel selection params (comment in for channel selection) ***
#selected_channels =  []# ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
#inverse = False # define True for channel selection


# for ANT Systems 
#emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
#selected_channels = ["EMG1", "EMG2"]


# EMG sampling rate

#fsamp_emg = 500 # for ANT

# *** processing parameters (if used comment in) *** 
# target_frequency = 500 # target frequency after downsampling (Cometa)

# # variance filter length 
n_var = 50 # calc length of variance filter (default 170, tested for 500 Hz sampling rate)

# # filter settings
# f_high = 20 # in Hz
# f_low = 245 # in Hz

# *********************************************************************************
# ************************* Load and process EMG data  ****************************
# *********************************************************************************

# define the processing params

# create EMGData Object
EMG = EMGData(data_path = data_path, format = "cometa", filenames = file_str)

#downsampling 
#EMG.mneRawMethod("resample", sfreq = 500.0) # show EMG data in mV (default unit is uV)

# filter raw data with variance filter 
#EMG.applyVarianceFilter(n_var=n_var, apply_to_structures="raw")

# should be replaced later 
#EMG.showEMGData()

# or use just mne's plot methods
EMG.mneRawMethod("plot", scalings = dict(emg=1e+3), n_channels = 8) # show EMG data in mV (default unit is uV)
plt.show()


