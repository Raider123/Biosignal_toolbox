# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 
import os 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
filenames = ["BR60D_bilateral_exo_vr_set1_data"]

f_samp_eeg = 500.0
#channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
channel_names = ["FC3", "FC1", "C3", "C1", "CZ", "CP3", "CP1", "CPZ", "CCP1h", "FCC1h", "CCP3h", "FCC3h"]
data_path = proj_path+"/data/"
# *********************************************************************************
# *********************************************************************************
# *********************************************************************************


EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = filenames, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

events = EEG_data.getEvents()
print("events recoded: (index, 0, markernumber)")
print(events.shape)
print(events)

