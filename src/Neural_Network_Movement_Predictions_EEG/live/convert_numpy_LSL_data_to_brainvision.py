import matplotlib.pyplot as plt
import numpy as np

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# path setting 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# ***********************************************
# ************ user params **********************
# ***********************************************

LSL_filenames = ["XY90_unilateral_set4_data"]
target_filename = "test4"

# eeg params 
channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
f_samp_eeg = 500.0 #sample Frequency of eeg


# ***********************************************
# ************ user params end ******************
# ***********************************************

EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = LSL_filenames, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
EEG_data.mneRawToBrainvision(folder = data_path, filename = target_filename)







