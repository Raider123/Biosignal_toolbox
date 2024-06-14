# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
import numpy as np 

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

# train_file_list = ["20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set2.vhdr"]
# val_test_file_list = ["20211210_r_JV43_intentional_unilateral_set3.vhdr"]


# names for saving 
scenario_name = "intentional_unilateral"
result_file_name = "zeroGustavMedianCorrCut"


# filter model name 
filter_model_name_05_4Hz = "pooling_model_filterNet_05_04Hz_1000ms"
filter_model_name_05_40Hz = "pooling_model_filterNet_05_040Hz_1000ms"


# *********************************************************************************
# ************** Network parameters ***********************************************
# *********************************************************************************

#machine learning params
num_classes = 2

# fcn model parameter 
n_epochs = 200 #300 training epochs (max since early stopping is used)
n_batch_size_EEGNet = 16 # 16 for EEGNet
n_batch_size_MLP = 16 # 64 #64 for MLP
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
early_stopping_patience = 70 # 50 
# reduce_patients = 1

#EEGNet-parameter
kern_length_EEGNET = 50 # 50 before 
F1 = 8 # 8 
D = 2 
F2 = 16 # 16 
dropout_EEGNet = 0.5

# n_xDAWN = 4 # if used 

# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 
metrics = "accuracy"

# training windows and features
train_windows = ["bis-2500", "bis-1900", "bis-2300", "bis-2000", "bis-1700", "bis-1500", "bis-100", "bis-80", "bis-60", "bis-40", "bis-20", "bis0"]#, "bis-50", "bis0"] # alternatively 
window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]# alternative 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # 900, 1000 numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
use_norm_layer = True # use the input norm layer 

show_train_results = False

# validation trials used for performance evaluation 
n_test_trials = 20 

# window wise metric evaluation
window_size = 3000 # windowsize in ms (analog to pySPACE evaluation) + add 100 ms for cutting after filtering 
window_step = 20 # stepsize in ms (analog to pySPACE evaluation)

f_samp_eeg = 500.0 #sample Frequency of eeg
marker_number = 22 # onset markernumber (Qualisys)
error_number = 3 # number of the error marker 


# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
#channel_list = [] # do not drop channels
channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]


# just remap the parameters (need to be adapted)
t1 = -7.0
t2 = 0.0 # to cut this off later


# filter settings 
f_highpass = 0.5
f_lowpass_MLP = 4.0
f_lowpass_EEGNet = None

#  n_moving average 
# n_moving_ave = 30 

# switch between online and offline preprocessing 
use_offline_processing = False
use_net = False # use the autoencoder net for preprocessing 


# train_test iterations (mapping of train test sets)
train_test_conditions = [
    #extra one
    {"train": ["31102023_BR60D_unilateral_set1.vhdr"], "test": ["31102023_BR60D_unilateral_set2.vhdr"]}, 
    {"train": ["31102023_BR60D_unilateral_set2.vhdr"], "test": ["31102023_BR60D_unilateral_set1.vhdr"]},
    # JV43 
    {"train": ["20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set2.vhdr"], "test": ["20211210_r_JV43_intentional_unilateral_set3.vhdr"]},
    {"train": ["20211210_r_JV43_intentional_unilateral_set2.vhdr", "20211210_r_JV43_intentional_unilateral_set3.vhdr"], "test": ["20211210_r_JV43_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set3.vhdr"], "test": ["20211210_r_JV43_intentional_unilateral_set2.vhdr"]}, 
    # RA12
    {"train": ["20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set2.vhdr"], "test": ["20211216_r_RA12_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20211216_r_RA12_intentional_unilateral_set2.vhdr", "20211216_r_RA12_intentional_unilateral_set3.vhdr"], "test": ["20211216_r_RA12_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set3.vhdr"], "test": ["20211216_r_RA12_intentional_unilateral_set2.vhdr"]},
    # AV82 
    {"train": ["20211220_r_AV82_intentional_unilateral_set1.vhdr", "20211220_r_AV82_intentional_unilateral_set2.vhdr"], "test": ["20211220_r_AV82_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20211220_r_AV82_intentional_unilateral_set2.vhdr", "20211220_r_AV82_intentional_unilateral_set3.vhdr"], "test": ["20211220_r_AV82_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20211220_r_AV82_intentional_unilateral_set1.vhdr", "20211220_r_AV82_intentional_unilateral_set3.vhdr"], "test": ["20211220_r_AV82_intentional_unilateral_set2.vhdr"]},
    # UP28 
    {"train": ["20211223_r_UP28_intentional_unilateral_set1.vhdr", "20211223_r_UP28_intentional_unilateral_set2.vhdr"], "test": ["20211223_r_UP28_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20211223_r_UP28_intentional_unilateral_set2.vhdr", "20211223_r_UP28_intentional_unilateral_set3.vhdr"], "test": ["20211223_r_UP28_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20211223_r_UP28_intentional_unilateral_set1.vhdr", "20211223_r_UP28_intentional_unilateral_set3.vhdr"], "test": ["20211223_r_UP28_intentional_unilateral_set2.vhdr"]},
    # ZS27 
    {"train": ["20220104_r_ZS27_intentional_unilateral_set1.vhdr", "20220104_r_ZS27_intentional_unilateral_set2.vhdr"], "test": ["20220104_r_ZS27_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20220104_r_ZS27_intentional_unilateral_set2.vhdr", "20220104_r_ZS27_intentional_unilateral_set3.vhdr"], "test": ["20220104_r_ZS27_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20220104_r_ZS27_intentional_unilateral_set1.vhdr", "20220104_r_ZS27_intentional_unilateral_set3.vhdr"], "test": ["20220104_r_ZS27_intentional_unilateral_set2.vhdr"]},
    # JD68 
    {"train": ["20220105_r_JD68_intentional_unilateral_set1.vhdr", "20220105_r_JD68_intentional_unilateral_set2.vhdr"], "test": ["20220105_r_JD68_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20220105_r_JD68_intentional_unilateral_set2.vhdr", "20220105_r_JD68_intentional_unilateral_set3.vhdr"], "test": ["20220105_r_JD68_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20220105_r_JD68_intentional_unilateral_set1.vhdr", "20220105_r_JD68_intentional_unilateral_set3.vhdr"], "test": ["20220105_r_JD68_intentional_unilateral_set2.vhdr"]},
    # QS70 
    {"train": ["20220107_r_QS70_intentional_unilateral_set1.vhdr", "20220107_r_QS70_intentional_unilateral_set2.vhdr"], "test": ["20220107_r_QS70_intentional_unilateral_set3.vhdr"]}, 
    {"train": ["20220107_r_QS70_intentional_unilateral_set2.vhdr", "20220107_r_QS70_intentional_unilateral_set3.vhdr"], "test": ["20220107_r_QS70_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20220107_r_QS70_intentional_unilateral_set1.vhdr", "20220107_r_QS70_intentional_unilateral_set3.vhdr"], "test": ["20220107_r_QS70_intentional_unilateral_set2.vhdr"]},
    # XP01 
    {"train": ["20211222_r_XP01_intentional_unilateral_set1.vhdr", "20211222_r_XP01_intentional_unilateral_set2.vhdr"], "test": ["20211222_r_XP01_intentional_unilateral_set3.vhdr", "20211222_r_XP01_intentional_unilateral_set4.vhdr"]}, 
    {"train": ["20211222_r_XP01_intentional_unilateral_set2.vhdr", "20211222_r_XP01_intentional_unilateral_set3.vhdr", "20211222_r_XP01_intentional_unilateral_set4.vhdr"], "test": ["20211222_r_XP01_intentional_unilateral_set1.vhdr"]}, 
    {"train": ["20211222_r_XP01_intentional_unilateral_set1.vhdr", "20211222_r_XP01_intentional_unilateral_set3.vhdr", "20211222_r_XP01_intentional_unilateral_set4.vhdr"], "test": ["20211222_r_XP01_intentional_unilateral_set2.vhdr"]},
    # test on extra one 
    ]

# 31102023_BR60D_unilateral_set1

# print(train_test_conditions[0]["train"])





