# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
import numpy as np 

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

# use LSL file recorded 
train_file_LSL = ["BR60D_bilateral_exo_vr_set2_data", "BR60D_bilateral_exo_vr_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]

# subject params 
subject = "current"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
iteration = 0 # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "_live"

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

# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 
metrics = "accuracy"


# training windows and features
train_windows = ["bis-2500", "bis-1900", "bis-2300", "bis-2000", "bis-1700", "bis-1500", "bis-100", "bis-80", "bis-60", "bis-40", "bis-20", "bis0"]#, "bis-50", "bis0"] # alternatively 
window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]# alternative 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
#feature_indices_windows = np.arange(900, 1000, step = 4) # (900, 1000) means last 100 ms of a window are used 

# reduced feature number 
#feature_indices_windows = np.array([900, 910, 920, 930, 940, 950, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 998, 999])
feature_indices_windows = np.array([900, 920, 940, 960, 965, 970, 975, 980, 985, 990, 995, 999])
use_norm_layer = True # use the input norm layer 

# validation trials used for performance evaluation 
n_val_trials = 10 

# window wise metric evaluation
window_size = 1000 # windowsize in ms (analog to pySPACE evaluation) + add 100 ms for cutting after filtering 
window_step = 20 # stepsize in ms (analog to pySPACE evaluation)

f_samp_eeg = 500.0 #sample Frequency of eeg
marker_number = 20 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 3 # number of the error marker 

# eeg stream params 
#channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
channel_names = ["FC3", "FC1", "C3", "C1", "CZ", "CP3", "CP1", "CPZ", "CCP1h", "FCC1h", "CCP3h", "FCC3h"]
# time selection for epoching of the data 

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
channel_list = [] # do not drop channels
#channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]

# just remap the parameters (need to be adapted)
t1 = -5.0
t2 = 0.0

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************