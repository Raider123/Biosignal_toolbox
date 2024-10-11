# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model, MLP_Model_reduced


print(tf.config.experimental.list_physical_devices('GPU'))

#from Neural_Network_Movement_Predictions_EEG.live.userparams_train import * # --> parameters are stored and fully imported from this script 

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

# use LSL file recorded # "BR60D_bilateral_exo_vr_set2_data", "BR60D_bilateral_exo_vr_set3_data", 
train_file_LSL = ["BR60D_bilateral_exo_vr_set6_data", "BR60D_bilateral_exo_vr_set7_data", "BR60D_bilateral_exo_vr_set8_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]

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
train_windows = ["bis-2200", "bis-1900", "bis-2100", "bis-2000", "bis-1700", "bis-1500", "bis-100", "bis-80", "bis-60", "bis-40", "bis-20", "bis0"]#, "bis-50", "bis0"] # alternatively 
window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]# alternative 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
#feature_indices_windows = np.arange(900, 1000, step = 4) # (900, 1000) means last 100 ms of a window are used 

# reduced feature number 
#feature_indices_windows = np.array([900, 910, 920, 930, 940, 950, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 998, 999])
feature_indices_windows = np.array([900, 920, 940, 960, 965, 970, 975, 980, 985, 990, 995, 999])
use_norm_layer = True # use the input norm layer 

# validation trials used for performance evaluation 
n_val_trials = 5

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

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 


# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []


# measure execution time
time_start = perf_counter()

# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

#  loading and epoching for training   
#data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)

EEG_train, EEG_val = EEG_data.splitTrainTestEpochs(n_test_epochs=n_val_trials) # split in train and val_test 
#EEG_val, EEG_test = EEG_val_test.splitTrainTestEpochs(n_test_epochs=5) # split into val and test 


channel_names = EEG_data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************


# first stage processing for both methods and train validation data 
EEG_data_train = pipeline.firstStageOnlinePreprocessing(EEG_train, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
EEG_data_val = pipeline.firstStageOnlinePreprocessing(EEG_val, window_size = window_size, window_step = window_step, windows_selected = train_windows) 


# filter specifications 
sos_MLP = EEG_data_train.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos") #--> good one 
sos_EEGNet = EEG_data_train.designFilter(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")

# # train validation data splitting 
# EEG_train_MLP = EEG_train # time domain feates MLP
# EEG_train_freq_MLP = copy.deepcopy(EEG_train_MLP) # for frequency features of MLP
# EEG_train_EEGNet = copy.deepcopy(EEG_train_MLP) # for EEGNet

# EEG_val_MLP = EEG_val # time domain feates MLP
# EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features of MLP
# EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet


# get features by running processing pipeline 
x_train_MLP, y_train_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_train), copy.deepcopy(EEG_data_train), window_labels_train, feature_indices_windows, sos_MLP)
x_val_MLP, y_val_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_val), copy.deepcopy(EEG_data_val), window_labels_train, feature_indices_windows, sos_MLP)


# Load model with norm layer  
MLP = MLP_Model_reduced(x_train_MLP, use_norm_layer = use_norm_layer)
MLP_model = MLModel(model = MLP, type= "keras")
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_MLP_"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train_MLP, y_train= y_train_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_train_MLP, labels = y_train_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_MLP = MLP_model.getPerfResults()

# EEGNet processing pipeline 
x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_train), window_labels_train, num_classes, sos_EEGNet)
x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_val), window_labels_train, num_classes, sos_EEGNet)


#shape_input = EEG_train_EEGNet.getWindows().shape # get train data shape for network 
shape_input = x_train_EEGNet.shape
print("EEGNet Input shape", shape_input)
model_EEGNet = EEGNet(nb_classes=num_classes,Chans=shape_input[1], Samples=shape_input[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_train_EEGNet, use_norm_layer = use_norm_layer)
EEGNet_model = MLModel(model = model_EEGNet, type="keras")
EEGNet_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_EEGNet"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_EEGNet, class_weights=None, x_train=x_train_EEGNet, y_train= y_train_EEGNet, x_val = x_val_EEGNet, y_val = y_val_EEGNet, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)

# # predict and get results 
print("predict EEGNet")
EEGNet_model.predict(data = x_train_EEGNet, labels = y_train_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_EEGNet = EEGNet_model.getPerfResults()

# perf results 
perf_results_total_MLP.append(perf_results_MLP[0:3])
perf_results_total_EEGNet.append(perf_results_EEGNet[0:3])

# save res 
np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


