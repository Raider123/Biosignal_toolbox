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
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
tf.config.set_visible_devices([], 'GPU') # disable now 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"


data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

# use LSL file recorded 
train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]


# subject params 
subject = "current"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
iteration = 0 # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "_live"


#machine learning params
num_classes = 2

# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size_EEGNet = 16 # 16 for EEGNet
n_batch_size_MLP = 64 #64 for MLP

weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
early_stopping_patience = 100 # 50 


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
#train_windows = ["bis-2500", "bis-2300" ,"bis-2100", "bis-1900", "bis-1700", "bis-1300" ,"bis-1000", "bis-150", "bis-100", "bis-50", "bis0"] #["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]
train_windows = ["bis-2500", "bis-1900", "bis-1500" ,"bis-1200", "bis-150", "bis-100", "bis-50", "bis0"]#, "bis-50", "bis0"] # alternatively 
#test_windows = ["bis-2500", "bis-2050", "bis-150", "bis-100"] # ["bis-2500", "bis-2050", "bis-100", "bis0"]

#window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]# 
window_labels_train = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]# alternative 
#window_labels_test = [0.0, 0.0, 1.0, 1.0] #  [0.0, 0.0, 1.0, 1.0]


features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # 900, 1000 numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
use_norm_layer = True # use the input norm layer 

# validation trials used for performance evaluation 
n_val_trials = 5 

# window wise metric evaluation
window_size = 1100 #windowsize in ms (analog to pySPACE evaluation) # testweise 
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

f_samp_eeg = 500.0 #sample Frequency of eeg
marker_number = 22 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 3 # number of the error marker 
# eeg stream params 
channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
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


# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []

# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

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

EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)

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
EEG_train = pipeline.firstStagePreprocessing(EEG_train, window_size, window_step, train_windows)
EEG_val = pipeline.firstStagePreprocessing(EEG_val, window_size, window_step, train_windows)


# train validation data splitting 
EEG_train_MLP = EEG_train # time domain feates MLP
EEG_train_freq_MLP = copy.deepcopy(EEG_train_MLP) # for frequency features of MLP
EEG_train_EEGNet = copy.deepcopy(EEG_train_MLP) # for EEGNet

EEG_val_MLP = EEG_val # time domain feates MLP
EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features of MLP
EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet


# get features by running processing pipeline 
x_train_MLP, y_train_MLP = pipeline.MLPProcessing(EEG_train_MLP, EEG_train_freq_MLP, window_labels_train, feature_indices_windows)
x_val_MLP, y_val_MLP = pipeline.MLPProcessing(EEG_val_MLP, EEG_val_freq_MLP, window_labels_train, feature_indices_windows)


# Load model with norm layer  
MLP = MLP_Model(x_train_MLP, use_norm_layer = use_norm_layer)
MLP_model = MLModel(model = MLP, type= "keras")
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_MLP_"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train_MLP, y_train= y_train_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_train_MLP, labels = y_train_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_MLP = MLP_model.getPerfResults()

# EEGNet processing pipeline 
x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessing(EEG_train_EEGNet, window_labels_train, num_classes)
x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessing(EEG_val_EEGNet, window_labels_train, num_classes)


shape_input = EEG_train_EEGNet.getWindows().shape # get train data shape for network 
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


