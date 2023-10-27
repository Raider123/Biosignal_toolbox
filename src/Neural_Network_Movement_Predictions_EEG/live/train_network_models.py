# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
#sys.path.append(proj_path+"/lib/biosignal_toolbox") # path to lib folder

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
train_file_list = ["20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set2.vhdr"]

# subject params 
subject = "JV43"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
iteration = 0 # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "live_train_results"

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
train_windows = ["bis-2700", "bis-2500", "bis-2300" ,"bis-2100", "bis-1900", "bis-1700", "bis-1500", "bis-1300" ,"bis-1000", "bis-150", "bis-100", "bis-50", "bis0"] #["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"] # ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] # [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
#window_labels_test = [0.0, 0.0, 1.0, 1.0] #  [0.0, 0.0, 1.0, 1.0]

#used_trials_training = 80 # trials to use for training 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # 900, 1000 numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 

# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

f_samp_eeg = 500 #sample Frequency of eeg
marker_number = 100 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 3 # number of the error marker 

# specifying the movement onset marker 
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -5.0 # time in seconds (start epoch)
epoching_time_after_onset = 0.0 # time in seconds (0 = movement onset)

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
channel_list = ["x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]

# just remap the parameters (need to be adapted)
t1 = epoching_time_before_onset
t2 = epoching_time_after_onset


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
data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
data_train.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel, event_id_used, t1, t2)

# get the epochs after  --> required ? 
# time_axis_eeg_batch, lrp_epochs_train = data_train.getEpochs()
# remaining_eeg_channel_names = data_train.getChannelNames()


# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************
EEG_train = data_train

# window EEG epochs 
EEG_train.windowEEGEpochs(window_size, window_step)
#EEG_val.windowEEGEpochs(window_size, window_step)

# window selection 
EEG_train.windowSelection(train_windows)
#EEG_val.windowSelection(test_windows) 

EEG_val_MLP = EEG_train # time domain feates MLP
EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features of MLP
EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet

#     # ******** MLP processing *******************

# bandpass filter data 
EEG_val_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 

# # specify the window labels (not needed)
EEG_val_MLP.setWindowLabels(window_labels_train)
EEG_val_EEGNet.setWindowLabels(window_labels_train)

# time domain features (MLP)
EEG_val_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
EEG_val_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")

# feauture combination 
x_val_freq = EEG_val_freq_MLP.getFeatures() # get features of freq
EEG_val_MLP.addFeatures(x_val_freq) # add frequency domain features 

# input features network 
x_val_MLP = EEG_val_MLP.getFeatures()
y_val_MLP = EEG_val_MLP.getTrainLabels()

# Load model with norm layer  
MLP = MLP_Model(x_val_MLP, use_norm_layer = True)
MLP_model = MLModel(model = MLP, train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_val_MLP, y_train= y_val_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
MLP_model.trainModel(save_trained_model = True, model_filename =subject+"_"+scenario_name+result_file_name+"_model_"+str(iteration))

# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_val_MLP, labels = y_val_MLP, encoding = "binary", show_results = True, show_pred_time = False)
perf_results_MLP = MLP_model.getPerfResults()


# *********** EEGNet processing *******************
EEG_val_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 
# reshape windows for net
EEG_val_EEGNet.reshapeWindowsForCNNnets()
EEG_val_EEGNet.labelsToCategorical(num_classes = num_classes) 

# get train windows 
x_val_EEGNet = EEG_val_EEGNet.getWindows()
y_val_EEGNet = EEG_val_EEGNet.getTrainLabels()

train_wind_shape = x_val_EEGNet.shape # get train data shape for network 
model_EEGNet = EEGNet(nb_classes=num_classes, Chans=train_wind_shape[1], Samples=train_wind_shape[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_val_EEGNet, use_norm_layer = True)
EEGNet_model = MLModel(model = model_EEGNet, train_epochs= n_epochs, batch_size=n_batch_size_EEGNet, class_weights=None, x_train=x_val_EEGNet, y_train= y_val_EEGNet, x_val = x_val_EEGNet, y_val = y_val_EEGNet, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
EEGNet_model.trainModel(save_trained_model = True, model_filename =subject+"_"+scenario_name+result_file_name+"_model_"+str(iteration))

# # predict and get results 
EEGNet_model.predict(data = x_val_EEGNet, labels = y_val_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = False, eval_type = "online")
perf_results_EEGNet = EEGNet_model.getPerfResults()
    
# perf results 
perf_results_total_MLP.append(perf_results_MLP[0:3])
perf_results_total_EEGNet.append(perf_results_MLP[0:3])

# save res 
np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)



