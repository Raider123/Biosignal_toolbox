# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy 
from time import perf_counter_ns

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
scenario_name = "intentional_unilateral"
preprocessed_data_filename_end = "34ch_raw_no_scale"  #"34ch_raw_no_scale" # TODO: implement online filter and normalization  
#eval_name = "fcn_network_results_34ch_MLP_scalings_test"

# model names 
MLP_eval_name = "fcn_network_results_34ch_MLP_online_pre_test_reduced_net"
EEGNet_eval_name = "fcn_network_results_34ch_EEGNet_online_pre_test_reduced_net"

f_samp_eeg = 500 #sample Frequency of eeg


# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 

metrics = "accuracy"

# training windows and features
use_all_windows = True
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_test = [0.0, 0.0, 1.0, 1.0]
n_classes = 2
decision_bound = 0.7

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 


# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)


#********** Load models for subject ***********

# example data 
subject = "JV43"
iteration = 0

# load MLP model 
MLP_model = MLModel() 
MLP_model.loadModel(path =data_path, filename = subject+"_"+scenario_name+MLP_eval_name+"_model_"+str(iteration))

# load EEGNet model 
model_EEGNet = MLModel()
model_EEGNet.loadModel(path =data_path, filename = subject+"_"+scenario_name+EEGNet_eval_name+"_model_"+str(iteration))


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# load each individual train, val and test sets (preprocessed)
lrp_epochs_val_scaled_start = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
channel_names = np.load(data_path+"remaining_eeg_channel_names"+".npy")

for trial_idx in range(0, 20): 

    lrp_epochs_val_scaled_cut = lrp_epochs_val_scaled_start[trial_idx, :, : ]
    lrp_epochs_val_scaled = copy.deepcopy(lrp_epochs_val_scaled_start)
    lrp_epochs_val_scaled = np.expand_dims(lrp_epochs_val_scaled_cut, axis=0) # fit shape of trials                       
    #print(lrp_epochs_val_scaled.shape)

    # **********************************************************************************
    # ********************* Preprocessing for data of both networks ********************
    # **********************************************************************************
    #load
    EEG_val = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_val_scaled, f_samp = f_samp_eeg, channel_names = list(channel_names))

    # window EEG epochs (use only one window for testing)
    EEG_val.windowEEGEpochs(window_size, window_step)
    one_window = EEG_val.windows[:, :, :, -1] # select one window for now 
    EEG_val.windows = np.expand_dims(one_window, axis=3) # expand window dim to one for standard format 
    print("windows shape:", EEG_val.windows.shape)


    # **********************************************************************************
    # ********************* Here starts the online part  *******************************
    # **********************************************************************************

    t_copy_1 = perf_counter_ns()
    EEG_val_MLP = EEG_val # time domain feates MLP
    EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features of MLP
    EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet

    t_copy_2 = perf_counter_ns()

#     # ******** MLP processing *******************
    
    # bandpass filter data 
    EEG_val_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 
    
    # # specify the window labels (not needed)
    # EEG_val_MLP.setWindowLabels(window_labels_test)
    # EEG_val_EEGNet.setWindowLabels(window_labels_test)

    # time domain features (MLP)
    EEG_val_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_val_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")
    
    # feauture combination 
    x_val_freq = EEG_val_freq_MLP.getFeatures() # get features of freq
    EEG_val_MLP.addFeatures(x_val_freq) # add frequency domain features 
    

    # input features network 
    x_val_MLP = EEG_val.getFeatures()
    # y_val_MLP = EEG_val.getTrainLabels()

    t_pre_MLP_2 = perf_counter_ns()


#     # *********** EEGNet processing *******************

    EEG_val_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 
    # reshape windows for net

    #EEG_val_EEGNet.reshapeWindowsForCNNnets()

    # EEG_val_EEGNet.labelsToCategorical(num_classes = n_classes) # not needed

    # get train windows 
    x_val_EEGNet = EEG_val_EEGNet.getWindows()

    # y_val_EEGNet = EEG_val_EEGNet.getTrainLabels()

    t_pre_EEGNet_2 = perf_counter_ns()


#     # ********** make model prediction  ***********

#     # predict and get results 
    MLP_model.predict(data = x_val_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = True, eval_type = "online")

    # # predict and get results 
    model_EEGNet.predict(data = x_val_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = True, eval_type = "online")

    # postprocessing 
    prod_score = model_EEGNet.prediction_scores[1] *MLP_model.prediction_scores[0] # final output score 

    t_predict_2 = perf_counter_ns()
    
    # eval times 
    t_pre_MLP_1  = t_copy_2 # same time 
    t_pre_EEGNet_1 = t_pre_MLP_2
    t_predict_1 = t_pre_EEGNet_2

    t_pre_MLP = (t_pre_MLP_2-t_pre_MLP_1)/1000000
    t_pre_EEGNet = (t_pre_EEGNet_2-t_pre_EEGNet_1)/1000000
    t_predict = (t_predict_2 -t_predict_1)/1000000

    print("preprocessing time MLP in ms: ", t_pre_MLP)
    print("preprocessing time EEGNet in ms: ", t_pre_EEGNet)
    print("prediction time (both) in ms: ", t_predict)
    print("total time in ms:", t_pre_MLP+t_pre_EEGNet+t_predict)#

