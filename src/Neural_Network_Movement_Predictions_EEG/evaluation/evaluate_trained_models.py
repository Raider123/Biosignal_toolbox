# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter_ns
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
#sys.path.append(proj_path+"/lib/biosignal_toolbox") # path to lib folder

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
tf.config.set_visible_devices([], 'GPU') 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
scenario_name = "intentional_unilateral"
preprocessed_data_filename_end = "34ch_raw_ica"  #"34ch_raw_no_scale" # TODO: implement online filter and normalization  
#eval_name = "fcn_network_results_34ch_MLP_scalings_test"

# model names 
MLP_eval_name = "fcn_network_results_34ch_MLP_ica_test"
EEGNet_eval_name = "fcn_network_results_34ch_EEGNet_ica_test"

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

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 1) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 


# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# example data 
subject = "RA12"
iteration = 0

# load each individual train, val and test sets (preprocessed)
lrp_epochs_val_scaled_start = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
channel_names = np.load(data_path+"remaining_eeg_channel_names"+".npy")

for trial_idx in range(0, 20): 

    lrp_epochs_val_scaled_cut = lrp_epochs_val_scaled_start[trial_idx, :, : ]
    lrp_epochs_val_scaled = copy.deepcopy(lrp_epochs_val_scaled_start)
    lrp_epochs_val_scaled = np.expand_dims(lrp_epochs_val_scaled_cut, axis=0) # fit shape of trials                       
    print(lrp_epochs_val_scaled.shape)

    
    # **********************************************************************************
    # ********************* Preprocessing for data of both networks ********************
    # **********************************************************************************
    #load
    EEG_val = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_val_scaled, f_samp = f_samp_eeg, channel_names = list(channel_names))

    # window EEG epochs 
    EEG_val.windowEEGEpochs(window_size, window_step)

    if (use_all_windows): # if all windows used 
        test_windows = EEG_val.getWindowNames()
        window_labels_test = np.zeros((81))
        window_labels_test[-4:] = 1.0

    # window selection 
    EEG_val_MLP = EEG_val
    EEG_val_MLP.windowSelection(test_windows) 
    EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features 
    EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet


    # ******** MLP processing *******************

    # bandpass filter data 
    EEG_val_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 

    # # specify the window labels 
    EEG_val_MLP.setWindowLabels(window_labels_test)
    EEG_val_EEGNet.setWindowLabels(window_labels_test)

    # time domain features (MLP)
    EEG_val_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
    EEG_val_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")

    
    # feauture combination 
    x_val_freq = EEG_val_freq_MLP.getFeatures() # get features of freq
    EEG_val_MLP.addFeatures(x_val_freq) # add frequency domain features 
    

    # input features network 
    x_val_MLP = EEG_val.getFeatures()
    y_val_MLP = EEG_val.getTrainLabels()


    # *********** EEGNet processing *******************

    EEG_val_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 
    # reshape windows for net 
    EEG_val_EEGNet.reshapeWindowsForCNNnets()

    EEG_val_EEGNet.labelsToCategorical(num_classes = n_classes)
    # get train windows 
    x_val_EEGNet = EEG_val_EEGNet.getWindows()
    y_val_EEGNet = EEG_val_EEGNet.getTrainLabels()


    # ********** Load and apply models ***********

    MLP_model = MLModel() 
    MLP_model.loadModel(path =data_path, filename = subject+"_"+scenario_name+MLP_eval_name+"_model_"+str(iteration))

    # predict and get results 
    MLP_model.predict(data = x_val_MLP, labels = y_val_MLP, encoding = "binary", show_results = True, show_pred_time = True)

    # model setup and training 

    model_EEGNet = MLModel()
    model_EEGNet.loadModel(path =data_path, filename = subject+"_"+scenario_name+EEGNet_eval_name+"_model_"+str(iteration))

    # # predict and get results 

    model_EEGNet.predict(data = x_val_EEGNet, labels = y_val_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = True)

    # evaluate an ensemble model of both predictions 
    scores_EEGNet = model_EEGNet.getPredictionScores()
    scores_MLP = MLP_model.getPredictionScores()


    # show single trial scores 
    fig = plt.figure()
    plt.scatter(np.arange(0, len(scores_EEGNet)), scores_EEGNet)
    plt.scatter(np.arange(0, len(scores_EEGNet)), scores_MLP)
    plt.plot(np.arange(0, len(scores_EEGNet)), (scores_EEGNet+scores_MLP)/2, color = "green", linewidth=2.0)
    plt.plot(np.arange(0, len(scores_EEGNet)), (scores_EEGNet*scores_MLP), color = "purple", linewidth=4.0, marker='o', linestyle='solid')
    plt.axhline(y=0.7, xmin=0, xmax=81, linestyle="--")
    plt.legend(["EEGNet", "MLP", "Average", "Multiply"])
    plt.ylim((0.0, 1.0))
    plt.xlabel("Windows")
    plt.ylabel("Probability")
    plt.title("Single trial probability (ensemble model)")
    #fig.savefig("ensemble_models.png")

    # mean_scores = (scores_EEGNet*scores_MLP)

    # ens_model = copy.deepcopy(MLP_model) 
    # ens_model.calcPerformance(predictions = mean_scores, labels = y_val_MLP, encoding = "binary", show_results = True)

plt.show()
