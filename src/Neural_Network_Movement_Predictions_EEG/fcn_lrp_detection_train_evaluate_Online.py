# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from time import perf_counter
import sys


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib

# models 
from EEGModels import EEGNet

# own model 
from MlpErp import MLP_Model


# disable GPU for testing
tf.config.set_visible_devices([], 'GPU')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43","RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "fcn_network_results_34ch_time_freq_domain_feat"
preprocessed_data_filename_end = "34ch_05_4Hz"
preprocessed_data_filename_end_f = "34ch_05_40Hz"

f_samp_eeg = 500 #sample Frequency of eeg


#machine learning params

num_classes = 2

# fcn model parameter 
n_epochs = 300 #30 training epochs (max since early stopping is used)
n_batch_size = 64 # batch size # 64 seems to work nice
shuffle_data = True # shuffle all data for training, validation and testing
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
show_training_results = False
multiprocessing_cpus = 16
early_stopping_patience = 100

#EEGNet-parameter
# dropoutRate = 0.5
# kernLength = 60
# F1 = 8
# D = 2
# F2 = 16
# # make automatic or something 
# kernels, chans, samples = 1, 34, 2500 # 


# training params 
loss_fcn = "binary_crossentropy"
optimizer = "Nadam"
metrics = "accuracy"


# training windows and features
train_windows = ["bis-2500", "bis-2050", "bis-2200", "bis-100", "bis-50", "bis0"]
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_train = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
window_labels_test = [0.0, 0.0, 1.0, 1.0]

feature_times_windows = (900, 1000) # time inside the windows to be used as features (last 200 ms)
# feature types used for generating training data 
feature_type1 = "timepoints" 
feature_type2 = "meanfreqs"

features_used = "fusion" # which features to be used for classification 

# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

# specify params for metric evaluation with method "relabelling", otherwise the parameters are not relevant if use_relabelling = False
searching_bounds = [61, 81] # boundaries where the "label change point" is determined, values are the numbers of the windows (see wind_names param for which windows are selected as bounds)



# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

# init performance results list
perf_results_total = []
max_val_indices = []

# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# measure execution time
time_start = perf_counter()


# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=early_stopping_patience,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# checkpoint = tf.keras.callbacks.ModelCheckpoint(results_path+'model_file.h5', 
#                     monitor="val_accuracy", mode="max", 
#                     save_best_only=True, verbose=1)


for subject in subject_names:
    for iteration in interations:
        
        # *********************************************************************************
        # ***************** Load train, test, val sets for every iteration ****************
        # *********************************************************************************

        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_train_"+str(iteration)+".npy")
        lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
        lrp_epochs_test_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_val_"+str(iteration)+".npy")


        # *********************************************************************************
        # ********************* Prepare data for network **********************************
        # *********************************************************************************

        
        # windowing of the data 
        train_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_train_scaled, f_samp_eeg, window_size, window_step)
        val_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_val_scaled, f_samp_eeg, window_size, window_step)
        test_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_test_scaled, f_samp_eeg, window_size, window_step)


        # train and test window selection 
        train_windows_EEG_select = eeg_lib.windowSelection(train_windows_EEG, wind_names, train_windows)
        val_windows_EEG_select = eeg_lib.windowSelection(val_windows_EEG, wind_names, test_windows)
        test_windows_EEG_select = eeg_lib.windowSelection(test_windows_EEG, wind_names, test_windows)

        # set window labels 
        y_train = eeg_lib.setWindowLabels(train_windows_EEG_select, window_labels_train)
        y_val = eeg_lib.setWindowLabels(val_windows_EEG_select, window_labels_test)
        y_test = eeg_lib.setWindowLabels(test_windows_EEG_select, window_labels_test)


        # feature extraction from windows
        # time domain 
        x_train_t = eeg_lib.featureExtractionFromWindows(train_windows_EEG_select, feature_type1, f_samp_eeg, feature_times_windows)
        x_val_t = eeg_lib.featureExtractionFromWindows(val_windows_EEG_select, feature_type1, f_samp_eeg, feature_times_windows)
        x_test_t = eeg_lib.featureExtractionFromWindows(test_windows_EEG_select, feature_type1, f_samp_eeg, feature_times_windows)


        # ***************************************************************************
        # ************************Frequency preprocessing ***************************
        # ***************************************************************************
        # for now use this with different processing steps for frequency features 


        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled_f = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_f+"_train_"+str(iteration)+".npy")
        lrp_epochs_val_scaled_f = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_f+"_test_"+str(iteration)+".npy")
        lrp_epochs_test_scaled_f = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_f+"_val_"+str(iteration)+".npy")

        
        # windowing of the data 
        train_windows_EEG_f, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_train_scaled_f, f_samp_eeg, window_size, window_step)
        val_windows_EEG_f, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_val_scaled_f, f_samp_eeg, window_size, window_step)
        test_windows_EEG_f, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_test_scaled_f, f_samp_eeg, window_size, window_step)


        # train and test window selection 
        train_windows_EEG_select_f = eeg_lib.windowSelection(train_windows_EEG_f, wind_names, train_windows)
        val_windows_EEG_select_f = eeg_lib.windowSelection(val_windows_EEG_f, wind_names, test_windows)
        test_windows_EEG_select_f = eeg_lib.windowSelection(test_windows_EEG_f, wind_names, test_windows)


        #frequency domain 
        x_train_f = eeg_lib.featureExtractionFromWindows(train_windows_EEG_select_f, feature_type2, f_samp_eeg)
        x_val_f = eeg_lib.featureExtractionFromWindows(val_windows_EEG_select_f, feature_type2, f_samp_eeg)
        x_test_f = eeg_lib.featureExtractionFromWindows(test_windows_EEG_select_f, feature_type2, f_samp_eeg)



        # which train features should be used 
        if(features_used == "fusion"): 
            x_train = np.concatenate((x_train_t, x_train_f), axis = 1)
            x_val = np.concatenate((x_val_t, x_val_f), axis = 1)
            x_test = np.concatenate((x_test_t, x_test_f), axis = 1)

        elif(features_used == "frequency_domain"): 
            x_train = x_train_f
            x_test = x_test_f
            x_val = x_val_f

        else: 
            x_train = x_train_t
            x_val = x_val_t
            x_test = x_test_t


        print("Shape of train data: ", x_train.shape)
        print("Shape of test data: ", x_test.shape)
        print("Shape of validation data: ", x_val.shape)
        

        # *********************************************************************************
        # ********************* Build ML model in keras  ***********************************
        # *********************************************************************************


        # MLP setup
        model = MLP_Model(x_train)

        
        # # EEGNet setup 
        # Y_train = to_categorical(y_train, num_classes)
        # Y_validate = to_categorical(y_val, num_classes)
        # Y_test = to_categorical(y_test, num_classes)

        # x_train = x_train_f.reshape(x_train_f.shape[0], chans, samples, kernels)
        # x_val = x_val_f.reshape(x_val_f.shape[0], chans, samples, kernels)
        # X_test = x_test_f.reshape(x_test_f.shape[0], chans, samples, kernels)


        # # EEG Net 
        # model_EEGNet = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
        #                dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
        #                dropoutType='Dropout')



        # *********************************************************************************
        # ********************* Compile and train model  ***********************************
        # *********************************************************************************
        
        # early stopping callback
        #callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=early_stopping_patience, verbose=0, mode="auto", baseline=None, restore_best_weights=True)


        model.compile(loss=loss_fcn, optimizer=optimizer, metrics=metrics)
        history = model.fit(x_train,
                            y_train,
                            epochs  = n_epochs,
                            batch_size= n_batch_size,
                            shuffle = True,
                            workers=multiprocessing_cpus,
                            class_weight={0: weight_no_lrp_class, 1: weight_lrp_class},
                            use_multiprocessing=True,
                            validation_data = (x_val, y_val),
                            callbacks = [early_callback])
#
        #model.load_weights(results_path+'model_file.h5') # load best model weights 
        

        print("*********************************")
        print("Training basic model done")
        print("*********************************")


        # *********************************************************************************
        # ********************* Get and show training results   ***************************
        # *********************************************************************************

        # history of training process
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        num_of_epochs = range(1, len(loss_values)+1)

        if(show_training_results):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(6, 8))
            fig.subplots_adjust(hspace = 0.3)
            ax1.plot(num_of_epochs, loss_values, "bo", label="Training loss")
            ax1.plot(num_of_epochs, val_loss_values, "b", label="Validation loss")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.set_title("Loss values over trained epochs")

            acc_values = history_dict["accuracy"]
            val_acc_values = history_dict["val_accuracy"]

            ax2.plot(num_of_epochs, acc_values, "bo", label="Training accuracy")
            ax2.plot(num_of_epochs, val_acc_values, "b", label="Validation accuracy")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.set_title("Accuracy over trained epochs")

            plt.show()

        val_acc_values = history_dict["val_accuracy"]
        #max_val_idx = np.argmax(val_acc_values)
        #max_val_indices.append(max_val_idx)


        # *********************************************************************************
        # *********************Single trial predictions   *********************************
        # *********************************************************************************

        
        val_predictions = model.predict(x_val)
        val_pred_labels = np.array([0 if score <0.5 else 1 for score in val_predictions])
        tnr_val, tpr_val, acc_val, ba_val = eeg_lib.calcTestAccAndRates(val_pred_labels.flatten(), y_val.flatten())


        print("")
        print("Single trial metrics val data windows:")
        print("TNR: ",np.round(tnr_val, 3))
        print("TPR: ",np.round(tpr_val, 3))
        print("Acc: ", np.round(acc_val, 3))
        print("BA: ", np.round(ba_val, 3))
        print("")


        #perf_results = np.array([np.round(ba_train, 3), np.round(tpr_train, 3), np.round(tnr_train, 3)])
        perf_results = np.array([np.round(ba_val, 3), np.round(tpr_val, 3), np.round(tnr_val, 3)])
        perf_results_total.append(perf_results)


#save results
perf_results_total = np.array(perf_results_total)
np.savetxt(results_path+result_file_name, perf_results_total, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


#show model
print(model.summary())