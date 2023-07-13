
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from time import perf_counter
import sys

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib


# disable GPU for testing
tf.config.set_visible_devices([], 'GPU')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43", "RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "fcn_network_results_pooling_scale_ot_20t"
preprocessed_data_filename_end = "_34ch_05_4Hz_pool_tune_adapt_scale1"
use_fine_tuning = True

preprocessed_emg_file_name_end = "_emg" 

f_samp_eeg = 500 #sample Frequency of eeg

continues_selection = False # if True sampels for nolrp class are used from each part which has not been labeled to lrp
fuse_emg_eeg = False

#machine learning params
#n_samp_features = 100 # corresponds to 100 ms of data for both classes, for lrp last sampels to movement and for nolrp sampels from -5000 to -1000 are used with a stepsize
n_samp_lrp_label = 200 # label defs for window wise evaluation
# lrp definition and data starts -n_samp_features to 0
first_layer_units = 8 # neurons of first layer
second_layer_units = 8 # neurons of second layer
third_layer_units = 8 # neurons of third layer
n_epochs = 50 #30 training epochs (max since early stopping is used)
n_batch_size = 64 # batch size # 64 seems to work nice
dropout_rate = 0.1 #0.1 # dropout rate of every layer of the network
shuffle_data = True # shuffle all data for training, validation and testing
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
show_training_results = False
validation_rate = 0.2
multiprocessing_cpus = 16
early_stopping_patience = 15 # 3
leaky_alpha = 0.5


# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)
evaluation_time_per_window = 200 # 200 # the time in ms used at the end of each window for the prediction (in respect to 4 sampels at 20 Hz downsampling as features!)
with_channel_dim = False # metrics has no channel dim
use_relabelling = False # should the metrics be calculated with relabelling after training the classifier ? --> be careful when setting to True since youre changing the actual labels
# specify params for metric evaluation with method "relabelling", otherwise the parameters are not relevant if use_relabelling = False
determine_labels = 3
searching_bounds = [61, 81] # boundaries where the "label change point" is determined, values are the numbers of the windows (see wind_names param for which windows are selected as bounds)

# times in seconds where to get the train data from epoched signals
# t1_noLRP = -5000 # in ms
# t2_noLRP = -1000

# postprocessing params 
# short_tresh = 0.85
# mid_tresh = 0.7
# long_tresh = 0.6
# short_sampels = -75
# mid_sampels = -250
# long_sampels = -500
# num_class_instances = 1


# training windows 
pos_class_train_windows = [(-1100, -100), (-1000, 0)] # windows for training in ms 
neg_class_train_windows = [(-3050, -2050), (-3500, -2500)] # windows for training in ms 
feature_times_windows = (800, 1000) # time inside the windows to be used as features (last 200 ms)

#neg_class_train_windows = [(-1600, -600), (-1500, -500)] # windows for training in ms 


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


for subject in subject_names:
    for iteration in interations:
        
        # *********************************************************************************
        # ***************** Load train, test, val sets for every iteration ****************
        # *********************************************************************************

        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_train_"+str(iteration)+".npy")
        lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
        lrp_epochs_test_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_val_"+str(iteration)+".npy")

        print(lrp_epochs_train_scaled.shape)

        # subset of trials for training: 
        #lrp_epochs_train_scaled = lrp_epochs_train_scaled[:, :, :] # 20 trials 
        
        if(use_fine_tuning): 
            lrp_epochs_train_tune_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_trainTune_"+str(iteration)+".npy")
            lrp_epochs_train_tune_scaled = lrp_epochs_train_tune_scaled[0:20, :, :]

        # fuse eeg and emg data on data level 
        if (fuse_emg_eeg): 
            emg_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_emg_file_name_end+"_train_"+str(iteration)+".npy")
            emg_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_emg_file_name_end+"_test_"+str(iteration)+".npy")
            emg_epochs_test_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_emg_file_name_end+"_val_"+str(iteration)+".npy")

            epochs_train_scaled = np.concatenate((emg_epochs_train_scaled, lrp_epochs_train_scaled), axis = 1)
            epochs_val_scaled = np.concatenate((emg_epochs_val_scaled, lrp_epochs_val_scaled), axis = 1)
            epochs_test_scaled = np.concatenate((emg_epochs_test_scaled, lrp_epochs_test_scaled), axis = 1)

            #print("shape merged: ", epochs_train_scaled.shape)
        else: 
            
            epochs_train_scaled = lrp_epochs_train_scaled
            epochs_val_scaled = lrp_epochs_val_scaled
            epochs_test_scaled = lrp_epochs_test_scaled
            
            if(use_fine_tuning): 
                epochs_train_tune_scaled = lrp_epochs_train_tune_scaled


        # *********************************************************************************
        # ********************* Prepare data for network **********************************
        # *********************************************************************************
    
        # preparing training, validation and test data

        # split prepared data into train, validation and test


        x_train, y_train = eeg_lib.timeDomainFeaturesFromWindows(epochs_train_scaled, time_axis_eeg_batch, shuffle_data, pos_class_train_windows, neg_class_train_windows, feature_times_windows)
        x_val, y_val = eeg_lib.timeDomainFeaturesFromWindows(epochs_val_scaled, time_axis_eeg_batch, shuffle_data, pos_class_train_windows, neg_class_train_windows, feature_times_windows)
        x_test, y_test = eeg_lib.timeDomainFeaturesFromWindows(epochs_test_scaled, time_axis_eeg_batch, shuffle_data, pos_class_train_windows, neg_class_train_windows, feature_times_windows)

        if(use_fine_tuning): 
            x_train_tune, y_train_tune = eeg_lib.timeDomainFeaturesFromWindows(epochs_train_tune_scaled, time_axis_eeg_batch, shuffle_data, pos_class_train_windows, neg_class_train_windows, feature_times_windows)


        print("Shape of train data: ", x_train.shape)
        print("Shape of test data: ", x_test.shape)
        print("Shape of validation data: ", x_val.shape)

        if(use_fine_tuning): 
            print("Shape of train tune data: ", x_train_tune.shape)


        # *********************************************************************************
        # ********************* Build ML model in keras  ***********************************
        # *********************************************************************************

        # MLP setup
        model = Sequential()
        model.add(Dense(units=first_layer_units, input_shape=(x_train.shape[1],)))
        model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=second_layer_units))
        model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=third_layer_units))
        model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
        model.add(Dense(units=1, activation="sigmoid"))

        
        #show model
        #print(model.summary())

        # *********************************************************************************
        # ********************* Compile and train model  ***********************************
        # *********************************************************************************

        # early stopping callback
        #callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=early_stopping_patience, verbose=0, mode="auto", baseline=None, restore_best_weights=True)

        model.compile(loss="binary_crossentropy", optimizer="Nadam", metrics="accuracy")
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
        
        print("********")
        print("Training done")
        print("********")

        # train the trained model again with tuning data
        if(use_fine_tuning): 
            history_tune = model.fit(x_train_tune,
                                    y_train_tune,
                                    epochs  = n_epochs,
                                    batch_size= n_batch_size,
                                    shuffle = True,
                                    workers=multiprocessing_cpus,
                                    class_weight={0: weight_no_lrp_class, 1: weight_lrp_class},
                                    use_multiprocessing=True,
                                    validation_data = (x_val, y_val),
                                    callbacks = [early_callback])
            print("********")
            print("Tuning done")
            print("********")


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

        predicted_labels_train, true_labels_train, trial_prediction_train = eeg_lib.getKerasPredictionResultsLRP(model, epochs_train_scaled, n_samp_lrp_label)
        tnr_train, tpr_train, acc_train, ba_train = eeg_lib.calcTestAccAndRates(predicted_labels_train.flatten(), true_labels_train.flatten())


        predicted_labels_val, true_labels_val, trial_prediction_val = eeg_lib.getKerasPredictionResultsLRP(model, epochs_val_scaled, n_samp_lrp_label)
        tnr_val, tpr_val, acc_val, ba_val = eeg_lib.calcTestAccAndRates(predicted_labels_val.flatten(), true_labels_val.flatten())

        
        # predictions scores (each sample) of testset (shape: (trials, sampels))
        trial_prediction_val = trial_prediction_val[:,:,0] # just because of shape issues 


        # print("")
        # print("Single trial metrics val data (each sampel):")
        # print("TNR: ",np.round(tnr_val, 3))
        # print("TPR: ",np.round(tpr_val, 3))
        # #print("Acc: ", np.round(acc_test, 3))
        # print("BA: ", np.round(ba_val, 3))


        # *********************************************************************************
        # ********************* Window wise evaluation of performance   *******************
        # *********************************************************************************

        # comment for window eval 
        wind_arr_predictions, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(predicted_labels_val, f_samp_eeg, window_size, window_step) # currently only with channel dim False is supported!
        indices_relevant_winds = [wind_names.index("bis-2500"), wind_names.index("bis-2050"), wind_names.index("bis-100"), wind_names.index("bis0")]

        tnr_wind, tpr_wind, acc_wind, ba_wind, window_predictions, window_true_labels = eeg_lib.calcWindowMetrics(wind_arr_predictions, evaluation_time_per_window, window_step, f_samp_eeg, n_samp_lrp_label, use_relabelling, determine_labels, searching_bounds)

        # select only the specified windows for calculation of metrics 
        selected_win_predictions = window_predictions[:, indices_relevant_winds]
        selected_win_true_labels = window_true_labels[:, indices_relevant_winds]

        #print("selected_win_predictions:", selected_win_predictions)
        #print("selected_win_true_labels:", selected_win_true_labels)


        #calculate metricson selected windows  
        tnr_win_select, tpr_win_select, acc_win_select, ba_win_select = eeg_lib.calcTestAccAndRates(selected_win_predictions.flatten(), selected_win_true_labels.flatten())

        print("")
        print("Metrics on selected windows:")
        print("TNR: ",np.round(tnr_win_select, 3))
        print("TPR: ",np.round(tpr_win_select, 3))
        print("Acc: ", np.round(acc_win_select, 3))
        print("BA: ", np.round(ba_win_select, 3))


        # *********************************************************************************
        # ********************* Postprocessing of performance for online use   ************
        # *********************************************************************************

        

        perf_results = np.array([np.round(ba_win_select, 3), np.round(tpr_win_select, 3), np.round(tnr_win_select, 3)]) # window evaluation 
        
        perf_results_total.append(perf_results)


#save results
perf_results_total = np.array(perf_results_total)
np.savetxt(results_path+result_file_name, perf_results_total, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print((perf_counter()-time_start))