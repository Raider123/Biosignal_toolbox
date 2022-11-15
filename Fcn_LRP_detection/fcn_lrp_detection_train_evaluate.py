
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import sys 

# own libs 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder 
import eeg_lib

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated 
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used 
scenario_name = "intentional_unilateral"
result_file_name = "fcn_fixed_win_performance_325_samp(original)"

f_samp_eeg = 500 #sample Frequency of eeg

continues_prediction = True # if True sampels for nolrp class are used from each part which has not been labeled to lrp

#machine learning params 
n_samp_features = 325 # corresponds to 200 ms of data for both classes, for lrp last sampels to movement and for nolrp sampels from -5000 to -1000 are used with a stepsize 
# lrp definition and data starts -n_samp_features to 0 
input_dim = 34 # input dim of the network (feature dim)
first_layer_units = 4 # neurons of first layer 
second_layer_units = 4 # neurons of second layer 
third_layer_units = 4 # neurons of third layer 
n_epochs = 20 # training epochs 
n_batch_size = 64 # batch size 
dropout_rate = 0.1 # dropout rate of every layer of the network 
shuffle_data = True # shuffle all data for training, validation and testing 
weight_no_lrp_class = 0.6 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.4 
show_training_results = False
validation_rate = 0.2
multiprocessing_cpus = 16 

# window wise metric evaluation 
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)
evaluation_time_per_window = 200 # the time in ms used at the end of each window for the prediction (in respect to 4 sampels at 20 Hz downsampling as features!)
with_channel_dim = False # metrics has no channel dim 

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

# init performance results list 
perf_results_total = []

# load the time axis of the epoched data 
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

for subject in subject_names: 
    for iteration in interations: 

        # *********************************************************************************
        # ***************** Load train, test, val sets for every iteration ****************
        # *********************************************************************************

        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+"_train_"+str(iteration)+".npy")
        lrp_epochs_test_scaled = np.load(data_path+subject+"_"+scenario_name+"_test_"+str(iteration)+".npy")
        lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+"_val_"+str(iteration)+".npy")


        # *********************************************************************************
        # ********************* Prepare data for network **********************************
        # *********************************************************************************

        # preparing trainin, validation and test data 
        # times in seconds where to get the train data from epoched signals 
        t1 = 0 
        t2 = 2000
        # split prepared data into train, validation and test  
        x_train, y_train = eeg_lib.prepareEpochsForNetwork(lrp_epochs_train_scaled, time_axis_eeg_batch, n_samp_features, continues_prediction, shuffle_data, t1, t2)
        x_val, y_val = eeg_lib.prepareEpochsForNetwork(lrp_epochs_val_scaled, time_axis_eeg_batch, n_samp_features, continues_prediction, shuffle_data, t1, t2)
        x_test, y_test = eeg_lib.prepareEpochsForNetwork(lrp_epochs_test_scaled, time_axis_eeg_batch, n_samp_features, continues_prediction, shuffle_data, t1, t2)

        print("Shape of train data: ", x_train.shape)
        print("Shape of test data: ", x_test.shape)
        print("Shape of validation data: ", x_val.shape)

        # *********************************************************************************
        # ********************* Build ML model in keras  ***********************************
        # *********************************************************************************

        # MLP setup 
        model = Sequential()
        model.add(Dense(units=first_layer_units, activation="relu", input_shape=(x_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=second_layer_units, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=third_layer_units, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1, activation="sigmoid")) 

        #show model 
        #print(model.summary())

        # *********************************************************************************
        # ********************* Compile and train model  ***********************************
        # *********************************************************************************

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
        history = model.fit(x_train, 
                            y_train, 
                            epochs  = n_epochs, 
                            batch_size= n_batch_size, 
                            shuffle = True, 
                            workers=multiprocessing_cpus,
                            class_weight={0: weight_no_lrp_class, 1: weight_lrp_class},
                            use_multiprocessing=True,
                            validation_data = (x_val, y_val))

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

        # *********************************************************************************
        # *********************Single trial predictions   *********************************
        # *********************************************************************************

        predicted_labels_train, true_labels_train = eeg_lib.getPredictionResults(model, lrp_epochs_train_scaled, n_samp_features)
        tnr_train, tpr_train, acc_train, ba_train = eeg_lib.calcTestAccAndRates(predicted_labels_train.flatten(), true_labels_train.flatten())

        predicted_labels_test, true_labels_test = eeg_lib.getPredictionResults(model, lrp_epochs_test_scaled, n_samp_features)
        tnr_test, tpr_test, acc_test, ba_test = eeg_lib.calcTestAccAndRates(predicted_labels_test.flatten(), true_labels_test.flatten())

        print("")
        print("Single trial metrics train data (each sampel):")
        print("TNR: ",np.round(tnr_train, 3))
        print("TPR: ",np.round(tpr_train, 3)) 
        #print("Acc: ", np.round(acc_train, 3)) 
        print("BA: ", np.round(ba_train, 3)) 


        print("")
        print("Single trial metrics test data (each sampel):")
        print("TNR: ",np.round(tnr_test, 3))
        print("TPR: ",np.round(tpr_test, 3)) 
        #print("Acc: ", np.round(acc_test, 3)) 
        print("BA: ", np.round(ba_test, 3)) 


        # *********************************************************************************
        # ********************* Window wise evaluation of performance   *******************
        # *********************************************************************************

        wind_arr_metrics, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(predicted_labels_test, f_samp_eeg, window_size, window_step, with_channel_dim) # currently only with channel dim False is supported!
        tnr_wind_test, tpr_wind_test, acc_wind_test, ba_wind_test, window_predictions, window_true_labels = eeg_lib.calcWindowMetrics(wind_arr_metrics, evaluation_time_per_window, window_step, f_samp_eeg, n_samp_features)

        print("")
        print("Metrics single trial data (window evaluation, fixed label):")
        print("TNR: ",np.round(tnr_wind_test, 3))
        print("TPR: ",np.round(tpr_wind_test, 3)) 
        print("BA: ", np.round(ba_wind_test, 3)) 
        print("")

        perf_results = np.array([np.round(ba_wind_test, 3), np.round(tpr_wind_test, 3), np.round(tnr_wind_test, 3)])
        perf_results_total.append(perf_results)


perf_results_total = np.array(perf_results_total)
np.savetxt(results_path+result_file_name, perf_results_total, delimiter=",", fmt = "%1.8f")
print("all done")

