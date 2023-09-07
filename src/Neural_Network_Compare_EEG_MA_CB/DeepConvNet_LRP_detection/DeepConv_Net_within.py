import sys
import eeg_lib
from datetime import datetime
import tensorflow as tf
import csv
import seaborn as sns
import os
import numpy as np
import pandas as pd
from itertools import combinations as comb
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from mne import io
import mne
from sklearn.metrics import classification_report
import time as tm
# EEGModel-specific imports
from EEGModels import DeepConvNet

start_time = tm.time()
# Check if GPU is available
tf.config.list_physical_devices('GPU')


# Preprocessing
def preprocess(raw_data):
    raw_data.resample(120)
    raw_data.filter(0, 40, method='iir')
    raw_data.apply_function(lambda x: ((x - np.mean(x)) / np.std(x)))

    events, event_id = mne.events_from_annotations(raw_data)

    events_filtered = events[events[:, 2] == event_id['Stimulus/S100']]
    events_filtered[:, 2] = 1
    events_rest = events[events[:, 2] != event_id['Stimulus/S100']]
    events_rest[:, 2] = 0

    events_combined = np.concatenate((events_filtered, events_rest), axis=0)
    events_combined = events_combined[np.argsort(events_combined[:, 0])]

    events = events_combined
    event_id = {1: 'Stimulus/S100', 0: 'Other events'}

    exclude_channels = ["x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
    event_id = {str(i): i for i in event_id}
    picks = mne.pick_types(raw_data.info, exclude=exclude_channels, meg=False, eeg=True, eog=False)
    epochs = mne.Epochs(raw_data, events, 1, tmin=-5, tmax=0, picks=picks, event_repeated='drop', proj=False, baseline=None, preload=True, verbose=False)

    X = epochs.get_data()

    windows_to_include = [30, 39, 78, 80]
    wind_arr, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(X, 120, windows_to_include, window_size=1000, window_step=50, with_channel_dim=True)

    wind_arr = wind_arr.transpose((0, 3, 1, 2)).reshape((-1, wind_arr.shape[1], wind_arr.shape[2]))

    return wind_arr

def shuffle(data, labels):
    indices = tf.range(start=0, limit=len(data), dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    return tf.gather(data, shuffled_indices).numpy(), tf.gather(labels, shuffled_indices).numpy()

#EEGNet-parameter
dropoutRate = 0.5
kernLength = 32
F1 = 8
D = 2
F2 = 16


proj_path = 'C:/Users/Chrissy/PycharmProjects/Masterarbeit/'

sys.path.append(proj_path+"/venv/Lib/site-packages/") # path to lib folder


data_path = proj_path+"/Multimodal_Data/"

current_date = datetime.now()
date = current_date.strftime('%d%m%Y')
time = datetime.now().strftime("%H%M")


# Path of the datasets that are available
data_str_uni_JV43 = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set2.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set3.vhdr"])
data_str_uni_RA12 = np.array([data_path+"20211216_r_RA12_intentional_unilateral_set1.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set2.vhdr", data_path+"20211216_r_RA12_intentional_unilateral_set3.vhdr"])
data_str_uni_AV82 = np.array([data_path+"20211220_r_AV82_intentional_unilateral_set1.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set2.vhdr", data_path+"20211220_r_AV82_intentional_unilateral_set3.vhdr"])
data_str_uni_UP28 = np.array([data_path+"20211223_r_UP28_intentional_unilateral_set1.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set2.vhdr", data_path+"20211223_r_UP28_intentional_unilateral_set3.vhdr"])
data_str_uni_XP01 = np.array([data_path+"20211222_r_XP01_intentional_unilateral_set1.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set2.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set3.vhdr", data_path+"20211222_r_XP01_intentional_unilateral_set4.vhdr"])
data_str_uni_ZS27 = np.array([data_path+"20220104_r_ZS27_intentional_unilateral_set1.vhdr", data_path+"20220104_r_ZS27_intentional_unilateral_set2.vhdr", data_path+"20220104_r_ZS27_intentional_unilateral_set3.vhdr"])
data_str_uni_JD68 = np.array([data_path+"20220105_r_JD68_intentional_unilateral_set1.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set2.vhdr", data_path+"20220105_r_JD68_intentional_unilateral_set3.vhdr"])
data_str_uni_QS70 = np.array([data_path+"20220107_r_QS70_intentional_unilateral_set1.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set2.vhdr", data_path+"20220107_r_QS70_intentional_unilateral_set3.vhdr"])


datasets = {
    "JV43": data_str_uni_JV43,
    "RA12": data_str_uni_RA12,
    "AV82": data_str_uni_AV82,
    "UP28": data_str_uni_UP28,
    "XP01": data_str_uni_XP01,
    "ZS27": data_str_uni_ZS27,
    "JD68": data_str_uni_JD68,
    "QS70": data_str_uni_QS70,
}

results_df = pd.DataFrame(columns=['Dataset', 'Test Accuracy', 'Train Accuracy', 'Test Balanced Accuracy', 'Train Balanced Accuracy', 'Test TPR', 'Train TPR', 'Test TNR', 'Train TNR'])

# Results folder path
base_folder_path = "C:/Users/Chrissy/Desktop/Results/" + str(dropoutRate) + "," + str(kernLength) + "," + str(F1) + "," + str(D) + "," + str(F2) + "_" + date + "_" + time + "_DeepConvNet" + "/"

# Create folder if it doesn't exist
if not os.path.exists(base_folder_path):
    os.makedirs(base_folder_path)

avg_test_accuracy = {}
avg_train_accuracy = {}
avg_test_bal_accuracy = {}
avg_train_bal_accuracy = {}
avg_min_val_loss = {}
avg_train_loss = {}

for dataset_key in datasets:
    dataset_list = datasets[dataset_key]

    # Create a folder for each dataset
    dataset_name = dataset_list[0].split("/")[-1].split("_")[2]
    dataset_folder_path = os.path.join(base_folder_path, dataset_name)
    if not os.path.exists(dataset_folder_path):
        os.makedirs(dataset_folder_path)

    # Initialize lists to store results for the current dataset
    combination_results = []
    confusion_matrices = []
    average_bal_acc_list = []

    # Generate combinations of datasets for training and testing
    if dataset_key == "XP01":
        combinations_list = list(comb(range(len(dataset_list) - 1), 2))
    else:  # generating all 3 combinations for all other datasets
        combinations_list = [tuple(range(i)) + tuple(range(i+1, 3)) for i in range(3)]

    for idx, combination in enumerate(combinations_list):
        # create a unique directory for each dataset and combination
        combination_name = f'{dataset_name}_combination_{idx + 1}'
        combination_folder_path = os.path.join(dataset_folder_path, combination_name)
        if not os.path.exists(combination_folder_path):
            os.makedirs(combination_folder_path)

        # Load and preprocess the datasets for training and validation
        raw_data_train_val = None
        for i in combination:
            raw = None
            if dataset_key == 'XP01' and i == 2:  # if XP01 and set3 is in the combination
                raw1 = io.read_raw_brainvision(data_path + "20211222_r_XP01_intentional_unilateral_set3.vhdr",
                                               preload=True, verbose=False)
                raw2 = io.read_raw_brainvision(
                    data_path + "20211222_r_XP01_intentional_unilateral_set4.vhdr", preload=True, verbose=False)
                raw = mne.concatenate_raws([raw1, raw2])  # concatenate set3 and set4
            else:
                raw = io.read_raw_brainvision(dataset_list[i], preload=True, verbose=False)

            if raw_data_train_val is None:
                raw_data_train_val = raw
            else:
                raw_data_train_val = mne.concatenate_raws([raw_data_train_val, raw])

        # Preprocess the training and validation data
        wind_arr_train_val = preprocess(raw_data_train_val)

        # Load and preprocess the raw data for testing
        test_dataset_index = [i for i in range(3) if i not in combination][0]
        test_dataset = dataset_list[test_dataset_index]

        if dataset_key == 'XP01' and test_dataset == data_path + "20211222_r_XP01_intentional_unilateral_set3.vhdr":
            raw_data_test = io.read_raw_brainvision(data_path + "20211222_r_XP01_intentional_unilateral_set3.vhdr",
                                                    preload=True, verbose=False)
            raw_data_test2 = io.read_raw_brainvision(data_path + "20211222_r_XP01_intentional_unilateral_set4.vhdr",
                                                     preload=True, verbose=False)
            raw_data_test = mne.concatenate_raws([raw_data_test, raw_data_test2])  # concatenate set3 and set4
        else:
            raw_data_test = io.read_raw_brainvision(test_dataset, preload=True, verbose=False)

        wind_arr_test = preprocess(raw_data_test)


        # Get the number of trials
        n_trials_train_val = wind_arr_train_val.shape[0]
        n_trials_test = wind_arr_test.shape[0]

        # Create labels for all trials before splitting
        all_labels = np.tile([0, 0, 1, 1], n_trials_train_val + n_trials_test)

        # Split the train and validation sets
        n_train = int(n_trials_train_val * 0.8)  # 80% of data for training
        n_val = n_trials_train_val - n_train  # 20% of data for validation

        # Assign labels to each set
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:n_train + n_val]
        test_labels = all_labels[n_train + n_val:]

        train_arr = wind_arr_train_val[:n_train]
        val_arr = wind_arr_train_val[n_train:n_train + n_val]
        test_arr = wind_arr_test

        # Shuffle the data
        X_train, Y_train = shuffle(train_arr, train_labels)
        X_validate, Y_validate = shuffle(val_arr, val_labels)
        X_test, Y_test = shuffle(test_arr, test_labels)


        kernels, chans, samples = 1, 34, 120

        # convert labels to one-hot encodings.

        num_classes = 2
        Y_train = to_categorical(Y_train, num_classes)
        Y_validate = to_categorical(Y_validate, num_classes)
        Y_test = to_categorical(Y_test, num_classes)

        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        # print('X_train:', X_train)
        print('Y_train shape:', Y_train.shape)
        print(Y_train[0, :])
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')


        model = DeepConvNet(2, Chans=chans, Samples=samples, dropoutRate=0.5)


        # compile the model
        model.compile(loss='BinaryCrossentropy', optimizer='adam',
                      metrics=['accuracy'])


        numParams = model.count_params()


        # set path to record model checkpoints
        checkpointer = ModelCheckpoint(
            filepath="/tmp/checkpoint/" + str(dropoutRate) + "," + str(kernLength) + "," + str(F1) + "," + str(
                D) + "," + str(F2) + "/" + dataset_name + "_DCN" + '.h5', verbose=1,
            save_best_only=True)



        class_weights = {0: 1, 1: 1}

        print('xtrain:', X_train.shape)

        print('ytrain:', Y_train.shape)


        fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300,
                                verbose=2, validation_data=(X_validate, Y_validate),
                                callbacks=[checkpointer], class_weight=class_weights)

        # Load optimal weights
        model.load_weights("/tmp/checkpoint/" + str(dropoutRate) + "," + str(kernLength) + "," + str(F1) + "," + str(D) + "," + str(F2) + "/" + dataset_name + "_DCN" + '.h5')

        # Evaluate model on test set
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
        print('Test accuracy:', test_acc)
        # Min Train Loss
        train_loss = np.min(fittedModel.history['loss'])
        print('Train Loss', train_loss)
        # Min Val Loss
        min_val_loss = fittedModel.history['val_loss'][-1]
        print('Validation Loss', min_val_loss)

        # Plot training and validation accuracy over epochs
        plt.figure()
        plt.plot(fittedModel.history['accuracy'])
        plt.plot(fittedModel.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(combination_folder_path, "accuracy.png"))

        # Plot training and validation loss over epochs
        plt.figure()
        plt.plot(fittedModel.history['loss'])
        plt.plot(fittedModel.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(combination_folder_path, "loss.png"))

        # Make prediction on train and test set
        train_probs = model.predict(X_train)
        train_preds = train_probs.argmax(axis=-1)

        test_probs = model.predict(X_test)
        test_preds = test_probs.argmax(axis=-1)

        # Calculate and print accuracy for both sets
        train_acc = np.mean(train_preds == Y_train.argmax(axis=-1))
        print("Training accuracy: %f " % (train_acc))

        test_acc = np.mean(test_preds == Y_test.argmax(axis=-1))
        print("Test accuracy: %f " % (test_acc))



        # Display balanced accuracy, true positive rate, and true negative rate for training set
        train_bal_acc = balanced_accuracy_score(Y_train.argmax(axis=-1), train_preds)
        train_tpr = recall_score(Y_train.argmax(axis=-1), train_preds, pos_label=1)
        train_tnr = recall_score(Y_train.argmax(axis=-1), train_preds, pos_label=0)
        print("Training balanced accuracy: %f " % (train_bal_acc))
        print("Training true positive rate: %f " % (train_tpr))
        print("Training true negative rate: %f " % (train_tnr))

        # Display balanced accuracy, true positive rate, and true negative rate for test set
        test_bal_acc = balanced_accuracy_score(Y_test.argmax(axis=-1), test_preds)
        test_tpr = recall_score(Y_test.argmax(axis=-1), test_preds, pos_label=1)
        test_tnr = recall_score(Y_test.argmax(axis=-1), test_preds, pos_label=0)
        print("Test balanced accuracy: %f " % (test_bal_acc))
        print("Test true positive rate: %f " % (test_tpr))
        print("Test true negative rate: %f " % (test_tnr))

        if dataset_name not in avg_test_accuracy:
            avg_test_accuracy[dataset_name] = []
            avg_train_accuracy[dataset_name] = []
            avg_test_bal_accuracy[dataset_name] = []
            avg_train_bal_accuracy[dataset_name] = []
            avg_min_val_loss[dataset_name] = []
            avg_train_loss[dataset_name] = []
        avg_test_accuracy[dataset_name].append(test_acc)
        avg_train_accuracy[dataset_name].append(train_acc)
        avg_test_bal_accuracy[dataset_name].append(test_bal_acc)
        avg_train_bal_accuracy[dataset_name].append(train_bal_acc)
        avg_min_val_loss[dataset_name].append(min_val_loss)
        avg_train_loss[dataset_name].append(train_loss)

        # Compute confusion matrices
        train_cm = confusion_matrix(Y_train.argmax(axis=-1), train_preds)
        test_cm = confusion_matrix(Y_test.argmax(axis=-1), test_preds)

        # Print confusion matrices
        print("Training confusion matrix:")
        print(train_cm)
        print("Test confusion matrix:")
        print(test_cm)

        # Calculate percentage confusion matrices
        train_cm_percentage = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
        test_cm_percentage = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

        # Print confusion matrices
        print("Training confusion matrix:")
        print(train_cm_percentage)
        print("Test confusion matrix:")
        print(test_cm_percentage)

        # Save confusion matrices as images
        plt.figure()
        sns.heatmap(train_cm_percentage, annot=True, cmap='Blues')
        plt.title("Training Confusion Matrix (Percentage)")
        plt.savefig(os.path.join(combination_folder_path, "train_confusion_matrix.png"))

        plt.figure()
        sns.heatmap(test_cm_percentage, annot=True, cmap='Blues')
        plt.title("Test Confusion Matrix (Percentage)")
        plt.savefig(os.path.join(combination_folder_path, "test_confusion_matrix.png"))

        # Calculate and print average balanced accuracy of combinations
        # average_bal_acc = test_bal_acc
        print("Average Balanced Accuracy: %f" % test_bal_acc)

        # Save average balanced accuracy as a boxplot in the dataset folder
        average_bal_acc_list.append(test_bal_acc)
        plt.figure()
        plt.boxplot(average_bal_acc_list)
        plt.title("Average Balanced Accuracy")
        plt.ylabel("Balanced Accuracy")
        plt.savefig(os.path.join(dataset_folder_path, "average_balanced_accuracy_boxplot.png"))

        #create report
        report = classification_report(Y_test.argmax(axis=-1), test_preds)

        # Write the report to a text file
        with open(os.path.join(combination_folder_path, "classification_report.txt"), "w") as text_file:
            text_file.write(report)

        # Save parameters and performance to a csv file
        parameters_performance = [dropoutRate, kernLength, F1, D, F2, train_acc, train_bal_acc, train_tpr, train_tnr, test_acc, test_bal_acc, test_tpr, test_tnr]
        with open(os.path.join(combination_folder_path, "parameters_performance.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Dropout Rate", "Kernel Length", "F1", "D", "F2", "Train Accuracy", "Train Balanced Accuracy", "Train TPR", "Train TNR", "Test Accuracy", "Test Balanced Accuracy", "Test TPR", "Test TNR"])
            writer.writerow(parameters_performance)

        # Save model architecture and weights
        model_json = model.to_json()
        with open(os.path.join(combination_folder_path, "model.json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(combination_folder_path, "model.h5"))


# Calculate average accuracies for each dataset
for dataset_name in avg_test_accuracy:
    avg_test_accuracy[dataset_name] = np.mean(avg_test_accuracy[dataset_name])
    avg_train_accuracy[dataset_name] = np.mean(avg_train_accuracy[dataset_name])
    avg_test_bal_accuracy[dataset_name] = np.mean(avg_test_bal_accuracy[dataset_name])
    avg_train_bal_accuracy[dataset_name] = np.mean(avg_train_bal_accuracy[dataset_name])
    avg_min_val_loss[dataset_name] = np.mean(avg_min_val_loss[dataset_name])
    avg_train_loss[dataset_name] = np.mean(avg_train_loss[dataset_name])

# boxplot for average of all datasets
plt.figure()
plt.boxplot(list(avg_test_bal_accuracy.values()))
plt.title("Overall Average Balanced Accuracy")
plt.ylabel("Balanced Accuracy")
plt.savefig(os.path.join(base_folder_path, "overall_average_balanced_accuracy_boxplot.png"))

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Dataset': list(avg_test_accuracy.keys()),
    'Train Accuracy': list(avg_train_accuracy.values()),
    'Test Accuracy': list(avg_test_accuracy.values()),
    'Train Balanced Accuracy': list(avg_train_bal_accuracy.values()),
    'Test Balanced Accuracy': list(avg_test_bal_accuracy.values()),
    'Validation Loss': list(avg_min_val_loss.values()),
    'Train Loss': list(avg_train_loss.values()),
})
print("avg_test:", list(avg_test_bal_accuracy.values()))
# Save the results to a CSV file
results_df.to_csv(os.path.join(base_folder_path, "results_DCN.csv"), index=False, sep=';')

end_time = tm.time()

execution_time = end_time - start_time
print(f"The script executed in {execution_time} seconds")

# Print the results
print(results_df)
