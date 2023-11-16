# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from time import perf_counter_ns
from dtw import *
import copy 

import tensorflow as tf

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************


class MLModel: 

    def __init__(self, type = "keras", model = None, train_epochs = 10, batch_size = 16, shuffle=True, class_weights = None, x_train = None, y_train = None, x_val = None, y_val = None, callbacks = None, model_summary = False, loss_fcn = None, optimizer = None, metrics = "accuracy"): 

        self.type = type 
        self.model = model 
        self.epochs = train_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.class_weights = class_weights
        self.x_train = x_train
        self.x_val = x_val
        self.y_val = y_val
        self.y_train = y_train
        self.callbacks = callbacks
        self.loss_fcn = loss_fcn
        self.optimizer = optimizer
        self.metrics = metrics 
        self.perf_results = None
        #self.use_input_norm = use_input_norm

        if(model_summary): 
            print(model.summary())


    def trainModel(self, show_train_results = False, save_trained_model = False, model_filename = "test"):         


        # compile model 
        if (self.type == "keras"): 
            self.model.compile(loss=self.loss_fcn, optimizer=self.optimizer, metrics=self.metrics)
            history = self.model.fit(self.x_train,
                                self.y_train,
                                epochs  = self.epochs,
                                batch_size= self.batch_size,
                                shuffle = self.shuffle,
                                class_weight=self.class_weights,
                                validation_data = (self.x_val, self.y_val),
                                callbacks = self.callbacks)
        
            
            # history of training process
            history_dict = history.history
            loss_values = history_dict["loss"]
            val_loss_values = history_dict["val_loss"]
            num_of_epochs = range(1, len(loss_values)+1)

            if(show_train_results):
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

        if(save_trained_model): 
            save_model(self.model, model_filename+".h5") # save 


    def loadModel(self, filename, path = ""): 
        self.model = load_model(filepath = path+filename+".h5")


    def calcTestAccAndRates(self, prediction_labels, true_labels):

        """
        Get metrics from classification output of the test data. Currently the accuracy, balanced accuracy,  tnr and tpr are calculated. 

        Arguments:
            prediction_labels: The predicted labels as one dimensional numpy array (flatten the array if it has more dimensions).
            true_labels: The true labels as one dimensional numpy array (flatten the array if it has more dimensions). 

        Returns:
            tnr: True negative rate 
            tpr: True positive rate
            acc: Accuracy
            ba: Balanced accuracy
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
        """

        prediction_labels = prediction_labels.astype(int)
        true_labels = true_labels.astype(int)
        
        negatives = np.sum(np.abs(true_labels-1))
        positives = np.sum(true_labels)
        
        n = len(true_labels)
        tps= 0 
        tns = 0
        for index in range(0, n): 
            if (prediction_labels[index] == 0 and true_labels[index] == 0):
                tns = tns+1
            elif (prediction_labels[index] == 1 and true_labels[index] == 1): 
                tps = tps+1
        tnr = tns/negatives
        tpr = tps/positives
        acc = ((tns+tps)/n) #in percent
        ba = (tnr+tpr)/2

        return tnr, tpr, acc, ba
    
    def calcPerformance(self, predictions, labels, encoding = "binary", type="binary", show_results = False): 
        
        if(type == "binary"):
            if(encoding == "onehotencoding"): 
                print(predictions.shape)
                predictions = predictions[:, 1] # convert to binary from onehotencoding 
                labels = labels[:, 1]


            elif(encoding == "distance_array"): # for dtw algorithm 
                print("predicitons shape", predictions.shape) # trials, channels
                predictions_binary = []
                for index in range(0, predictions.shape[0]): 
                    if(predictions[index, 0] < predictions[index, 1]): # first index is negative class second positive
                        predictions_binary.append(0.0)
                    else: 
                        predictions_binary.append(1.0)

                predictions = np.array(predictions_binary)

            pred_labels = np.array([0 if score <0.5 else 1 for score in predictions])
            self.prediction_scores = np.array(predictions).flatten() # store predictions 
            self.predicted_labels = pred_labels

            tnr, tpr, acc, ba = self.calcTestAccAndRates(pred_labels.flatten(), labels.flatten())
            perf_results = np.array([np.round(ba, 3), np.round(tpr, 3), np.round(tnr, 3), np.round(acc, 3)])
            self.perf_results = perf_results # store perf results 

            if(show_results): 
                print("")
                print("Single trial metrics test data windows:")
                print("TNR: ",np.round(tnr, 3))
                print("TPR: ",np.round(tpr, 3))
                print("Acc: ", np.round(acc, 3))
                print("BA: ", np.round(ba, 3))
                print("")
                
        else: 
            self.prediction_scores = np.array(predictions) # not flatten because n classes
            self.predicted_labels = np.argmax(self.prediction_scores, axis=1)

            # acc_metrics = tf.keras.metrics.Accuracy()
            # acc_metrics.update_state(labels, self.prediction_scores)
            # acc = acc_metrics.result().numpy()
            # print("acc", acc)
            acc = 0.0 # not implemented 
            perf_results = np.array([np.round(acc, 3)])
            self.perf_results = perf_results

            if(show_results): 
                print("")
                print("Single trial metrics test data windows:")
                print("Acc: ", np.round(acc, 3))
                print("")

    
    def predict(self, data, labels,  encoding = "binary", n_classes = 2, show_results = False, show_pred_time = False, eval_type = "offline", templates = None): 

        if(self.type == "keras"): 

            if(eval_type == "offline"): 
                if (show_pred_time): 
                    time1 = perf_counter_ns()
                
                print("data shape input ", data.shape)
                predictions = self.model(data) # call the model, is a lot faster than using predict method 
                print("predictions", predictions)
                #predictions = self.model.predict_on_batch(data)

                if(show_pred_time): 
                    time2 = perf_counter_ns()
                    print("pred time ms", (time2-time1)/1000000)

                if (n_classes == 2): 
                    self.calcPerformance(predictions = predictions, encoding = encoding, show_results = show_results, labels=labels, type="binary")
                elif (n_classes > 2): 
                    self.calcPerformance(predictions = predictions, encoding = encoding, show_results = show_results, labels=labels, type="multiclass")

            elif(eval_type == "online"):
                
                self.prediction_scores =  np.array(self.model(data)).flatten()

        elif(self.type == "dtw"): 
            #templates have shape n_train, channel, sampels 

            predictions = []
            if(eval_type == "offline"): 
                if(n_classes == 2): 
                    
                    print("xtrain shape", data.shape)
                    for train_idx in range(0, data.shape[0]):
                        distances_pos_class = []
                        distances_neg_class = []
                        for channel_idx in range(0, data.shape[1]): 
                            
                            # pos class distances 
                            dtw_obj_pos_class = dtw(data[train_idx, channel_idx, :], y=templates[1][channel_idx, :], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 50}) 
                            distances_pos_class.append(dtw_obj_pos_class.distance)
                            #neg class 
                            dtw_obj_neg_class = dtw(data[train_idx, channel_idx, :], y=templates[0][channel_idx, :], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 50}) 
                            distances_neg_class.append(dtw_obj_neg_class.distance)

                        predictions.append([np.mean(np.array(distances_neg_class)), np.mean(np.array(distances_pos_class))]) # predictions are mean distances for both classes 

                    predictions = np.array(predictions)
                    self.calcPerformance(predictions = predictions, encoding = "distance_array", show_results = show_results, labels=labels, type="binary")


                else: 
                    print("not implemented ... ")

            else: 
                print("not implemented")
            

    def getPerfResults(self): 
        return self.perf_results
    
    def modelSummary(self): 
        self.model.summary()

    def getPredictionScores(self): 
        return self.prediction_scores

