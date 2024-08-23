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
import warnings

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************


class MLModel:
    """
    This class contains methods to be used to run machine learning algorithms for the classification and evaluation of biosignals

    Parameters
    -------
    type : str, optional
        The type of the macine learning model (string) that should be used for the classification or regression task.
        Currently the types "keras" and "dtw" are implemented
    model : instance of keras, optional
        The model passed for further actions to be taken, if an instance of model can be created.
        Currently only a instance of keras model can be passed
    model_summary : bool, optional
        Flag if a summary of the model is printed (e.g. to show the size, parameters of a trained of created model)

    Author
    ------
    Author : Niklas Kueper \n
    Last changed: 16.11.2023 (by Niklas Kueper)
    """

    def __init__(self, type = "keras", model = None, model_summary = False): 

        """
        Constructor of the MLModel class. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 16.11.2023 (by Niklas Kueper)  
        """

        self.type = type 
        self.model = model 
        
        #self.use_input_norm = use_input_norm

        if(model_summary): 
            print(model.summary())


    def trainModel(self, show_train_results = False, save_trained_model = False, model_filename = "test",  train_epochs = 10, batch_size = 16, shuffle=True, class_weights = None, x_train = None, y_train = None, x_val = None, y_val = None, validation_split=0.2, callbacks = None, loss_fcn = None, optimizer = None, metrics = "accuracy"):
        """
        This method trains a machine learning model that was passed or created

        Parameters
        ----------
        show_train_results : bool, optional
            If True the results are plotted, by default False
        save_trained_model : bool, optional
            If True the trained model is saved in the data folder. The keras model is stored in .h5 format, by default False
        model_filename : str, optional
            The filename of the model to be saved, by default "test"
        train_epochs : int, optional
            The number of epochs for the training of, by default 10
        batch_size : int, optional
            The batch size used for training the ML-model, by default 16
        shuffle : bool, optional
            If True the data is shuffled befor training a model, by default True
        class_weights : list, optional
            The weights of the classes if a classification task is performed, by default None
        x_train : numpy array, optional
            The training data or features. The shape might depend on the input shape of each model, by default None
        y_train : numpy array, optional
            The class labels used for training, by default None
        x_val : numpy array, optional
            The validation data used in the training process. The shape might depend on the input shape of each model, by default None
        y_val : numpy array, optional
            The class labels used for the validation data, by default None
        validation_split : float, optional
            The ratio of validation data separated from the training
        callbacks : list, optional
            A callback function passed to be applied in the training procedure, by default None
        loss_fcn : str, optional
            The loss function used for training a ML-model, espsecially for neural networks, by default None
        optimizer : str, optional
            The optimizer used in the optimization process (i.e. weight updates for networks), by default None
        metrics : str, optional
            The metrics used for the performance evaluation, by default "accuracy"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 16.11.2023 (by Niklas Kueper
        """

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
        
        # compile model 
        if (self.type == "keras"): 
            self.model.compile(loss=self.loss_fcn, optimizer=self.optimizer, metrics=self.metrics)
            history = self.model.fit(self.x_train,
                                self.y_train,
                                epochs  = self.epochs,
                                batch_size= self.batch_size,
                                shuffle = self.shuffle,
                                class_weight=self.class_weights,
                                validation_split=validation_split,
                                # validation_data = (self.x_val, self.y_val),
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

                acc_values = history_dict[self.metrics]
                val_acc_values = history_dict["val_"+self.metrics]
                
                ax2.plot(num_of_epochs, acc_values, "bo", label="Training accuracy")
                ax2.plot(num_of_epochs, val_acc_values, "b", label="Validation accuracy")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel(self.metrics)
                ax2.legend()
                ax2.set_title(self.metrics+" over trained epochs")

                # plt.show()

            val_acc_values = history_dict["val_"+self.metrics]

        if(save_trained_model): 
            save_model(self.model, model_filename+".h5") # save 


    def loadModel(self, filename, path = ""): 
        """
        This function loads a ML-model stored in a specific folder

        Parameters
        ----------
        filename : str
            The filename of the model to be loaded
        path : str, optional
            The path where the model is stored, by default ""

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 16.11.2023 (by Niklas Kueper
        """

        self.model = load_model(filepath = path+filename+".h5")


    def calcTestAccAndRates(self, prediction_labels, true_labels):
        """
        This function gets the metrics from classification output of the test data.
        Currently the accuracy, balanced accuracy, tnr and tpr are calculated

        Parameters
        ----------
        prediction_labels : numpy array
            The predicted labels as 1D-numpy array (flatten the array if it has more dimensions)
        true_labels : numpy array
            The true labels as 1D-numpy array (flatten the array if it has more dimensions)

        Returns
        -------
        tuple
            tnr : float
                True negative rate
            tpr : float
                True positive rate
            acc : float
                Accuracy
            ba : float
                Balanced accuracy

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2023 (by Niklas Kueper
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
        """
        This method calculates the classification or regression perfomance of a trained model.
        Please note that this is only a helper function inside the class

        Parameters
        ----------
        predictions : numpy array
            The prediction scores (output) of a trained ML-model
        labels : numpy array
            The true labels used to calculate the performances
        encoding : str, optional
            The encoding of the class labels, can be "binary" for 0.0 and 1.0 as class labels, "onehotencoding" for onehotenconded labels or "distance_arry" for unsupervised methods like the dtw algorithm, by default "binary"
        type : str, optional
            The type of the classification or regression task
            Currently only "binaray" classification is implemented, by default "binary"
        show_results : bool, optional
            If True the results of the predictions are plotted. For eval_type = "online" the results are not shown to save computation time, by default False

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 16.11.2023 (by Niklas Kueper
        """
        
        if(type == "binary"):
            if(encoding == "onehotencoding"): 
                #print(predictions.shape)
                predictions = predictions[:, 1] # convert to binary from onehotencoding 
                labels = labels[:, 1]


            elif(encoding == "distance_array"): # for dtw algorithm 
                #print("predicitons shape", predictions.shape) # trials, channels
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

            # print("acc", acc)
            acc = 0.0 # not implemented 
            perf_results = np.array([np.round(acc, 3)])
            self.perf_results = perf_results

            if(show_results): 
                print("")
                print("Single trial metrics test data windows:")
                print("Acc: ", np.round(acc, 3))
                print("")

    
    def predictTarget(self, data, labels = None,  encoding = "binary", classification = True, n_classes = 2, show_results = True, show_pred_time = False, eval_type = "offline", templates = None, threshold = None):
        """
        This method is used to do predictions on new data using a trained model (if training is required)

        Parameters
        ----------
        data : numpy array
            The data on which the prediction should be done. The shape depends on the input shape of the model
        labels : numpy array, optional
            The true class labels (for a classification task) a 1D-numpy array. Not required for eval_type = "online" since no ground truth labels are available, by default None
        encoding : str, optional
            The encoding of the class labels (string), can be "binary" for 0.0 and 1.0 as class labels, "onehotencoding" for onehotencoded labels or "distance_array" for unsupervised methods like the the dtw algorithm.
        n_classes : int, optional
            The number of classes for which the predictions are made, by default 2
        show_results : bool, optional
            If True the results of the predictions are plotted. For eval_type = "online" the results are not shown to save computation time, by default True
        show_pred_time : bool, optional
            A flag weather to show how long the time was to perform the prediction on the data provided, by default False
        eval_type : str, optional
            The type of the evaluation. Can be either "offline" or "online". For "offline" results are shown by default and prediction times can be measured. For "online" only the prediction scores are calculated and not further evaluated into a performance, by default "offline"
        templates : numpy array, optional
            If using the "dtw" or another matching algorithm (unsupervised), the templates with shape (n_channels, n_sampels) (e.g. for the dtw algorithm), by default None
        threshold : int, optional
            Treshold, by default None

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 09.03.2024 (by Niklas Kueper) 
        """

        if(self.type == "keras"): 

            if(eval_type == "offline"): 
                if (show_pred_time): 
                    time1 = perf_counter_ns()
                
                print("Test data input shape: ", data.shape)
                predictions = self.model(data) # call the model, is a lot faster than using predict method 
                
                if(show_pred_time): 
                    time2 = perf_counter_ns()
                    print("pred time ms", (time2-time1)/1000000)

                if(classification): 
                    if (n_classes == 2): 
                        self.calcPerformance(predictions = predictions, encoding = encoding, show_results = show_results, labels=labels, type="binary")
                    elif (n_classes > 2): 
                        self.calcPerformance(predictions = predictions, encoding = encoding, show_results = show_results, labels=labels, type="multiclass")
                else: 
                    self.prediction_scores = predictions # for regression this is the result 

            
            elif(eval_type == "online"):
                
                if(classification): 
                    self.prediction_scores =  np.array(self.model(data)).flatten()
                else: 
                    self.prediction_scores = predictions


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
                            dtw_obj_pos_class = dtw(data[train_idx, channel_idx, :], y=templates[1][channel_idx, :], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20}) 
                            distances_pos_class.append(dtw_obj_pos_class.distance)
                            #neg class 
                            dtw_obj_neg_class = dtw(data[train_idx, channel_idx, :], y=templates[0][channel_idx, :], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20}) 
                            distances_neg_class.append(dtw_obj_neg_class.distance)


                        if(threshold): 
                            predictions.append(np.mean(np.array(distances_pos_class))) # predictions are only distances for positive class 
                        else: 
                            predictions.append([np.mean(np.array(distances_neg_class)), np.mean(np.array(distances_pos_class))]) # predictions are mean distances for both classes 

                    predictions = np.array(predictions)
                    print("len pos class", len(distances_pos_class))

                    if(threshold): 
                        print("Distances:", predictions)
                        predictions = (predictions < threshold).astype(float)
                        self.calcPerformance(predictions = predictions, encoding = "binary", show_results = show_results, labels=labels, type="binary")
                    else: 
                        self.calcPerformance(predictions = predictions, encoding = "distance_array", show_results = show_results, labels=labels, type="binary")


                else: 
                    warnings.warn("not implemented yet .. ")

            else: 
                warnings.warn("not implemented yet .. ")
            

    def getPerfResults(self):
        """
        This method returns the performance results after a prediction was performed

        Returns
        -------
        numpy array
            perf_results : 1D-numpy array of floats
            The performance results depending on the type of classification or regression task. For binary classification the array contains BA, TPR, TNR and ACC. For multiclass, only the ACC is currently returned 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.11.2023 (by Niklas Kueper
        """

        return self.perf_results
    
    def modelSummary(self): 
        """
        This method prints a summary of a created ML-model

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.11.2023 (by Niklas Kueper
        """

        self.model.summary()

    def getPredictionScores(self):
        
        """
        This method returns the unprocessed prediction scores. This is usually a value betrween 0 and 1 from e.g. a sigmoidal fit

        Returns
        -------
        numpy array
            Prediction_scores : float
                The prediction scores as a 1D-numpy array for every predicted instance (n_predictions)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.11.2023 (by Niklas Kueper
        """
        
        return np.array(self.prediction_scores) 

    def printKerasModelLayerNames(self): 

        layer_names = [layer.name for layer in self.model.layers]

        # Print the layer names
        print("Layer names:")
        for name in layer_names:
            print(name)

    def printKerasModelLayerWeights(self, layer_name): 

        layer = self.model.get_layer(name=layer_name)

        # Print the weights
        weights = layer.get_weights()
        print("Weights for layer '{}':".format(layer.name))
        for w in weights:
            print(w)
        
