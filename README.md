# Biosignal_toolbox

## General remarks

### Project structure
This project contains all files for biosignal analysis (especially EMG and EEG) as well as machine learning flows in python. The src folder contains all the source files and includes project folders for a specific analysis or evaluation. You can find basic examples for biosignal processing and classification in the examples folder (under src). The lib folder contains the libraries containing methods for data processing for the biosignals. 
Single scripts or jupyter notebooks can be directly pushed to this repository. If you work on a project or evaluation containing more than one file, please create a new folder for the multiple files. If you created a machine learning flow, please consider structuring the evaluation in seperate files (e.g. loading, preprocessing, train_model, prediction). 
This repository should **ONLY** contain python scripts and no datasets, plots or other file formats. For the storage of data that should be analysed or evaluated, a data folder should be created containing all the files from experiments. The data folder will be ignored when adding and commiting the changes you made. If you are new to git, please have a look at the git documentation (https://www.git-scm.com/doc). 


### Installation 
To install the biosignal toolbox move to the **lib** folder run the following command to install the Python package: **pip install -e .**

### Coding Conventions
This section describes the general coding guidelines to follow during development process.
- **Programming Language** <br />
In this project, Python is used as the primary programming language.
- **Documentation** <br />
For the purpose of better readability and re-usability, it is essential that every script, attribute or method be well documented. In this project, the Google Style Python Docstrings should be used. (Refer : [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for more information). 
- **Defining a class** <br />
A new class in python requires a class name. Please use Pascal case (MyNewClass) for this purpose.
- **Defining a function or method** <br />
Please use Camel case (myFunc) for this purpose.
- **Declaring a variable or attribute** <br />
Please use Snake case (new_var) for this purpose.

## Files and folders (src) 


### EEGNet_LRP_detection 
This folder contains all files for the prediction of movement intentions by using the EEGNet architecture (CNN-Net approach). Neural network librarys like tensorflow and keras as well as the mne library for loading and processing of EEG-data are used. The machine learning flow is currently not up to date ! (TO BE UPDATED) 

### Neural_Network_Movement_Predictions_EEG
This folder contains neural networks for preprocessing and classification based on an MLP net (own development) and the EEGNet from lawhern et. al. 

### Fcn_LRP_detection (not up to date)
This folder contains the implementation of a fully connected neural network that is used for the classification of movement intentions based on the LRP and MRCPs. The model is implemented in keras and several methods for data processing and classification are integrated in the eeg library. 

### Multimodal_EEG_labelling
This folder contains scripts for the generation of EEG labels based on different modalities like motion tracking data (Qualisys), audio data or EMG signals. 

### EEG_denoising
The scripts in EEG denoising contain methods to reduce the noise and artifacts in the EEG data. Currently only scripts for testing autoencoders to reduce the noise are included. 

### Boxplots_results.py 
This Python file creates boxplots of classification results that are stored in a results folder. The boxplots are created with seaborn. 


