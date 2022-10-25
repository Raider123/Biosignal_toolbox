# MNE_machine_learning

## General remarks

### Project structure
This project contains all files for mne EEG- and EMG based analysis and machine learning flows in python. 
Single scripts can be directly pushed to this repository. If you work on a project or evaluation containing more than one file, please create a new folder for the multiple files. 
This repository should ONLY contain python scripts and no datasets, created plots or other file formats. For the storage of data that should be analysed or evaluated, a data folder should be created containing all the files from experiments. The data folder will be ignored when adding and commiting the changes you made. If you are new to git, please have a look at the git documentation (https://www.git-scm.com/doc). 

### Coding Conventions
This section describes the general coding guidelines to follow during development process.
- **Programming Language** <br />
In this project, Python is used as the primary programming Language.
- **Documentation** <br />
For the purpose of better readability and re-usability, it is essential that every script, attribute or method be well documented. In this project, the Google Style Python Docstrings should be used. (Refer : [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for more information). 
- **Defining a class** <br />
A new class in python requires a class name. Please use Pascal case (MyNewClass) for this purpose.
- **Defining a function or method** <br />
Please use Camel case (myFunc) for this purpose.
- **Declaring a variable or attribute** <br />
Please use Snake case (new_var) for this purpose.

## Files and folders

### MNE_EEG_P300_flow.ipynb
Standard P300 flow (jupyter notebook) implementation in mne in respect to the frequently used pySPACE flow.
Classify EEG data from single trials into standard or target with machine learning classifier. For the moment an SVM (SVM-C) is used.
You need a set of EEG data in Brain Products format. Also they need to have markers (stimuli for the P300) included in the *.vmrk file. Here 
The code will apply a bandpass filter to the data and extract all the events. After that the epochs are defined (three different ways to do that are implemented) and a downsampling will take place. The ML algorithm will classify the different configurations. Also there is a flow with xDawn filter applied in the pipeline and one without.
At the end there are some visualizations. At first there are all epochs visualized. Second the averaged epochs from one event over the whole set. At least there are the different accuracies visualized with matplotlib.pyplot.pcolormesh.

### EEGNet_LRP_detection 
This folder contains all files for the LRP prediction by using the EEGNet architecture (CNN-Net approach). Neural network librarys like tensorflow and keras as well as the mne library for loading and processing of EEG-data are used. For further information have a look at the README in this folder. 





