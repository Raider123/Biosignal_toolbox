# MNE_machine_learning

## MNE_EEG_P300_flow.ipynb
Standard P300 flow (jupyter notebook) implementation in mne in respect to the frequently used pySPACE flow.
Classify EEG data from single trials into standard or target with machine learning classifier. For the moment an SVM (SVM-C) is used.
You need a set of EEG data in Brain Products format. Also they need to have markers (stimuli for the P300) included in the *.vmrk file. Here 
The code will apply a bandpass filter to the data and extract all the events. After that the epochs are defined (three different ways to do that are implemented) and a downsampling will take place. The ML algorithm will classify the different configurations. Also there is a flow with xDawn filter applied in the pipeline and one without.
At the end there are some visualizations. At first there are all epochs visualized. Second the averaged epochs from one event over the whole set. At least there are the different accuracies visualized with matplotlib.pyplot.pcolormesh.




