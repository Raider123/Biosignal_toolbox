# MNE_machine_learning

## Description EEGFlow_MNE
Standard version of the EEG Flow from pySPACE.
Classify EEG data from single Trials into Standard or target with machine learning classifier. For the moment an SVM is used.
You need a set of EEG data in Brain Products format. Also they need to have markers in their *.vhdr file.
The code will apply a bandpass filter to the data and extract all the events. After that the epochs are defined (three different ways to do that are implemented) and a downsampling will take place. The ML algorithm will classify the different configurations. Also there is a flow with xDawn filter applied in the pipeline and one without.
At the end there are some visualizations. At first there are all epochs visualized. Second the averaged epochs from one event over the whole set. At least there are the different accuracies visualized with matplotlib.pyplot.pcolormesh




