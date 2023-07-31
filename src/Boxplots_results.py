#!usr/bin/python 

from ast import Num
import numpy as np
import matplotlib.pyplot as plt 
import os 
import csv
import seaborn as sns
import pandas as pd

# ****** Datasets of all evaluations ***** 

# Resulting paths  
result_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning/results/"

#title = "Comparison of training trials (Fcn net, continous classification)"
title = "fcn_network_results_34ch_time_frequency_features"#"Training trials compare pooling"
#xLabels = ["within sub", "cross sub"]
#xLabels = ["34 ch. motor", "exclude motor ch.", "all 64 ch."]
xLabels = ["time", "frequency", "fusion", "EEGNet"] #["standard 80", "standard 40", "standard 20", "pool 80","pool 40","pool 20", "pool 20 ot"]#, "eeg", "emg eeg fusion"]
result_list = ["fcn_network_results_34ch_time_domain_feat", "fcn_network_results_34ch_freq_domain_feat", "fcn_network_results_34ch_time_freq_domain_feat", "fcn_network_results_34ch_EEGNet_test"]
# "fcn_network_results_40_train_trials","fcn_network_results_30_train_trials", "fcn_network_results_20_train_trials", "fcn_network_results_10_train_trials", "fcn_network_results_5_train_trials"]# "fcn_network_results_eeg_all_subs", "fcn_network_results_eeg_emg_all_subs"]

compare_type = 'features'
#compare_type = 'number of training trials'
ylabel = "Performance (BA)"


# ****** Load the evaluation results from the individual resulting folders  ***** 
data_list = []

for file in result_list: 
    data = np.loadtxt(result_path+file, dtype = float, delimiter= ',')
    data_list.append(data)

data_arr = np.array(data_list) # has shape (evaluations, result vals, metric type)
data_arr_ba = data_arr[:, :, 0].T


data_frames = []
medians = []

for num_evaluations in range(0, data_arr_ba.shape[1]): 

    medians.append(np.median(data_arr_ba[:, num_evaluations]))  
    current_frame = pd.DataFrame({'Evaluation':data_arr_ba[:, num_evaluations],compare_type:xLabels[num_evaluations]})
    data_frames.append(current_frame)

all_frames = pd.concat(data_frames)


fig = plt.figure()
box_plot = sns.boxplot(data = all_frames, x = compare_type, y = 'Evaluation')

plt.ylabel(ylabel)
plt.title(title)
plt.ylim([0.5, 1.0])
plt.yticks(np.arange(0.5, 1.05, step= 0.05))

i = 0
for xtick in box_plot.get_xticks():
    vertical_offset = medians[i]* (-0.02)
    box_plot.text(xtick,medians[xtick] + vertical_offset, np.round(medians[xtick], 3), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
    
fig.savefig(title+".svg")

plt.show()



