#!usr/bin/python 


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

# ****** Datasets of all evaluations *****  

# Resulting paths  
result_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox/results/"

#title = "Comparison of training trials (Fcn net, continous classification)"
title = "offline vs. online performance" #"Training trials compare pooling"
#xLabels = ["within sub", "cross sub"] 
#xLabels = ["34 ch. motor", "exclude motor ch.", "all 64 ch."]
xLabels = ["offline EEGNet", "offline MLP", "zerophasebutterWhole", "zerophasegustav1", "forwardButter200msshift", "zerogustav50ms", "zerophasepaddMeanCorr", "zeroGustDetrend", "gustavZeroOwnPad", "ZeroOwnPadEven", "contiousZeroPhase", "baseline MLP", "MLP add PSD Bands", "8 ch"] #["standard 80", "standard 40", "standard 20", "pool 80","pool 40","pool 20", "pool 20 ot"]#, "eeg", "emg eeg fusion"]
result_list = ["offline_model25ms_EEGNet", "offline_model25ms_MLP", "zerophaseButterwholeseg_MLP_2024-06-12_23-10-10","meanCorrZeroGustav_MLP_2024-06-11_22-07-06", "forwardButter200msShift_MLP_2024-06-13_16-27-52", "zeroGustavCut50ms_MLP_2024-06-13_22-04-30", "zeroPhasePaddingMeanCorr_MLP_2024-06-14_10-04-35", 
               "zeroPhaseGustavDetrendAfter_MLP_2024-06-14_14-13-39", "zeroPhaseGustavOwnPadding_MLP_2024-06-16_18-36-30", "zeroPhaseOwnPaddingEvenNoMeanCorr_MLP_2024-06-16_20-16-17", "continousZeroPhaseTest_MLP_2024-06-28_10-50-49", "baselineZeroPhaseEvenPadMLP_MLP_2024-08-08_14-55-44", "MLPAdditionalPSDBands_MLP_2024-08-08_17-34-36", "Use8ChannelTest_MLP_2024-08-08_20-05-41"]


# "fcn_network_results_40_train_trials","fcn_network_results_30_train_trials", "fcn_network_results_20_train_trials", "fcn_network_results_10_train_trials", "fcn_network_results_5_train_trials"]# "fcn_network_results_eeg_all_subs", "fcn_network_results_eeg_emg_all_subs"]

compare_type = 'method'
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
plt.yticks(np.arange(0.5, 1.06, step= 0.05))

i = 0
for xtick in box_plot.get_xticks():
    vertical_offset = medians[i]* (-0.02)
    box_plot.text(xtick,medians[xtick] + vertical_offset, np.round(medians[xtick], 3), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
    
fig.savefig(title+".svg")

plt.show()



