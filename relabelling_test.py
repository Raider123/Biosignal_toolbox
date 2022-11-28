
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt

# own libs 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder 
import eeg_lib


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

determine_labels = 3
searching_bounds = [61, 81]

ba_fixed_labels = []
ba_relabelling_labels = []
tnr_relabelling = []
tpr_relabelling = []

N = 1000
for i in range(0, N): 

    rand_arr = np.random.rand(40, 81)
    rand_arr_bin = rand_arr > 0.5

    fixed_labels = np.zeros(rand_arr.shape) 
    fixed_labels[:, searching_bounds[0]:] = 1.0 

    new_true_labels = eeg_lib.applyRelabelling(rand_arr_bin, determine_labels, searching_bounds)

    tnr_fix, tpr_fix, acc_fix, ba_fix = eeg_lib.calcTestAccAndRates(rand_arr_bin.flatten(), fixed_labels.flatten())

    #print("BA fixed: ", ba_fix)

    tnr, tpr, acc, ba = eeg_lib.calcTestAccAndRates(rand_arr_bin.flatten(), new_true_labels.flatten())
    
    #print("BA: ", ba)

    ba_fixed_labels.append(ba_fix)
    ba_relabelling_labels.append(ba)
    tnr_relabelling.append(tnr)
    tpr_relabelling.append(tpr)


ba_fixed_labels_mean = np.mean(np.array(ba_fixed_labels)) 
ba_relabelling_labels_mean = np.mean(np.array(ba_relabelling_labels)) 
tpr_relabelling_mean = np.mean(np.array(tpr_relabelling)) 
tnr_relabelling_mean = np.mean(np.array(tnr_relabelling)) 


print("mean ba fixed labels: ", np.round(ba_fixed_labels_mean, 3))
print("mean ba relabelling : ", np.round(ba_relabelling_labels_mean, 3))  
print("mean tnr relabelling : ", np.round(tnr_relabelling_mean, 3)) 
print("mean tpr relabelling : ", np.round(tpr_relabelling_mean, 3)) 
