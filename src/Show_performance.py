#!usr/bin/python 

from ast import Num
import numpy as np
import matplotlib.pyplot as plt 
import os 
import csv

# ****** Datasets of all evaluations ***** 

# Resulting paths  
result_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox/results/"

filename = "offline_oldwindows_EEGNet"

result_arr1 = np.loadtxt(result_path+filename, dtype = float, delimiter= ',')


print("Mean, BA : ", np.round(np.mean(result_arr1[:, 0]), 3))
print("Mean, TPR : ", np.round(np.mean(result_arr1[:, 1]), 3)) 
print("Mean, TNR : ", np.round(np.mean(result_arr1[:, 2]), 3)) 
print("")





