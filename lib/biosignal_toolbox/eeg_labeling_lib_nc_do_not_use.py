# *** Functions *** 

#!usr/bin/python 

# imports 
import os
import numpy as np
import mne
import csv
from numpy import genfromtxt
from scipy.io.wavfile import read
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from shutil import copyfile
from scipy import signal 

def WeightingOptimization(STDs, Means): 

	a = 1/8000
	b = 1/40
	c = 41/20
	WeightFunction = np.zeros(len(Means))
	ResultVec = np.zeros(len(Means))
	for i in range(0, len(Means)): 
		WeightFunction[i] = (a*Means[i]**2) +b*Means[i] +c
		ResultVec[i] = WeightFunction[i]*STDs[i]


	return ResultVec

#Onset detection functions 
def Variance_Filter(Signal, VarianceWindow):
	# signal init 
	EMGVariances = np.zeros(len(Signal))
    
	for index in range(VarianceWindow,len(Signal)): 
		if (index < VarianceWindow): 
			EMGVariances[index] = 0
		else: 
			EMGVariances[index] = np.var(Signal[index-VarianceWindow:index])
	return EMGVariances


def Find_EMG_Onset_Treshold(ProcessedSignal, QualisysOnsetIndizes, k, BaselineSlice):
	TimeBeforeEMGOnset = -1000 # 0.5 sec * 2000 Hz
	OnsetIndizes = []
	BaselineStd = np.std(ProcessedSignal[BaselineSlice])
	#print('baseline Std:',BaselineStd)
	BaselineMean= np.mean(ProcessedSignal[BaselineSlice])
	#print('baseline Mean:',BaselineMean)
	BaseThresh = BaselineMean + k* BaselineStd  # changed here!! 
	#print(BaseThresh) 
    
	for currentIndex in QualisysOnsetIndizes:

		StartIndex = TimeBeforeEMGOnset+ currentIndex
		i = 0
		maxIter = 4000 # maximum search area 2 seconds 
        
		while ((ProcessedSignal[StartIndex+i]) < BaseThresh and i < maxIter): # form right go from movement to start 
			i = i+1
		#print((ProcessedSignal[StartIndex+i]))
		# found the onset 
		OnsetIndizes.append(StartIndex+i) # append found onset 
        
	OnsetIndizes = np.array(OnsetIndizes).astype(int)
	BinaryOnsetArray = np.zeros(ProcessedSignal.shape)
	BinaryOnsetArray[OnsetIndizes] = 1
	BinaryOnsetArray = BinaryOnsetArray.astype(bool)
    
	return BinaryOnsetArray, OnsetIndizes

def CalcdesynchronisationEMGEEG(EMGIndex, FsampleEMG):
    # calculated regression parameters 
    m = -5.752301373834057e-06
    b = 0.42599065190334207
    dt = m*EMGIndex+b # shift in ms /sample 
    shiftedSamples500Hz = dt/(FsampleEMG/1000)
    shiftedSamples500Hz = shiftedSamples500Hz.astype(int) # samples are in ints 
    
    return dt, shiftedSamples500Hz


def getMovementOnsetFromTrajectory(Signal, Thresh, TasterOnsetIndizes):
    TimeBeforeBoardOnset = 600 # 0.3 sec * 2000 Hz
    OnsetIndizes = []
    for currentIndex in TasterOnsetIndizes:
        # use mean of static phase as reference 
        StaticMean = np.mean(Signal[(currentIndex-5000):(currentIndex-400)]) # mean 
        StartIndex = TimeBeforeBoardOnset+ currentIndex
        i = 0
        
        while (np.abs(Signal[StartIndex+i] - StaticMean) > Thresh): # form right go from movement to start 
            i = i-1
        # found the onset 
        OnsetIndizes.append(StartIndex+i) # append found onset 
        
    OnsetIndizes = np.array(OnsetIndizes).astype(int)
    BinaryOnsetArray = np.zeros(Signal.shape)
    BinaryOnsetArray[OnsetIndizes] = 1
    BinaryOnsetArray = BinaryOnsetArray.astype(bool)
    
    return BinaryOnsetArray, OnsetIndizes

 
def getSpeechOffset(Signal, Thresh, QualisysOnsetIndizes):
	#TimeAfterQualisysOnset = 1000 # 0.5 sec * 2000 Hz
	TimeBeforeMaxSearch = 4000 # search 2 seconds before Qualisys onset for max magnitude
    	
	OnsetIndizes = []
	for currentIndex in QualisysOnsetIndizes:
		MaxVal = np.max(Signal[currentIndex-TimeBeforeMaxSearch:currentIndex])
		MaxIndex = np.argmax(Signal[currentIndex-TimeBeforeMaxSearch:currentIndex])
		CurrentThresh = Thresh*MaxVal
		# use mean of static phase as reference 
		StartIndex =currentIndex-(TimeBeforeMaxSearch-MaxIndex)
		i = 0
		while (Signal[StartIndex+i] > CurrentThresh): # form right go from movement to start 
			i = i+1
		# found the onset 
		OnsetIndizes.append(StartIndex+i) # append found onset 
        
	OnsetIndizes = np.array(OnsetIndizes).astype(int)
	BinaryOnsetArray = np.zeros(Signal.shape)
	BinaryOnsetArray[OnsetIndizes] = 1
	BinaryOnsetArray = BinaryOnsetArray.astype(bool)
    
	return BinaryOnsetArray, OnsetIndizes

def getSpeechOnset(Signal, Thresh, QualisysOnsetIndizes):
	#TimeAfterQualisysOnset = 1000 # 0.5 sec * 2000 Hz
	TimeBeforeMaxSearch = 4000 # search 2 seconds before Qualisys onset for max magnitude
	TimeToSearchBeforeMax = 2000 # 1 sec before max of speech 
	OnsetIndizes = []
	for currentIndex in QualisysOnsetIndizes:
		MaxVal = np.max(Signal[currentIndex-TimeBeforeMaxSearch:currentIndex])
		MaxIndex = np.argmax(Signal[currentIndex-TimeBeforeMaxSearch:currentIndex])
		CurrentThresh = Thresh*MaxVal
		# use mean of static phase as reference 
		StartIndex =currentIndex-(TimeBeforeMaxSearch-MaxIndex)-TimeToSearchBeforeMax
		i = 0
		while (Signal[StartIndex+i] < CurrentThresh):  
			i = i+1
		# found the onset 
		OnsetIndizes.append(StartIndex+i) # append found onset 
        
	OnsetIndizes = np.array(OnsetIndizes).astype(int)
	BinaryOnsetArray = np.zeros(Signal.shape)
	BinaryOnsetArray[OnsetIndizes] = 1
	BinaryOnsetArray = BinaryOnsetArray.astype(bool)
    
	return BinaryOnsetArray, OnsetIndizes


def writeBpMarkersToMarkerfile(markernumber, markerfile_str, marker_indizes, data_path): 
	# *** Write new markerfile ***

	markerfile = os.path.join(data_path, markerfile_str) # get full path to file 

	# use specified file for writing marker into it  
	copyfile(markerfile, markerfile+'_original')  # save the original
	fh, TempFileStr = mkstemp() # create temp file 

	MinNumOfCommas= 3

	# open Markerfile and new temp file 
	with fdopen(fh, 'w') as New_File: 
		with open(markerfile) as Old_File:
			OldStimIndex = 0 # init with 0 (no markers before )
			CurrentStimIndex = 0

			for Line in Old_File:
				DecodeLine = Line
				Stripped = DecodeLine.strip("=")

				if (Stripped[0:2]== "Mk"): # these are the markers 
					#decoding lines 
					DecodeLine = DecodeLine.strip('\n') 
					SplitLine = DecodeLine.split(',')

					#find lines with marker information 
					if (len(SplitLine) > MinNumOfCommas): 
						StimNumber = SplitLine[1]
						StimIndex  = SplitLine[2]
						CurrentStimIndex = int(StimIndex)


						#loop over all the marker indizes 
						for index in marker_indizes: 
							if (CurrentStimIndex > index and OldStimIndex < index):
								#create new line with own onset marker 
								AddedSplitLine = SplitLine
								AddedSplitLine[1] = markernumber
								AddedSplitLine[2] = str(index)
								AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
								# added new line 
								New_File.write(AddedLine)

							elif(CurrentStimIndex == index): # if indizes are both exacly the same 
								#create new line with own onset marker 
								AddedSplitLine = SplitLine
								AddedSplitLine[1] = markernumber
								AddedSplitLine[2] = str(index-1)
								AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
								# added new line 
								New_File.write(AddedLine)


				# current line gets added after onset line 
				New_File.write(Line)
				OldStimIndex = CurrentStimIndex

	#Copy the file permissions from the old file to the new file
	copymode(markerfile, TempFileStr)
	#Remove original file
	remove(markerfile)
	#Move new file
	move(TempFileStr, markerfile)

	#close file
	Old_File.close()
	New_File.close()

	# move files
	os.rename(markerfile+'_original', markerfile) # go back to old name for original file

	print("Manipulated the marker files!") 
