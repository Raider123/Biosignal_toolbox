#!usr/bin/python 

# imports 
import os
import numpy as np
import mne
import csv
import sys
from scipy.io.wavfile import read
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from shutil import copyfile
from scipy import signal 
import MarkerCalculationFunctions as CalcFct 
import matplotlib.pyplot as plt


# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data_temp') # path where the data lays in a temp folder 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 



# ******** Params specified ********** 


# Params specified by user 
#
# MovementOnsetThresh = 0.6 # in mm
#DeadSamples = int(4.8*2000) # nmber of dead samples where resting phase is 


OnsetMarkerNameToWrite = "S300" # Speechoffset (S300) is used 
#NoMoveMarkerNameToWrite = "S 99"


#audio params 
FsampleAudio = 44100 # 44,1 kHz
AudioThresh = 0.05 # 5 % of maximum speech magnitude

WriteMarkerfile = True # should we write a markerfile 

SubjectNames = ['JV43', 'RA12', 'AV82','XP01','UP28','ZS27','JD68','QS70']

SubjectFiles = ['20211210_r_JV43_intentional_unilateral_speech_set','20211216_r_RA12_intentional_unilateral_speech_set','20211220_r_AV82_intentional_unilateral_speech_set','20211222_r_XP01_intentional_unilateral_speech_set','20211223_r_UP28_intentional_unilateral_speech_set','20220104_r_ZS27_intentional_unilateral_speech_set','20220105_r_JD68_intentional_unilateral_speech_set','20220107_r_QS70_intentional_unilateral_speech_set'] #need to add more 
Path = 'Data/StudyMeasurements_PrepaUsedEMGChannelsred_for_Evaluation/'


# pre specified sets 
Sets = [2, 3, 4]
SetsJV43 = [2, 4, 5]
SetsXP01 = [1, 2, 4]
subjectNumbers = [0, 1 ,2, 3, 4, 5, 6, 7]


# Specifications
#Qualisys Specifications
FsamplQualisys = 500 #500 Hz samplerate 
RowsToData = 11  # How many rows till data starts in tsv. file 
Axis = 3 # axis of Position data 

# EMG Reading Parameter 
FsampleEMG = 2000 # in Hz (fixed)

FsampleEEG = 500 #sample Frequency EEG
StaticPhaseQualisys = np.arange(0,2000) # 1 sec in beginning is static (no movement) 


# ******** End Params ********** 


# Subject specifications
for Subject in subjectNumbers:
	if(Subject == 0): 
		SetsUsed = SetsJV43
	elif(Subject == 3): 
		SetsUsed = SetsXP01
	else: 
		SetsUsed = Sets
 

	for Set in SetsUsed:
		DatasetStrEMG =  Path+SubjectNames[Subject]+'/EMG_Video/'+SubjectFiles[Subject]+str(Set)+'.txt'
		DatasetStrEEG = Path+SubjectNames[Subject]+'/EEG/'+SubjectFiles[Subject]+str(Set)+'.vhdr'
		QualisysStr = Path+SubjectNames[Subject]+'/Qualisys/'+SubjectFiles[Subject]+str(Set)+'.tsv'
		EEGMarkerfile = Path+SubjectNames[Subject]+'/EEG/'+SubjectFiles[Subject]+str(Set)+'.vmrk'
		AudioStr = Path+SubjectNames[Subject]+'/Speech/'+SubjectFiles[Subject]+str(Set)+'.wav'
	

		raw = mne.io.read_raw_brainvision(DatasetStrEEG)


		#extracting marked events 
		events, event_dict = mne.events_from_annotations(raw)
		

		# Marker before cropping 
		MarkerIndizesOld = events[:,0]
		MarkerNumbersOld = events[:,2]

		StartMarker = np.where(MarkerNumbersOld == 8)[0][0]
		StartMarkerIndex = MarkerIndizesOld[StartMarker]
		StopMarker = np.where(MarkerNumbersOld == 9)[0][0]
		StopMarkerIndex = MarkerIndizesOld[StopMarker]

		#Time points equal to Marker (in seconds)
		StartTimeScenario = raw.times[StartMarkerIndex]
		StopTimeScenario = raw.times[StopMarkerIndex]

		#crop EEG Channels to Start and end of scenario 
		raw = raw.crop(tmin=StartTimeScenario, tmax=StopTimeScenario, include_tmax=True)

		# Marker after cropping
		events, event_dict = mne.events_from_annotations(raw)
		S8MarkerIndexOffset = events[0,0]
		events[:,0] = events[:,0]- S8MarkerIndexOffset # need to correct marker indizes 
		MarkerIndizesNew = events[:,0]
		MarkerNumbersNew = events[:,2]


		# Extract S22 (Movement Onset,leaving plate)
		S22Indizes = np.where(MarkerNumbersNew == 22)[0]
		S22Marker = MarkerIndizesNew[S22Indizes]
		# use leaving plate (S22) as movement onset marker 
		OnsetMarkerIndex = np.copy(S22Marker) 


		#MarkerNumbers: 
		#8: Start Scenario 
		#1: Start Qualisys/EMG aquisition (trigger)
		#9: End Scenario 

		#PushPlateIndex = PushPlateIndex[:]
		OnsetMarkerIndex1 = OnsetMarkerIndex.copy()
		OnsetMarkerIndex1 = OnsetMarkerIndex1[:]

		# **** Load Qualisys Data **** 

		    
		tsv_file = open(QualisysStr)
		QualisysDataset = csv.reader(tsv_file, delimiter="\t")

		QualisysDataMatrix = np.array(list(QualisysDataset))
		tsv_file.close()
		    
		MarkerNames = np.array(QualisysDataMatrix[RowsToData-2])
		MarkerNames = MarkerNames[1:len(MarkerNames)]
		NumOfMarkers = len(MarkerNames) # since one name is the word names 

		
		#Prepare Qualisys Data 
		Nsampl = len(QualisysDataMatrix) -(RowsToData)
		QualisysNpMatrix = np.zeros((Nsampl,NumOfMarkers*Axis))
		DataIndizes = np.arange(RowsToData+1,len(QualisysDataMatrix))

		# run through data lists and shape to numpy matrix 
		j = 0
		for i in DataIndizes:
			DataRows = np.array(QualisysDataMatrix[i]).astype(float)
			QualisysNpMatrix[j,:] = DataRows # has shape (Values,channels)
			j = j+1    
			


		RightHandIndex = np.where(MarkerNames == "RightHand")[0][0]
		LeftHandIndex = np.where(MarkerNames == "LeftHand")[0][0]

		#extract the movements 
		RightHandMovement = QualisysNpMatrix[:,RightHandIndex*3:RightHandIndex*3+3]
		LeftHandMovement = QualisysNpMatrix[:,LeftHandIndex*3:LeftHandIndex*3+3]


		# *** Load EMG Data ***
		# dont use for now 
		EMGFileStrings = np.loadtxt(DatasetStrEMG, dtype = str, delimiter=None, max_rows=4, skiprows=0)
		EMGDataNamesStr = np.loadtxt(DatasetStrEMG, dtype = str, delimiter=':', max_rows=1, skiprows=4)
		EMGDataNamesStr = EMGDataNamesStr[0:-1] # cut off last val 
		EMGDataAll = np.loadtxt(DatasetStrEMG, dtype = float, delimiter=None, skiprows=5)
		# First Row is Time Axis, Second is Data !
		UsedEMGChannels = np.array([0])
		EMGData = EMGDataAll[:,UsedEMGChannels]



		# *** Load audio data *** 
		Audiosignal = read(AudioStr) # if used 
		AudiosignalArray = np.array(Audiosignal[1],dtype= np.int16)
		
		
		# *** Data Aligning *** 

		# Qualisys 
		EMGQualisysMarkerIndex = np.where(MarkerNumbersOld == 1)[0]
		EMGQualisysStartStopIndex = MarkerIndizesOld[EMGQualisysMarkerIndex]

		# Presentation marker 
		StartScenarioIndex = StartMarkerIndex
		StopScenarioIndex = StopMarkerIndex

		#length of scenario 
		ScenarioTime = (StopScenarioIndex - StartScenarioIndex)*raw.times[1]
		ScenarioStartTimeFromTrigger = (StartScenarioIndex - EMGQualisysStartStopIndex[0])*raw.times[1]
		ScenarioStartIndex500Hz = int(ScenarioStartTimeFromTrigger*FsamplQualisys) 
		ScenarioStartIndex2000Hz = int(ScenarioStartTimeFromTrigger*FsampleEMG) 

		#length of data during scenario in samples 
		ScenarioSampleLength500Hz = int(ScenarioTime*FsamplQualisys) 
		ScenarioSampleLength2000Hz = int(ScenarioTime*FsampleEMG) 

		#Indizes for Slicing data 
		QualisysCorrection = 10 # 20 ms correction (trigger delay )
		QualisysStartIndex = int(ScenarioStartIndex500Hz-QualisysCorrection) 
		QualisysStopIndex = int(ScenarioStartIndex500Hz+ScenarioSampleLength500Hz-QualisysCorrection) 

		# EMG indizes 
		EMGStartIndex = int(ScenarioStartIndex2000Hz)
		EMGStopIndex = int(ScenarioStartIndex2000Hz+ScenarioSampleLength2000Hz) 

		# Align Qualisys 
		RightHandMovementAligned = RightHandMovement[QualisysStartIndex:QualisysStopIndex,:]
		LeftHandMovementAligned = LeftHandMovement[QualisysStartIndex:QualisysStopIndex,:]

		#Align EMG 
		EMGDataAligned = EMGData[EMGStartIndex:EMGStopIndex,:]

		RightHandMovementAlignedUp = signal.resample(RightHandMovementAligned,len(EMGDataAligned))
		LeftHandMovementAlignedUp = signal.resample(LeftHandMovementAligned,len(EMGDataAligned))



		# Audio signal indizes 
		AudioStartMarkerIndex = MarkerIndizesOld[np.where(MarkerNumbersOld == 16)][0]

		# Audio Start Stop indizes 
		ScenarioStartTimeFromAudioStart = (StartScenarioIndex - AudioStartMarkerIndex)*raw.times[1]
		ScenarioStartIndex44_1kHz = int(ScenarioStartTimeFromAudioStart*FsampleAudio) 

		#length of data during scenario in samples 
		ScenarioSampleLength44_1kHz = int(ScenarioTime*FsampleAudio) 

		# Audio indizes 
		AudioStartIndex = int(ScenarioStartIndex44_1kHz)
		AudioStopIndex = int(ScenarioStartIndex44_1kHz+ScenarioSampleLength44_1kHz) 

		#Align Audiosignal
		AudiosignalAligned = AudiosignalArray[AudioStartIndex:AudioStopIndex]
				                   

		#downsamplint to 2000 Hz 
		AudiosignalAlignedDown = signal.resample(AudiosignalAligned,len(EMGDataAligned))


		#substract minimum as endpoint 
		#Processed only with z-Axis 
		L = len(RightHandMovementAlignedUp) # length of signal 
		RightHandMovementOneAxisAlignedUpSub = RightHandMovementAlignedUp[:,2]-np.min(RightHandMovementAlignedUp[int((1/3)*L):int((2/3)*L),2])
		L = len(LeftHandMovementAlignedUp) # length of signal 
		LeftHandMovementOneAxisAlignedUpSub = LeftHandMovementAlignedUp[:,2]-np.min(LeftHandMovementAlignedUp[int((1/3)*L):int((2/3)*L),2])#norming 

		# processed (cutted) movement signals 
		RightHandMovementOneAxisAlignedNormUp = RightHandMovementOneAxisAlignedUpSub/np.max(RightHandMovementOneAxisAlignedUpSub)
		LeftHandMovementOneAxisAlignedNormUp = LeftHandMovementOneAxisAlignedUpSub/np.max(LeftHandMovementOneAxisAlignedUpSub)


		# standardize names of signals 
		ProcessedRightHandMovementOneAxisNorm = RightHandMovementOneAxisAlignedNormUp
		ProcessedRightHandMovement = RightHandMovementAlignedUp
		ProcessedLeftHandMovementOneAxisNorm = LeftHandMovementOneAxisAlignedNormUp
		ProcessedLeftHandMovement = LeftHandMovementAlignedUp

		ProcessedAudioSignal = AudiosignalAlignedDown


		#ProcessedEEGSignals = EEGSelecetedChannelsAlignedUpsample # Selected EEG electrodes 
		ProcessedTimeAxis = np.arange(0, len(EMGDataAligned)/FsampleEMG, step = 1/FsampleEMG)

		# markerboard onsets 
		MarkerBoardOnsetArray = np.zeros(ProcessedTimeAxis.shape)
		MarkerBoardOnsetArray[OnsetMarkerIndex*4] = 1 
		MarkerboardOnsetIndizes = np.where(MarkerBoardOnsetArray == 1)[0]

		# *** Velocity Calculations *** 

		ProcessedRightHandMovementCorrected= ProcessedRightHandMovement - np.mean(ProcessedRightHandMovement[StaticPhaseQualisys],axis =0) 

		# init 
		ProcessedRightHandVelocity_X = np.zeros(len(ProcessedRightHandMovementCorrected))
		ProcessedRightHandVelocity = np.zeros(len(ProcessedRightHandMovementCorrected))
		
		# time step 
		dt = ProcessedTimeAxis[1]-ProcessedTimeAxis[0]

		# Calculate velocity signal 
		for index in range(1,len(ProcessedRightHandMovementCorrected)):
			dx =(ProcessedRightHandMovementCorrected[index,0]-ProcessedRightHandMovementCorrected[index-1,0])
			dy =(ProcessedRightHandMovementCorrected[index,1]-ProcessedRightHandMovementCorrected[index-1,1])
			dz =(ProcessedRightHandMovementCorrected[index,2]-ProcessedRightHandMovementCorrected[index-1,2])
		    
			ProcessedRightHandVelocity[index] = ((np.sqrt((dx**2)+(dy**2)+(dz**2)))/dt)/1000 # in m/s

		ProcessedRightHandMovementCorrectedVector = np.sqrt((ProcessedRightHandMovementCorrected[:,0]**2)+(ProcessedRightHandMovementCorrected[:,1]**2)+(ProcessedRightHandMovementCorrected[:,2]**2))

		ProcessedRightHandMovementCorrectedVectorForMaxCalc = ProcessedRightHandMovementCorrectedVector.copy()
		#dont use begin and end values, sometimes weird movements are included 
		ProcessedRightHandMovementCorrectedVectorForMaxCalc[0:MarkerboardOnsetIndizes[0]] = 0
		ProcessedRightHandMovementCorrectedVectorForMaxCalc[MarkerboardOnsetIndizes[-1]:-1] = 0
		ProcessedRightHandMovementCorrectedVectorNorm = ProcessedRightHandMovementCorrectedVector/np.max(ProcessedRightHandMovementCorrectedVectorForMaxCalc)


		ProcessedRightHandVelocityFilter= ProcessedRightHandVelocity.copy()
		ProcessedRightHandVelocityFilter[0:9000] = 0
		ProcessedRightHandVelocityFilter[-9000:len(ProcessedRightHandVelocityFilter)] = 0
		
		b, a = signal.butter(4, Flowpass, 'lp', analog=False, fs = FsampleEMG)
		ProcessedRightHandVelocityNorm = signal.filtfilt(b, a, ProcessedRightHandVelocityFilter)

		
		#ProcessedRightHandVelocityNorm = MovingAverageFilter(ProcessedRightHandVelocityFilter,N)
		#normalize velocity vector 
		ProcessedRightHandVelocityNorm = ProcessedRightHandVelocityNorm/np.max(ProcessedRightHandVelocityNorm[MarkerboardOnsetIndizes[0]:MarkerboardOnsetIndizes[-1]])

		#velocity vector not used 
		WeightedMovementFusion = ProcessedRightHandVelocityNorm*ProcessedRightHandMovementCorrectedVector


		# use fused signal 
		RightHandMovementOnsetArray, RightHandMovementOnsetIndizes = CalcFct.GetMovementOnsetFromTrajectory(WeightedMovementFusion, MovementOnsetThresh, MarkerboardOnsetIndizes)
 		


		# *** Audio processing *** 

		# highpass since under 15 Hz are no human speech ! 
		b, a = signal.butter(2, 15, 'hp', analog=False, fs = FsampleEMG)
		ProcessedAudioSignalHp = signal.filtfilt(b, a, ProcessedAudioSignal)


		# abs signal
		ProcessedAudioSignalHpAbs = np.abs(ProcessedAudioSignalHp)
		ProcessedAudioSignalHpAbsNorm = ProcessedAudioSignalHpAbs/np.max(ProcessedAudioSignalHpAbs)

		b, a = signal.butter(2, 15, 'lp', analog=False, fs = FsampleEMG)
		ProcessedAudioSignalSmoothed = signal.filtfilt(b, a, ProcessedAudioSignalHpAbs)


		# for testing 
		plt.figure()
		plt.plot(ProcessedAudioSignalSmoothed/np.max(ProcessedAudioSignalSmoothed)) 
		plt.plot(WeightedMovementFusion/np.max(WeightedMovementFusion)) 
		plt.show()

		# norm filtered audio 
		#ProcessedAudioSignalSmoothedNorm = ProcessedAudioSignalSmoothed/np.max(ProcessedAudioSignalSmoothed[MarkerboardOnsetIndizes[0]:MarkerboardOnsetIndizes[-1]])
		

		# On- and offset detection 
		SpeechOffsetArray, SpeechOffsetIndizes = CalcFct.GetSpeechOffset(ProcessedAudioSignalSmoothed,AudioThresh, RightHandMovementOnsetIndizes)
		SpeechOnsetArray, SpeechOnsetIndizes = CalcFct.GetSpeechOnset(ProcessedAudioSignalSmoothed,AudioThresh, RightHandMovementOnsetIndizes)
		    
		print("Length Speech: ",len(SpeechOffsetIndizes)) 

		# prints with timings 
		print("Number of Qualisys onsets:", np.sum(RightHandMovementOnsetArray))
		print("Number of Markerboard onsets:", int(np.sum(SpeechOffsetArray)))

		#check for sizes 
		OffsetDiffs = SpeechOffsetIndizes[0:-1]-RightHandMovementOnsetIndizes[0:-1]
		print("Offset Diffs in ms: ",OffsetDiffs/2)
		MeanDiff = np.mean(OffsetDiffs)
		print("Mean diff speech offset vs. Qualisys", MeanDiff/2, "ms")
		STDDiff = np.std(OffsetDiffs)
		print("STD diff speech offset vs. Qualisys ",STDDiff/2, "ms")

		print("")
		#check for sizes 
		OnsetDiffs = SpeechOnsetIndizes[0:-1]-RightHandMovementOnsetIndizes[0:-1]
		print("Onset Diffs in ms: ",OnsetDiffs/2)
		MeanDiff = np.mean(OnsetDiffs)
		print("Mean diff speech onset vs. Qualisys", MeanDiff/2, "ms")
		STDDiff = np.std(OnsetDiffs)
		print("STD diff speech onset vs. Qualisys ",STDDiff/2, "ms")
		
		
		print("")
		#check for sizes 
		OnsetOffsetDiffs = SpeechOnsetIndizes[0:-1]-SpeechOffsetIndizes[0:-1]
		print("Audio Length Diffs in ms: ",OnsetOffsetDiffs/2)
		MeanDiff = np.mean(OnsetOffsetDiffs)
		print("Mean diff speech onset vs. speech offset (Word length)", MeanDiff/2, "ms")
		STDDiff = np.std(OnsetOffsetDiffs)
		print("STD diff speech onset vs. speech offset (Word length)",STDDiff/2, "ms")

		#where speech offset is late 
		#LateOffsets = (SpeechOffsetIndizes-RightHandMovementOnsetIndizes)/2 < 0 # 50 ms should at least be before speech is done and movement initiated 
		#RightHandMovementOnsetIndizesReduced = RightHandMovementOnsetIndizes[LateOffsets]
		
		
		# Onset marker indizes from Qualisys 
		SpeechOffsetIndizes500Hz = S8MarkerIndexOffset+((SpeechOffsetIndizes/4).astype(int))+1
		#RightHandMovementOnsetIndizesReduced500Hz = S8MarkerIndexOffset+((RightHandMovementOnsetIndizes/4).astype(int))+1
		
		
		# Select speech offset or Qualisys as marker 

		if(ApplyOffsetShift == True):  
			Used500HzMarker = SpeechOffsetIndizes500Hz+ SpeechOffsetShifts[Subject]
		else: 
			Used500HzMarker = SpeechOffsetIndizes500Hz #+ SpeechOffsetShifts[Subject]


		# no_move marker generation 
		NoMoveMarker1 = Used500HzMarker+NoMovementWindows[0] 
		NoMoveMarker2 = Used500HzMarker+NoMovementWindows[1] 
		NoMoveMarker3 = Used500HzMarker+NoMovementWindows[2] 

		# *** Write new markerfile ***

		if(WriteMarkerfile == True):
		 
			# use specified file for writing marker into it  
			MarkerFileStr = EEGMarkerfile # got marker file
			copyfile(MarkerFileStr, EEGMarkerfile+'_original')  # save the original
			fh, TempFileStr = mkstemp() # create temp file 

			MinNumOfCommas= 3
			newLine = ['']

			# open Markerfile and new temp file 
			with fdopen(fh, 'w') as New_File: 
				with open(MarkerFileStr) as Old_File:
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

								# loop over all no_move indizes  
								for i in range(0, len(NoMoveMarker3)): 
									if (CurrentStimIndex > NoMoveMarker1[i] and OldStimIndex < NoMoveMarker1[i]):
										#create new line with own onset marker 
										AddedSplitLine = SplitLine
										AddedSplitLine[1] = NoMoveMarkerNameToWrite
										AddedSplitLine[2] = str(NoMoveMarker3[i])
										AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
										New_File.write(AddedLine)
										AddedSplitLine[2] = str(NoMoveMarker2[i])
										AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
										New_File.write(AddedLine)    
										AddedSplitLine[2] = str(NoMoveMarker1[i])
										AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
										New_File.write(AddedLine) 


								#loop over all the onset indizes 
								for index in Used500HzMarker: 
									if (CurrentStimIndex > index and OldStimIndex < index):
										#create new line with own onset marker 
										AddedSplitLine = SplitLine
										AddedSplitLine[1] = OnsetMarkerNameToWrite
										AddedSplitLine[2] = str(index)
										AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
										# added new line 
										New_File.write(AddedLine)

									elif(CurrentStimIndex == index): # if indizes are both exacly the same 
										#create new line with own onset marker 
										AddedSplitLine = SplitLine
										AddedSplitLine[1] = OnsetMarkerNameToWrite
										AddedSplitLine[2] = str(index-1)
										AddedLine = (",". join(AddedSplitLine)) +'\n' # set line together
										# added new line 
										New_File.write(AddedLine)


						# current line gets added after onset line 
						New_File.write(Line)
						OldStimIndex = CurrentStimIndex

			#Copy the file permissions from the old file to the new file
			copymode(MarkerFileStr, TempFileStr)
			#Remove original file
			remove(MarkerFileStr)
			#Move new file
			move(TempFileStr, MarkerFileStr)

			#close file
			Old_File.close()
			New_File.close()

			# move files
			os.rename(MarkerFileStr, data_path+SubjectFiles[Subject]+str(Set)+'.vmrk') #move markerfile
			os.rename(EEGMarkerfile+'_original', EEGMarkerfile) # go back to old name for original file


			
			print("Manipulated the marker files!. New files are moved to markerfile folder.") 
		else: 
			print("Nothing done here ...") 
	

