#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

# This code represents the script for generating and formatting the input for the DeepHRD prediction package.

import os
import torch
import pandas as pd
import random
import sys


def collectSampleIndex (file):
	sample = file.split("/")[-1].split(".")[0]
	sampleIndex = int(tileConversionMatrix.loc[sample].iloc[0])
	if sampleIndex < 10:
		sampleIndex = "00" + str(sampleIndex)
	elif 100 > sampleIndex > 10:
		sampleIndex = "0" + str(sampleIndex)
	else:
		sampleIndex = str(sampleIndex)
	return(sampleIndex)

def filterDuplicates ():
	svsFiles = [x for x in os.listdir(svsPath) if ".svs" in x or "ndpi" in x]
	sampleFilesDict = {}
	finalFiles = []
	uniqueSamples = 0
	for file in svsFiles:
		sampleIndex = collectSampleIndex(file)
		if not os.path.exists(tilesPath + sampleIndex + "/"):
			continue
		sample = "-".join([y for y in file.split("/")[-1].split("-")[:3]])
		if sample not in sampleFilesDict:
			uniqueSamples += 1
			sampleFilesDict[sample] = []
		sampleFilesDict[sample].append(file)

	for sample in sampleFilesDict:
		if len(sampleFilesDict[sample]) > 1:
			finalFiles.append(random.sample(sampleFilesDict[sample], 1)[0])
		else:
			finalFiles.append(sampleFilesDict[sample][0])
	with open(outputPath + "slidesUsed.txt", "w") as out:
		for svsFile in finalFiles:
			print(svsFile, file=out)

	return(finalFiles)

def collectSampleFileNames ():
	svsFiles = list(sigMatrix.index)

	sampleFilesDict = {}
	finalFiles = []
	uniqueSamples = 0
	skippedSamps = 0
	for file in svsFiles:
		sampleIndex = collectSampleIndex(file)
		if not os.path.exists(tilesPath + sampleIndex + "/"):
			print(file, sampleIndex)
			skippedSamps += 1
			continue
		sample = sigMatrix.loc[file, 'patient']
		if sample not in sampleFilesDict:
			uniqueSamples += 1
			sampleFilesDict[sample] = []
		sampleFilesDict[sample].append(file)

	print(skippedSamps)
	return(sampleFilesDict)

def partitionSamplesSubtypesBasedonSampleName ():

	subtypes = {}
	count = 0
	svsSamples = collectSampleFileNames()
	for sample in svsSamples:
		currentMat = sigMatrix[sigMatrix['patient']==sample]

		try:
			subtype = currentMat['subtype']
			if type(subtype) == pd.Series:
				print(subtype)
				subtype = subtype[0]
		except:
			subtype = "Missing_meta"

		if subtype not in subtypes:
			subtypes[subtype] = [[], []]

		try:
			sigExposure = currentMat['label']
			if type(sigExposure) == pd.Series:
				sigExposure = sigExposure[0]

			# Allow for future changes to only include extreme values (above a specfic threshold, or below a second threshold)
			if sigExposure > 0:
				subtypes[subtype][0].append(sample)
			elif sigExposure <= 0:
				subtypes[subtype][1].append(sample)
		except:
			continue

	trainFilesPos = []
	trainFilesNeg = []
	valFilesPos = []
	valFilesNeg = []
	testFilesPos = []
	testFilesNeg = []

	for subtype in subtypes:
		downSampleLength = min(len(subtypes[subtype][0]), len(subtypes[subtype][1]))
		trainFilesPos += random.sample(subtypes[subtype][0], int(TRAIN_PERCT*downSampleLength))
		trainFilesNeg += random.sample(subtypes[subtype][1], int(TRAIN_PERCT*downSampleLength))

		leftOverTestValPos = [x for x in subtypes[subtype][0] if x not in trainFilesPos]
		leftOverTestValNeg = [x for x in subtypes[subtype][1] if x not in trainFilesNeg]
		leftOverDownSampleLength = min(len(leftOverTestValPos), len(leftOverTestValNeg))
		valFilesPos += random.sample(leftOverTestValPos, int(0.5*leftOverDownSampleLength))
		valFilesNeg += random.sample(leftOverTestValNeg, int(0.5*leftOverDownSampleLength))
		testFilesPos += [x for x in leftOverTestValPos if x not in valFilesPos]
		testFilesNeg += [x for x in leftOverTestValNeg if x not in valFilesNeg]

	trainFiles = trainFilesPos + trainFilesNeg
	valFiles = valFilesPos + valFilesNeg
	testFiles = testFilesPos + testFilesNeg
	trainFiles = [y for sublist in [svsSamples[x] for x in trainFiles] for y in sublist]
	valFiles = [y for sublist in [svsSamples[x] for x in valFiles] for y in sublist]
	testFiles = [y for sublist in [svsSamples[x] for x in testFiles] for y in sublist]
	trainFilesPos = [y for sublist in [svsSamples[x] for x in trainFilesPos] for y in sublist]
	trainFilesNeg = [y for sublist in [svsSamples[x] for x in trainFilesNeg] for y in sublist]

	return(trainFiles, valFiles, testFiles)


def partitionSamplesSubtypes ():

	subtypes = {}
	count = 0
	svsFiles = filterDuplicates()
	for files in svsFiles:
		if "ndpi" in files:
			sample = files.split("_")[0].split("-")[0]
		else:
			sample = "-".join([x for x in files.split("-")[:3]])
		try:
			subtype = subtypeMatrix.loc[sample, 'Subtype']
		except:
			subtype = "Missing_meta"
		if subtype not in subtypes:
			subtypes[subtype] = [[], []]

		try:
			sigExposure =  sigMatrix.loc[sample, SIG]
			if sigExposure > SIG_CUTOFF:
				subtypes[subtype][0].append(files)
			elif sigExposure <= SIG_CUTOFF_LOWER:
				subtypes[subtype][1].append(files)
		except:
			continue

	trainFilesPos = []
	trainFilesNeg = []
	valFilesPos = []
	valFilesNeg = []
	testFilesPos = []
	testFilesNeg = []

	for subtype in subtypes:
		downSampleLength = min(len(subtypes[subtype][0]), len(subtypes[subtype][1]))
		trainFilesPos += random.sample(subtypes[subtype][0], int(TRAIN_PERCT*downSampleLength))
		trainFilesNeg += random.sample(subtypes[subtype][1], int(TRAIN_PERCT*downSampleLength))

		leftOverTestValPos = [x for x in subtypes[subtype][0] if x not in trainFilesPos]
		leftOverTestValNeg = [x for x in subtypes[subtype][1] if x not in trainFilesNeg]
		valFilesPos += random.sample(leftOverTestValPos, int(VAL_PERCT*len(leftOverTestValPos)))
		valFilesNeg += random.sample(leftOverTestValNeg, int(VAL_PERCT*len(leftOverTestValNeg)))
		testFilesPos += [x for x in leftOverTestValPos if x not in valFilesPos]
		testFilesNeg += [x for x in leftOverTestValNeg if x not in valFilesNeg]

	trainFiles = trainFilesPos + trainFilesNeg
	valFiles = valFilesPos + valFilesNeg
	testFiles = testFilesPos + testFilesNeg
	return(trainFiles, valFiles, testFiles)



def partitionSamples ():
	posFiles = []
	negFiles = []
	count = 0
	for files in [x for x in os.listdir(svsPath) if ".svs" in x or "ndpi" in x]:	
		if "ndpi" in files:
			sample = files.split("_")[0].split("-")[0]
		else:
			sample = "-".join([x for x in files.split("-")[:3]])
		try:
			sigExposure =  sigMatrix.loc[sample, SIG]
			if sigExposure > SIG_CUTOFF:
				posFiles.append(files)
			elif sigExposure <= SIG_CUTOFF_LOWER:
				negFiles.append(files)
		except:
			continue


	trainFilesPos = random.sample(posFiles, int(TRAIN_PERCT*len(posFiles)))
	trainFilesNeg = random.sample(negFiles, int(TRAIN_PERCT*len(posFiles)))
	leftOverTestValPos = [x for x in posFiles if x not in trainFilesPos]
	leftOverTestValNeg = [x for x in negFiles if x not in trainFilesNeg]
	valFilesPos = random.sample(leftOverTestValPos, int(0.5*len(leftOverTestValPos)))
	valFilesNeg = random.sample(leftOverTestValNeg, int(0.5*len(leftOverTestValPos)))
	testFilesPos = [x for x in leftOverTestValPos if x not in valFilesPos]
	testFilesNeg = [x for x in leftOverTestValNeg if x not in valFilesNeg]

	trainFiles = trainFilesPos + trainFilesNeg
	valFiles = valFilesPos + valFilesNeg
	testFiles = testFilesPos + testFilesNeg
	print("Bias for the two classes:")
	print("\tPosClass: ", len(trainFilesPos)/len(trainFiles))
	print("\tNegClass: ", len(trainFilesNeg)/len(trainFiles))
	return(trainFiles, valFiles, testFiles)


def partitionSamplesCustom (pathToSamplesLibrary):
	lib = torch.load(pathToSamplesLibrary)
	testFilesPos = []
	testFilesNeg = []
	for i,name in enumerate(lib['slides']):
		if lib['targets'][i] == 1:
			testFilesPos.append(name.split("/")[-1])
		else:
			testFilesNeg.append(name.split("/")[-1])

	posFiles = []
	negFiles = []
	for files in [x for x in os.listdir(svsPath) if ".svs" in x]:
		if files in testFilesPos:
			continue
		if files in testFilesNeg:
			continue
		sample = "-".join([x for x in files.split("-")[:3]])
		try:
			sigExposure =  sigMatrix.loc[sample, SIG]
			if sigExposure > SIG_CUTOFF:
				posFiles.append(files)
			elif sigExposure <= SIG_CUTOFF_LOWER:
				negFiles.append(files)
		except:
			continue
	trainFilesPos = random.sample(posFiles, int(TRAIN_PERCT*len(posFiles)))
	trainFilesNeg = random.sample(negFiles, int(TRAIN_PERCT*len(posFiles)))
	leftOverTestValPos = [x for x in posFiles if x not in trainFilesPos]
	leftOverTestValNeg = [x for x in negFiles if x not in trainFilesNeg]
	valFilesPos = random.sample(leftOverTestValPos, int(0.5*len(leftOverTestValPos)))
	valFilesNeg = random.sample(leftOverTestValNeg, int(0.5*len(leftOverTestValPos)))
	testFilesPos = [x for x in leftOverTestValPos if x not in valFilesPos]
	testFilesNeg = [x for x in leftOverTestValNeg if x not in valFilesNeg]


	trainFiles = trainFilesPos + trainFilesNeg
	valFiles = valFilesPos + valFilesNeg
	testFiles = testFilesPos + testFilesNeg
	return(trainFiles, valFiles, testFiles)





def gatherData (files, svsPath, tilesPath):
	data = {}
	data['slides'] = []
	data['tiles'] = []
	data['targets'] = []

	saveSamp = True
	countsPos = 0
	countsNeg = 0
	countsAmbig = 0
	samplesDone = []
	for file in files.index:
		try:
			sampleIndex = collectSampleIndex(file)
		except:
			continue

		currentTilePath = os.path.join(tilesPath, sampleIndex)
		if os.path.exists(currentTilePath) and len(os.listdir(currentTilePath)) > 0:
			newTiles = []
			for tileFile in [x for x in os.listdir(currentTilePath) if ".png" in x]:
				newTiles.append(os.path.join(currentTilePath, tileFile))
		else:
			continue

		# if True:

		#try:
		if 'sigMatrix' in globals():
			sigExposure =  sigMatrix.loc[file, "label"]
			sigExposureSoft = sigMatrix.loc[file, "softLabel"]
			if sigExposure > SIG_CUTOFF:
				countsPos += 1
				data['targets'].append(torch.tensor([1-sigExposureSoft, sigExposureSoft]))
			elif sigExposure <= SIG_CUTOFF_LOWER:
				countsNeg += 1
				data['targets'].append(torch.tensor([1-sigExposureSoft, sigExposureSoft]))
			else:
				countsAmbig += 1
				saveSamp = False
		else:
			data['targets'].append(torch.tensor([0, 1]))

		if saveSamp:
			data['slides'].append(os.path.join(svsPath, file))
			data['tiles'].append(newTiles)

		saveSamp = True

	return(data)


def collectFiles (prediction):
	if 'partition' not in sigMatrix.columns:
		print("No partition of samples listed in the metadata. Please add a 'partition' column to the metadata file that contains >1 'test' sample and >1 'train' and 'validation sample if training a model.")
		sys.exit()		
	if "test" not in list(sigMatrix['partition']):
		print("No test files listed in this data set. Please add a 'partition' column to the metadata file that contains >1 'test' sample.")
		sys.exit()

	testFiles = list(sigMatrix[sigMatrix['partition']=='test'].index)

	if prediction:
		return(testFiles)
	else:
		if "train" not in list(sigMatrix['partition']) or "validation" not in list(sigMatrix['partition']):
			print("No train and/or validation files listed in this data set. Please add a 'partition' column to the metadata file that contains >1 'test' sample.")
			sys.exit()
		trainFiles = list(sigMatrix[sigMatrix['partition']=='train'].index)
		valFiles = list(sigMatrix[sigMatrix['partition']=='validation'].index)
		return(trainFiles, valFiles, testFiles)


def  partitionSamplesSubtypesDefined (pathToSamplesLibrary):
	lib = torch.load(pathToSamplesLibrary)
	return([x.split("/")[-1] for x in lib['slides']])


def generateDataStructures (project, projectPath, metaDataFile, tilesPath, outputPath, prediction=True, softLabel=False):

	svsPath = os.path.join(projectPath, project)
	tileConversionFile = os.path.join(projectPath, "slideNumberToSampleName.txt")

	global tileConversionMatrix, sigMatrix, SIG_CUTOFF, SIG_CUTOFF_LOWER

	SIG_CUTOFF = 29
	SIG_CUTOFF_LOWER = 29

	tileConversionMatrix = pd.read_csv(tileConversionFile, sep="\t", header=None, index_col=1)
	tileConversionMatrix = tileConversionMatrix[~tileConversionMatrix.index.duplicated(keep='first')]

	sigMatrix = pd.read_csv(metaDataFile, sep="\t", header=0, index_col=0)

	if prediction:
		testFiles = collectFiles (prediction)
		testData = gatherData(sigMatrix.loc[testFiles], svsPath, tilesPath)
		torch.save(testData, os.path.join(outputPath,"testData.pt"))
	else:
		trainFiles, valFiles, testFiles = collectFiles (prediction)
		trainData = gatherData(sigMatrix.loc[trainFiles], svsPath, tilesPath)
		valData = gatherData(sigMatrix.loc[valFiles], svsPath, tilesPath)
		testData = gatherData(sigMatrix.loc[testFiles], svsPath, tilesPath)

		torch.save(trainData, os.path.join(outputPath,"trainData.pt"))
		torch.save(valData, os.path.join(outputPath,"valData.pt"))
		torch.save(testData, os.path.join(outputPath,"testData.pt"))
	


