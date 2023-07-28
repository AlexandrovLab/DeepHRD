#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

import pandas as pd
import numpy as np
import torch
import openslide
import matplotlib.pyplot as plt
import random
import os
import PIL.Image as Image
import pca 
import argparse
import multiprocessing as mp




def main ():
	'''
	Script to locate, select, and resample ROIs from the first resolution models. This method uses PCA to select the ROIs based upon
	the feature vectors extracted from the penultimate layer of the model's fully connected layers.

	Parameters:
		Passed via command line arguments (See command "python3 test_final.py -h" for more details)

	Returns:
		None

	Outputs:
		1. Newly resampled tiles at 20x magnification. 
		2. The new training, validaiton, and testing data structures.

	'''

	global args, slidePath, tileConversionMatrix, objectiveMat, RESOLUTION, outputPath, tileCountCutoff

	parser = argparse.ArgumentParser(description='Pulls regions of interests using a model trained at a lower magnification (i.e. 5x)')
	parser.add_argument('--project', type=str, default='', help='Project name')
	parser.add_argument('--projectPath', type=str, default='', help='Path to main project directory')
	# parser.add_argument('--model', type=str, default='1', help='Model number')
	# parser.add_argument('--checkpoint', type=str, default='', help='The checkpoint number')
	parser.add_argument('--objectiveFile', type=str, default='', help='The file containing the maximum objective magnification for each slide')
	parser.add_argument('--slidePath', type=str, default='.', help='Path to the original WSIs')
	parser.add_argument('--tileConv', type=str, default='.', help='File path to convert tile names back to sample names')

	parser.add_argument('--predict', action='store_true', help='Generate only for prediction samples; not for training/validating')

	parser.add_argument('--train_lib', type=str, default='', help='Path to the training datastructure. See README for more details on formatting')
	parser.add_argument('--val_lib', type=str, default='', help='Path to the validation datastructure. See README for more details on formatting')
	parser.add_argument('--test_lib', type=str, default='', help='Path to the testing datastructure. See README for more details on formatting')

	parser.add_argument('--feature_vectors_train', type=str, default='', help='Path to the training feature vectors.')
	parser.add_argument('--feature_vectors_val', type=str, default='', help='Path to the validation feature vectors.')
	parser.add_argument('--feature_vectors_test', type=str, default='', help='Path to the testing feature vectors.')
	parser.add_argument('--output', type=str, default='.', help='Path to the output where the new tiles will be saved')
	parser.add_argument('--maxROI', default=10000, type=int, help='Number of maximum ROIs that can be selected')
	parser.add_argument('--max_cpu', default=0, type=int, help='Maximum number of CPUs to utilize for parallelization (default: None - utilizes all available cpus)')

	args = parser.parse_args()
	
	# Resampling magnification
	RESOLUTION = 20
	# tileCountCutoff = 500000
	tileCountCutoff = args.maxROI


	outputPath = args.output
	slidePath = args.slidePath

	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	tileConversionMatrix = pd.read_csv(args.tileConv, sep="\t", header=None, index_col=1)
	tileConversionMatrix = tileConversionMatrix[~tileConversionMatrix.index.duplicated(keep='first')]
	objectiveMat = pd.read_csv(args.objectiveFile, header=None, names=['objective'], index_col=0, sep="\t")

	# Locate ROIs for the training, validation, and testing sets. The final output saves the new training, validation, and testing
	# data structures.
	predictionData = False
	if not args.predict:
		trainData20x = multiprocess_collectTiles (args.train_lib, args.feature_vectors_train, predictionData, args.max_cpu, trainData=True)
		torch.save(trainData20x, os.path.join(outputPath,"trainData20x.pt"))

		valData20x = multiprocess_collectTiles (args.val_lib, args.feature_vectors_val, predictionData, args.max_cpu, trainData=False)
		torch.save(valData20x, os.path.join(outputPath,"valData20x.pt"))

	else:
		predictionData = True

	testData20x = multiprocess_collectTiles (args.test_lib, args.feature_vectors_test, predictionData, args.max_cpu, trainData=False)
	torch.save(testData20x, os.path.join(outputPath,"testData20x.pt"))



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


def multiprocess_collectTiles (libraryfile, featureVectorsPath, predictionData, max_cpu, trainData):

	# Read in the feature vectors and the original library file
	featureVectors = pd.read_csv(featureVectorsPath, sep="\t", header=None, na_filter= False, index_col=[1,0]).fillna(0)
	lib = torch.load(libraryfile)

	trainData20x = {}
	trainData20x['slides'] = []
	trainData20x['tiles'] = []
	trainData20x['targets'] = []

	# Collect all available WSI
	availableSlides = [x for x in os.listdir(slidePath) if ".svs" in x or ".ndpi" in x or "jpg" in x]

	# Set-up parallelization:
	if max_cpu != 0:
		processors = max_cpu
	else:
		processors = mp.cpu_count()
	max_seed = processors
	if processors > len(availableSlides):
		max_seed = len(availableSlides)

	iterations_parallel = [[] for i in range(max_seed)]
	iter_bin = 0
	for i in range(0, len(availableSlides), 1):
		if iter_bin == max_seed:
			iter_bin = 0
		iterations_parallel[iter_bin].append(availableSlides[i])
		iter_bin += 1

	pool = mp.Pool(max_seed)
	results = []
	for i in range (0, len(iterations_parallel), 1):
		r = pool.apply_async(collectDownsampledTiles, args=(iterations_parallel[i], lib, featureVectors, predictionData, trainData))
		results.append(r)
	pool.close()
	pool.join()

	for r in results:
		r.wait()
		if not r.successful():
			# Raises an error when not successful
			r.get()

		currentTrainData20x = r.get()
		trainData20x['slides'] += currentTrainData20x['slides']
		trainData20x['tiles'] += currentTrainData20x['tiles']
		trainData20x['targets'] += currentTrainData20x['targets']

	# print("All ", trainData20x)
	return(trainData20x)



def collectDownsampledTiles (currentSamples, lib, featureVectors, predictionData, trainData):
	'''
	Performs PCA on the feature vectors for a single WSI at a time. The final ROIs are resampled and saved. 
	'''

	currentTrainData20x = {}
	currentTrainData20x['slides'] = []
	currentTrainData20x['tiles'] = []
	currentTrainData20x['targets'] = []

	currentAvailableSlides = currentSamples


	# Iterate across each WSI
	for x in set(featureVectors.index):	
		if x[1].split("/")[-1] not in currentSamples:
			continue
		try:
				# Collect relevant info for each WSI including the index number, sample name, sample index, and
				# the objective power. Determines the appropriate stepSize and tile size given the objective power
				# for the current slide.
				slideIDX = lib['slides'].index(x[1])
				sample = x[1].split("/")[-1].split(".")[0]
				sampleIndex = collectSampleIndex(sample)
				objective_power = objectiveMat.loc[int(sampleIndex), 'objective']
				if type(objective_power) == pd.Series:
					objective_power = list(objective_power)[0]
				if objective_power == 40:
						if RESOLUTION == 20:
								stepStize = 512
								length = 2048
				elif objective_power == 20:
						if RESOLUTION == 20:
								stepStize = 256
								length = 1024                           
		except:
				continue

		# Create the output directory for the new tiles for the current sample
		if not os.path.exists(os.path.join(outputPath, sampleIndex)):
				os.makedirs(os.path.join(outputPath, sampleIndex))

		# Downsample the feature vector file to the current sample and pull out the x,y coordinates for each tile
		currentFrame = featureVectors.iloc[featureVectors.index.get_level_values(1) == x[0]]
		coords = featureVectors.iloc[featureVectors.index.get_level_values(1) == x[0], [0, 1]]

		# Select the activation values for each node from the feature vector and perform PCA
		pcaFeatures = currentFrame[[i for i in range(4, 517)]]
		pcaFeatures.reset_index(drop=True, inplace=True)
		if len(coords.to_numpy().tolist()) < 2:
			continue
		indeces = pca.pcaCalc (pcaFeatures, False, outputPath, sample, '1', sample)
		
		# Pull out the top tile coordinates based on the selected indeces from the PCA along with their
		# corresponding probabilites
		topTiles = [list(coords.iloc[x]) for x in indeces]
		probs = [float(currentFrame.iloc[x, 2]) for x in indeces]

		# If the current data is from the training set, limit the max number of tiles
		if trainData or predictionData:
			if len(topTiles) > tileCountCutoff:
				topTiles = random.sample(topTiles, tileCountCutoff)

		# Open the current slide
		try:
			newSlide = [y for y in currentAvailableSlides if sample in y][0]
			s = openslide.open_slide(os.path.join(slidePath,newSlide))
			currentAvailableSlides.remove(newSlide)
		except:
			continue

		# For each selected ROI, resample at 20x magnification. This saves the new tiles into the current outputPath specified above.
		currentGrid = []
		for maxTile in topTiles:
			xPos = int(maxTile[0][1:])
			yPos = int(maxTile[1][1:])
			for i in range(xPos, xPos+length, stepStize):
				for l in range(yPos, yPos+length, stepStize):
					img_path = os.path.join(outputPath, sampleIndex, "-".join([args.project, sampleIndex, "tile", "x" + str(i), "y" + str(l), "w256", "h256.png"]))
					currentGrid.append(img_path)
					try:
						tile_region = s.read_region((i, l), 0, (stepStize, stepStize))
						tile_region = tile_region.resize((256,256),Image.BILINEAR)
						pil_img = tile_region.convert("RGB")
						pil_img.save(img_path)
					except:
						continue

		# Add the resampled tiles to the new data structure for training/validating/testing the second resolution model.
		currentTrainData20x['slides'].append(x[1])
		currentTrainData20x['tiles'].append(currentGrid)
		currentTrainData20x['targets'].append(lib['targets'][slideIDX])

	return(currentTrainData20x)



if __name__ == '__main__':
	main()
