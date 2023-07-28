#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
import base.utilsModel as util
import multiprocessing as mp

def convertColor (color):
	newColor = ()
	for x in color:
		newColor = newColor + (x*255,)
	return(newColor)


def extractCoordinates (tile, gap):
	coords = tile.split("/")[-1].split("tile-")[-1]
	if gap == 512:
		x = int(coords.split("-")[0][1:])
		y = int(coords.split("-")[1][1:])
	else:	
		x = int(coords.split("-")[2][1:])
		y = int(coords.split("-")[3][1:])

	return(x, y)


def displayTile (tile, tileProbs, gap, resolution, objective_power):
	img = mpimg.imread(tile)
	img *=255
	img = img.astype(np.int16)
	xPos, yPos = extractCoordinates(tile, gap)
	tileProb = float(tileProbs[str(xPos)+":"+str(yPos)])
	cmap = matplotlib.cm.get_cmap('seismic')
	mask  = np.full((256,256,3), convertColor(cmap(tileProb)[:3]), np.int16)
	img  = cv2.addWeighted(img, .2, mask, .8, 0)


	if objective_power == 20: 
		if resolution == '5x':
			gap -= 1024
		elif resolution == '20x':
			gap -= 256
	if str(xPos)+":"+str(yPos) in tileProbs20x and resolution == '5x':
		borderSize = 80
		borderoutput = cv2.copyMakeBorder(
    	# img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_CONSTANT, value=[255, 180, 0])
    	img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_CONSTANT, value=[153, 255, 51])
		plt.imshow(borderoutput, extent =[xPos, xPos+gap, yPos, yPos+gap], alpha=0.8)

	else:
		c = plt.imshow(img,
		                 extent =[xPos, xPos+gap, yPos, yPos+gap],
		                    interpolation ='nearest', origin ='lower', alpha=0.8)

	
	return(xPos, yPos)


def multiprocess_plotMasks (tilePath5x, tilePath20x, featurePath5x, featurePath20x, objectiveFile, tileConv, outputPath, max_cpu):
	global tileProbs20x, objectiveMat

	objectiveMat = pd.read_csv(objectiveFile, header=None, names=['objective'], index_col=0, sep="\t")
	tileConversionMatrix = pd.read_csv(tileConv, sep="\t", header=None, index_col=1)
	tileConversionMatrix = tileConversionMatrix[~tileConversionMatrix.index.duplicated(keep='first')]

	featureVectors20x = pd.read_csv(featurePath20x, sep="\t", header=None, na_filter= False, index_col=[1,0]).fillna(0)
	featureVectors5x = pd.read_csv(featurePath5x, sep="\t", header=None, na_filter= False, index_col=[1,0]).fillna(0)


	availableSamples = []
	sampleDict = {}
	for sample in set(featureVectors5x.index):
		if sample[1] == '':
			continue
		if sample[1] not in sampleDict:
			sampleDict[sample[1]] = sample[0]
			availableSamples.append(sample[1])


	# Set-up parallelization:
	if max_cpu:
		processors = max_cpu
	else:
		processors = mp.cpu_count()
	max_seed = processors
	if processors > len(availableSamples):
		max_seed = len(availableSamples)

	iterations_parallel = [[] for i in range(max_seed)]
	iter_bin = 0
	for i in range(0, len(availableSamples), 1):
		if iter_bin == max_seed:
			iter_bin = 0
		iterations_parallel[iter_bin].append(availableSamples[i])
		iter_bin += 1

	pool = mp.Pool(max_seed)
	results = []
	for i in range (0, len(iterations_parallel), 1):
		r = pool.apply_async(plotMasks, args=(iterations_parallel[i], sampleDict, tilePath5x, tilePath20x, featureVectors5x, featureVectors20x, objectiveMat, tileConversionMatrix, outputPath))
		results.append(r)
	pool.close()
	pool.join()

	for r in results:
		r.wait()
		if not r.successful():
			# Raises an error when not successful
			r.get()


def plotMasks (currentSamples, sampleDict, tilePath5x, tilePath20x, featureVectors5x, featureVectors20x, objectiveMat, tileConversionMatrix, outputPath):

	global tileProbs20x 

	for sample in set(featureVectors20x.index):
		if sample[1] not in currentSamples:
			continue
		gap =512

		resolution = "20x"
		if sample[1] == '':
			continue

		sampleIndex = util.collectSampleIndex(sample[1], tileConversionMatrix)
		objective_power = objectiveMat.loc[int(sampleIndex), 'objective']

		if type(objective_power) == pd.Series:
			objective_power = list(objective_power)[0]

		coords = featureVectors20x.iloc[(featureVectors20x.index.get_level_values(1) == sample[0]) | (featureVectors20x.index.get_level_values(1) == sample[1]), [0, 1]]
		xCoords = list(coords.loc[:,2])
		yCoords = list(coords.loc[:,3])
		probs = featureVectors20x.iloc[(featureVectors20x.index.get_level_values(1) == sample[0]) | (featureVectors20x.index.get_level_values(1) == sample[1]), 2]


		try:
			tiles = [os.path.join(tilePath20x,sampleIndex,x) for x in os.listdir(os.path.join(tilePath20x,sampleIndex)) if ".png" in x]
		except:
			continue
		tileProbs20x = {}
		for x,y,prob in zip(xCoords, yCoords, probs):
			tileProbs20x[x[1:]+":"+y[1:]] = prob

		xMin = 100000000
		xMax = 0
		yMin = 100000000
		yMax = 0

		for tile in tiles:
			try:
			# if True:
				xPos, yPos = displayTile(tile, tileProbs20x, gap, resolution, objective_power)
				xMax = max(xMax, xPos+gap)
				yMax = max(yMax, yPos+gap)
				xMin = min(xMin, xPos)
				yMin = min(yMin, yPos)
			except:
				# print(tile)
				continue

		plt.xlim([xMin, xMax])
		plt.ylim([yMin, yMax])
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.savefig(os.path.join(outputPath, "probability_masks", sample[1].split("/")[-1].split(".")[0] + "_20x.pdf"), dpi=1000)
		plt.close()


		gap = 2048
		resolution = "5x"

		coords = featureVectors5x.iloc[featureVectors5x.index.get_level_values(1) == sampleDict[sample[1]], [0, 1]]

		xCoords = list(coords.loc[:,2])
		yCoords = list(coords.loc[:,3])
		probs = featureVectors5x.iloc[featureVectors5x.index.get_level_values(1) == sampleDict[sample[1]], 2]


		try:
			tiles = [os.path.join(tilePath5x,sampleIndex,x) for x in os.listdir(os.path.join(tilePath5x,sampleIndex)) if ".png" in x]
		except:
			continue
		tileProbs = {}
		for x,y,prob in zip(xCoords, yCoords, probs):
			tileKey = x[1:]+":"+y[1:]
			tileProbs[tileKey] = prob

		xMin = 100000000
		xMax = 0
		yMin = 100000000
		yMax = 0

		for tile in tiles:
			try:
			# if True:
				xPos, yPos = displayTile(tile, tileProbs, gap, resolution, objective_power)
				xMax = max(xMax, xPos+gap)
				yMax = max(yMax, yPos+gap)
				xMin = min(xMin, xPos)
				yMin = min(yMin, yPos)
			except:
				continue
				# print(tile)

		plt.xlim([xMin, xMax])
		plt.ylim([yMin, yMax])
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.savefig(os.path.join(outputPath, "probability_masks", sample[1].split("/")[-1].split(".")[0] + "_5x.pdf"), dpi=1000)
		plt.close()



