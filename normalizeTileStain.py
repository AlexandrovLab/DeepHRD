#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

# This code represents the stain normalization script for the DeepHRD prediction package.
# 	The code directly implements the  Macenko method for normalization of tissue staining:
#		M. Macenko et al., ISBI 2009


import os
import normalizeStaining
import sys
import multiprocessing as mp


cutoff1 = 0

def listSamples(tilePath):
	return(sorted([x for x in os.listdir(tilePath) if "." != x[0]])[cutoff1:])


def listTiles (samplePath):
	return([os.path.join(samplePath,x) for x in os.listdir(samplePath) if x[0] != "."])


def stainNormalization (currentSamples, tilePath, outputPath):
	for i, sample in enumerate(currentSamples):
		samplePath = os.path.join(tilePath,sample)
		if os.path.exists(os.path.join(outputPath,sample)):
			continue
		else:
			os.makedirs(os.path.join(outputPath,sample))
		tiles = listTiles(samplePath + "/")
		for l, tile in enumerate(tiles):
			tileNum = tile.split(".")[0].split("/")[-1]
			try:
				normalizeStaining.normalizeStaining(tile, saveFile=os.path.join(outputPath,sample, tileNum), Io=240, alpha=1, beta=0.15)
			except:
				continue


def multiprocess_stainNorm (tilePath, outputPath, max_cpu):
	'''
	Initiates the input data for stain normalization, including parallelization.

	Parameters:
			tilePath	->	Path to tiles that are to be stain normalized (string)
			outputPath	->	Path to where the normalized tiles should be saved (string)
			max_cpu		->	Number of maximum CPUs to use for multiprocessing.

	Returns:
		None

	Outputs:
		Stain normalized tiles (.png) within outputPath.
	'''

	samples = listSamples(tilePath)

	# Set-up parallelization:
	if max_cpu:
		processors = max_cpu
	else:
		processors = mp.cpu_count()
	max_seed = processors
	if processors > len(samples):
		max_seed = len(samples)
	pool = mp.Pool(max_seed)

	iterations_parallel = [[] for i in range(max_seed)]
	iter_bin = 0

	for i in range(0, len(samples), 1):
		if iter_bin == max_seed:
			iter_bin = 0
		iterations_parallel[iter_bin].append(samples[i])
		iter_bin += 1

	# Perform stain normalization
	results = []
	for i in range (0, len(iterations_parallel), 1):
		r = pool.apply_async(stainNormalization, args=(iterations_parallel[i], tilePath, outputPath))
		results.append(r)
	pool.close()
	pool.join()

	for r in results:
		r.wait()
		if not r.successful():
			# Raises an error when not successful
			r.get()


 
