#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

# This code represents the main script for the DeepHRD package for training a new model(s).
# 	The code includes and adapts preprocessing methods of whole-slide images [1],
# 	the normalization for H&E staining [2] and the implementation of a multi-resolution 
#	prediction module that is based upon a multiple-instance learning framework [3]. 
#	The associated functions for processing tissue slides have been adapted from an open 
#	source tutorial on working with pathology slides from IBM: 
#		[1] https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images
#	The Macenko method for normalization of tissue staining is used directly 
#		[2] M. Macenko et al., ISBI 2009
#	The MIL backbone has been adpated from a previous implementation developed by Campanella et al. 2019:
#		[3] Campanella, G., Hanna, M.G., Geneslaw, L. et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat Med 25, 1301–1309 (2019). https://doi.org/10.1038/s41591-019-0508-1


import argparse
import os
import preprocessing_with_tile_data_overlap as preproc
import normalizeTileStain
import gatherDataSets
from base import utilsModel
import shutil
import sys
import pandas as pd
import plotProbabilityMasks
import multiprocessing
import torch


# python3 DeepHRD_train.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/output/ --max_gpu 1 --ensemble 5 --dropoutRate 0.7 --python python3 --metadata /restricted/alexandrov-ddn/users/ebergstr/histology/data/BRCA_HRD_DeepHRD_flash_frozen_meta.txt --preprocess --stainNorm --softLabel --generateDataSets --train5x --pullROIs --train20x --workers 16
# python3 DeepHRD_train.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/output2/ --max_gpu 4 --ensemble 4 --dropoutRate 0.7 --python python3 --metadata /restricted/alexandrov-ddn/users/ebergstr/histology/data/BRCA_HRD_DeepHRD_flash_frozen_meta.txt --stainNorm --softLabel --generateDataSets --train5x --calcFeatures --pullROIs --train20x --workers 16 --epochs 10
def main ():
	print("\n\n===================================================================================================================================================", flush=True)
	print("===================================================================================================================================================", flush=True)
	print("|\t\t\t\t\t\t\t\tWelcome to DeepHRD\t\t\t\t\t\t\t          |", flush=True)
	print("|\t\t\t\t\t\t\t\t------------------\t\t\t\t\t\t\t          |", flush=True)
	print("|\t\t\t\t\t\t\t     Author: Erik N Bergstrom\t\t\t\t\t\t\t          |", flush=True)
	print("|\t\t\t\t\t\t\t       Alexandrov Lab, UCSD\t\t\t\t\t\t\t          |", flush=True)
	print("===================================================================================================================================================", flush=True)
	print("===================================================================================================================================================\n\n", flush=True)
	print("Beginning analysis", flush=True)
	print("-----------------------------------------------\n\n", flush=True)

	parser = argparse.ArgumentParser(description='Multi-Resolution biomarker classifier prediction - 2023')

	parser.add_argument('--projectPath', type=str, default='', help='Path to the project directory')
	parser.add_argument('--project', type=str, default='BRCA', help='Project Name where the slides are located. projectPath + project should be the location to the slides.')
	parser.add_argument('--output', type=str, default=None, help='Path to the output and trained models are saved. Recommended projectPath + "output/')
	parser.add_argument('--metadata', type=str, default='', help='Path to the metadata file that contains the labels for each sample')

	parser.add_argument('--tileOverlap', default=0.0, type=float, help='The proportion of overlap between adjacenet tiles during preprocessing.')
	parser.add_argument('--stainNorm', action='store_true', help='Normalize the staining colors')
	parser.add_argument('--softLabel', action='store_true', help='Use soft labeling for target labels (i.e. float between [0,1])')
	# parser.add_argument('--partitionSamples', action='store_true', help='Flag to randomly generate train/validation/test sample partitions. IF NOT INCLUDED, THE COLUMN WITHIN THE METADATA FILE WILL BE USED.')
	parser.add_argument('--trainTestSplit', type=float, default=None, help='Proportion of samples to use for training samples in the train/test partition. Only needed if samples are not already split.')
	parser.add_argument('--checkpointModel', type=str, default=None, help='Path to a pretrained model; either a checkpoint or for transfer learning.')
	parser.add_argument('--batch_size', type=int, default=64, help='How many tiles to include for each mini-batch (default: 64)')
	parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
	parser.add_argument('--max_gpu', default=None, type=int, help='Number of gpus to use (default: 0 - uses all available)')
	parser.add_argument('--max_cpu', default=1, type=int, help='Maximum number of CPUs to utilize for parallelization (default: None - utilizes all available cpus)')
	parser.add_argument('--ensemble', default=5, type=int, help='Number of ensemble models to train.')
	parser.add_argument('--dropoutRate', default=0.2, type=float, help='Rate of dropout to be used within the fully connected layers.')
	parser.add_argument('--maxROI', default=10000, type=int, help='Number of maximum ROIs that can be selected.')
	parser.add_argument('--python', type=str, default='python3', help='Specify the python version..')
	
	parser.add_argument('--preprocess', action='store_true', help='Preprocess, filter, and tile WSI')
	parser.add_argument('--generateDataSets', action='store_true', help='Generate the initial 5x datasets')
	parser.add_argument('--train5x', action='store_true', help='Train a 5x ensemble model.')
	parser.add_argument('--train20x', action='store_true', help='Train a 20x ensemble model.')
	parser.add_argument('--calcFeatures', action='store_true', help='Generate tile feature vectors for each 5x model.')
	parser.add_argument('--pullROIs', action='store_true', help='Pull regions of interest using each 5x model.')
	parser.add_argument('--best5xModels', nargs='+', type=int, default=None, help='Provide a list of best models to use for the 5x training (Use the epoch number; i.e. checkpoint_best_5x_150.pth would be model 150). You should provide 1 value per ensemble model (i.e. ensemble of 5 models should have 5 model numbers. Default will use the final saved checkpoints after training the 5x model.')
	parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')


	args = parser.parse_args()


	max_process_per_gpu = 1

	save_top_tiles=True
	save_data=False

	softLabel = False
	if args.softLabel:
		softLabel = True

	if args.output is None:
		outputPath = os.path.join(args.projectPath, "output")
	else:
		outputPath = args.output
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	tilePath = os.path.join(args.projectPath, "tiles_png")



	############### Set up GPU parallelization if available ###############
	#######################################################################
	maxAvailableGPUs = torch.cuda.device_count()
	if args.max_gpu:
		max_seed = min(args.max_gpu, maxAvailableGPUs)
	else:
		max_seed = 1
	
	if max_seed > args.ensemble:
		max_seed = args.ensemble

	models_parallel = [[] for i in range(max_seed)]
	models_bin = 0
	for iModel in range(args.ensemble):
		if models_bin == max_seed:
			models_bin = 0	
		models_parallel[models_bin].append(iModel)
		models_bin += 1



	if args.best5xModels:
		if len(args.best5xModels) != args.ensemble:
			print("Please provide the same number of best 5x model checkpoints as the number of ensemble models")
			sys.exit()
		bestModels_parallel = [[] for i in range(max_seed)]
		models_bin = 0
		for iModel in args.best5xModels:
			if models_bin == max_seed:
				models_bin = 0	
			bestModels_parallel[models_bin].append(iModel)
			models_bin += 1
	else:
		bestModels_parallel = [[] for i in range(max_seed)]
		models_bin = 0
		for i in range(args.ensemble):
			if models_bin == max_seed:
				models_bin = 0	
			bestModels_parallel[models_bin].append(None)
			models_bin += 1

		# bestModels_parallel =[None for i in range(max_seed)]


	############### Preprocessing components ###############
	########################################################
	if args.preprocess and args.stainNorm:
		if args.preprocess:
			print("\tPreprocessing slides:", flush=True)
			print("\t\tFiltering and tiling image(s)...", end='', flush=True)
			preproc.preprocess_images(args.project, args.projectPath, args.max_cpu, save_top_tiles, save_data, args.tileOverlap)
			print("done")
		if args.stainNorm:
			print("\t\tNormalizing tissue staining...", end='', flush=True)
			normalizeTileStain.multiprocess_stainNorm(tilePath, os.path.join(args.projectPath, "tiles_png_stainNorm"), args.max_cpu)
			print("done", flush=True)

	if args.stainNorm:
		tilePath = os.path.join(args.projectPath, "tiles_png_stainNorm")	
	

	################ Generate Data Structures  ######################
	#################################################################
	if args.generateDataSets:
		if args.trainTestSplit:
			# gatherDataSets.
			pass
		print("\t\tGathering datasets for training...", end='', flush=True)
		gatherDataSets.generateDataStructures(args.project, args.projectPath, args.metadata, tilePath, outputPath, prediction=False, softLabel=softLabel)
		print("done\n", flush=True)



	##################### Train 5x models  ##########################
	#################################################################
	if args.train5x:
		print("\tTraining " + str(args.ensemble) + " models at 5x resolution. You can check progress of each model at " + os.path.join(outputPath, "training_m[model_number]"), flush=True)
		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		for i in range(max_seed):
			pool.apply_async(utilsModel.runMultiGpuTraining, args=(i, models_parallel[i], args.python, outputPath, args.batch_size, args.dropoutRate, "5x", args.workers, args.epochs, args.checkpointModel))
		pool.close()
		pool.join()
		print("\tAll " + str(args.ensemble) + " ensemble 5x models finished training.", flush=True)



	################ Pull Regions of Interest  ######################
	#################################################################
	
	if args.calcFeatures:
		print("\tGenerating tile feature vectors at 5x resolution for " + str(args.ensemble) + " models.", flush=True)
		# Generate feature vectors using best models
		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		results = []
		for i in range(max_seed):
			r = pool.apply_async(utilsModel.generateFeatureVectorsUsingBestModels, args=(i, models_parallel[i], args.project, args.projectPath, args.python, outputPath, args.batch_size, args.dropoutRate, "5x", bestModels_parallel[i], args.checkpointModel))
			results.append(r)
		pool.close()
		pool.join()

		for r in results:
			r.wait()
			if not r.successful():
				# Raises an error when not successful
				r.get()
		print("\tAll " + str(args.ensemble) + " ensemble feature vectors have been generated.", flush=True)

	if args.pullROIs:
		print("\tPulling regions of interest at 5x resolution and sampling each region at 20x resolution for " + str(args.ensemble) + " models.", flush=True)
		# Extract ROIs for each model
		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		results = []
		for i in range(max_seed):
			r = pool.apply_async(utilsModel.runMultiGpuROIs, args=(i, models_parallel[i], args.project, args.projectPath, args.python, outputPath, args.maxROI, args.max_cpu))
		results.append(r)
		pool.close()
		pool.join()

		for r in results:
			r.wait()
			if not r.successful():
				# Raises an error when not successful
				r.get()		

		print("\tAll " + str(args.ensemble) + " ensemble ROIs have been extracted.", flush=True)



	##################### Train 20x models  ##########################
	#################################################################
	if args.train20x:
		print("\tTraining " + str(args.ensemble) + " models at 20x resolution.", flush=True)
		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		for i in range(max_seed):
			pool.apply_async(utilsModel.runMultiGpuTraining, args=(i, models_parallel[i], args.python, outputPath, args.batch_size, args.dropoutRate, "20x", args.workers, args.epochs, args.checkpointModel))
		pool.close()
		pool.join()
		print("\tAll " + str(args.ensemble) + " ensemble 20x models finished training.", flush=True)



	############# Summary of final results ##########################
	#################################################################
	finalAnalysisList = []
	if args.preprocess:
		finalAnalysisList.append("Preprocessing")
	if args.preprocess and args.stainNorm:
		finalAnalysisList.append("Stain normalization")
	if args.generateDataSets:
		finalAnalysisList.append("Generating data structures")
	if args.train5x:
		finalAnalysisList.append("Training 5x ensemble for " + str(args.ensemble) + " models")
	if args.calcFeatures:
		finalAnalysisList.append("Calculating tile feature vectors for " + str(args.ensemble) + " models")
	if args.pullROIs:
		finalAnalysisList.append("Sampling ROIs for " + str(args.ensemble) + " models")
	if args.train20x:
		finalAnalysisList.append("Training 20x ensemble for " + str(args.ensemble) + " models")		
	print("\n===============================================================================", flush=True)			
	print("The current job has completed. The following analyses have finished running: ")	
	print("\n".join(["\t" + x for x in finalAnalysisList]))


if __name__ == "__main__":
	main()


