#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

# This code represents the main script for the DeepHRD prediction package.
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



# python3 DeepHRD_predict.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/DeepHRD_tissue/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/DeepHRD_tissue/output/ --stainNorm --model /restricted/alexandrov-ddn/users/ebergstr/histology/scripts/final_code/models/breast_ffpe/ --workers 16 --BN_reps 1 --reportVerbose
# python3 DeepHRD_predict.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/output/ --max_gpu 4 --ensemble 4 --dropoutRate 0.2 --python python3 --metadata /restricted/alexandrov-ddn/users/ebergstr/histology/data/BRCA_HRD_DeepHRD_flash_frozen_meta.txt --stainNorm --generateDataSets --BN_reps 10 --modelType breast_flash_frozen --model /restricted/alexandrov-ddn/users/ebergstr/histology/scripts/final_code/models/breast_flash_frozen/ --predict5x --pullROIs --predict20x --predictionMasks
# python3 DeepHRD_predict.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/output_custom/ --max_gpu 2 --ensemble 2 --dropoutRate 0.2 --python python3 --metadata /restricted/alexandrov-ddn/users/ebergstr/histology/data/BRCA_HRD_DeepHRD_flash_frozen_meta.txt --stainNorm --BN_reps 1 --generateDataSets --modelType breast_flash_frozen --model /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/test/output_test/models/ --predict5x --pullROIs --predict20x --predictionMasks --customThreshold 0.5
# python3 DeepHRD_predict.py --projectPath /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/DeepHRD_tissue/ --project BRCA --output /restricted/alexandrov-ddn/users/ebergstr/histology/BRCA/TCGA/DeepHRD_tissue/output/ --stainNorm --model /restricted/alexandrov-ddn/users/ebergstr/histology/scripts/final_code/models/breast_flash_frozen/ --workers 16 --BN_reps 2 --reportVerbose --max_gpu 2 --ensemble 2 --modelType breast_flash_frozen

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

	parser = argparse.ArgumentParser(description='Multi-Resolution biomarker classifier prediction - 2022')
	parser.add_argument('--projectPath', type=str, default='', help='Path to the project directory')
	parser.add_argument('--project', type=str, default='BRCA', help='Project Name where the slides are located. projectPath + project should be the location to the slides.')
	parser.add_argument('--output', type=str, default=None, help='Path to the output and predictions are saved. Recommended projectPath + "output/')
	parser.add_argument('--metadata', type=str, default='', help='Path to the metadata file that contains the labels for each sample')
	parser.add_argument('--model', type=str, default='models/breast_ffpe', help='Path to the pretrained models')
	parser.add_argument('--modelType', type=str, default='breast_ffpe', help='Specify the trained model for testing.')

	parser.add_argument('--tileOverlap', default=0.0, type=float, help='The proportion of overlap between adjacenet tiles during preprocessing.')
	parser.add_argument('--stainNorm', action='store_true', help='Normalize the staining colors')
	parser.add_argument('--batch_size', type=int, default=64, help='How many tiles to include for each mini-batch (default: 64)')
	parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
	parser.add_argument('--BN_reps', type=int, default=10, help='Number of MonteCarlo iterations to perform for bayesian network estimation (default is 10: sufficient for a dropout<0.2)')
	parser.add_argument('--max_gpu', default=0, type=int, help='Number of gpus to use (default: 0 - uses all available)')
	parser.add_argument('--max_cpu', default=1, type=int, help='Maximum number of CPUs to utilize for parallelization (default: None - utilizes all available cpus)')
	parser.add_argument('--ensemble', default=5, type=int, help='Number of ensemble models to test')
	parser.add_argument('--dropoutRate', default=0.2, type=float, help='Rate of dropout to be used within the fully connected layers.')
	parser.add_argument('--maxROI', default=10000, type=int, help='Number of maximum ROIs that can be selected.')
	parser.add_argument('--reportVerbose', action='store_true', help='Print final report to standard out once complete.')
	parser.add_argument('--python', type=str, default='python3', help='Specify the python version.')

	parser.add_argument('--preprocess', action='store_true', help='Preprocess, filter, and tile WSI')
	parser.add_argument('--generateDataSets', action='store_true', help='Generate the initial 5x datasets')
	parser.add_argument('--predict5x', action='store_true', help='Run inference for a 5x ensemble model.')
	parser.add_argument('--pullROIs', action='store_true', help='Pull regions of interest using each 5x model.')
	parser.add_argument('--predict20x', action='store_true', help='Run inference for a 20x ensemble model.')
	parser.add_argument('--predictionMasks', action='store_true', help='Generate prediction masks for each tissue sample.')
	parser.add_argument('--customThreshold', default=None, type=float, help='Threshold to use if running inference on custom models.')


	args = parser.parse_args()

	if not args.preprocess and not args.generateDataSets and not args.predict5x and not args.pullROIs and not args.predict20x:
		preprocess = True
		generateDataSets = True
		predict5x = True
		pullROIs = True
		predict20x = True
	else:
		if args.preprocess:
			preprocess = True
		if args.generateDataSets:
			generateDataSets = True
		if args.predict5x:
			predict5x = True
		if args.pullROIs:
			pullROIs = True
		if args.predict20x:
			predict20x = True		


	max_process_per_gpu = 1

	save_top_tiles=True
	save_data=False

	thresholds = {'breast_ffpe': 0.423, 'breast_flash_frozen': 0.433, 'ovarian_flash_frozen': 0.469}
	threshold = thresholds[args.modelType]
	if args.customThreshold:
		threshold = args.customThreshold

	if args.output is None:
		outputPath = os.path.join(args.projectPath, "output")
	else:
		outputPath = args.output
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	tilePath = os.path.join(args.projectPath, "tiles_png")

	BN_reps = args.BN_reps
	if args.dropoutRate == 0.0:
		BN_reps = 1



	# ############### Preprocessing components ###############
	# ########################################################
	if preprocess or args.stainNorm:
		print("\tPreprocessing slides:", flush=True)

		if preprocess:
			print("\t\tFiltering and tiling image(s)...", end='', flush=True)
			preproc.preprocess_images(args.project, args.projectPath, args.max_cpu, save_top_tiles, save_data, args.tileOverlap)
			print("done")
		if args.stainNorm:
			print("\t\tNormalizing tissue staining...", end='', flush=True)
			normalizeTileStain.multiprocess_stainNorm(tilePath, os.path.join(args.projectPath, "tiles_png_stainNorm"), args.max_cpu)
			print("done", flush=True)

	if args.stainNorm:
		tilePath = os.path.join(args.projectPath, "tiles_png_stainNorm")

	if generateDataSets:
		print("\t\tGathering datasets for inference...", end='', flush=True)
		gatherDataSets.generateDataStructures(args.project, args.projectPath, args.metadata, tilePath, outputPath, True)
		print("done", flush=True)



	###### Set up GPU parallelization if available #########
	########################################################
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



	############### Inference at 5x  #######################
	########################################################
	if predict5x:
		print("\n\tGenerating inference for " + str(args.ensemble) + " models at 5x resolution.", flush=True)

		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		results = []
		for i in range(max_seed):
			r = pool.apply_async(utilsModel.runMultiGpuInference, args=(i, models_parallel[i], args.python, outputPath, args.model, args.batch_size, args.dropoutRate, "5x", args.workers, args.BN_reps))
			results.append(r)
		pool.close()
		pool.join()
		for r in results:
			r.wait()
			if not r.successful():
				# Raises an error when not successful
				r.get()

		print("\tAll " + str(args.ensemble) + " ensemble 5x models finished training.", flush=True)

		sys.stdout.write('\r' + ' '*50)
		sys.stdout.flush()
		utilsModel.combinePredictions("5x", outputPath, args.ensemble, args.dropoutRate)

	if args.max_cpu == None:
		max_cpu = 0



	###################### ROIs ###########################
	#######################################################
	if pullROIs:
		print("\n\tAutomatically selecting Regions of Interest (ROIs)...", end='', flush=True)
		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		results = []
		for i in range(max_seed):
			r = pool.apply_async(utilsModel.runMultiGpuROIs, args=(i, models_parallel[i], args.project, args.projectPath, args.python, outputPath, args.maxROI, args.max_cpu, True))
		results.append(r)
		pool.close()
		pool.join()

		for r in results:
			r.wait()
			if not r.successful():
				# Raises an error when not successful
				r.get()		

		print("done", flush=True)



	############## Inference at 20x  #######################
	#######################################################
	if predict20x:
		print("\n\tPerforming ensemble inference at 20x magnification:", flush=True)

		pool = multiprocessing.Pool(max_seed * max_process_per_gpu)
		results = []
		for i in range(max_seed):
			r = pool.apply_async(utilsModel.runMultiGpuInference, args=(i, models_parallel[i], args.python, outputPath, args.model, args.batch_size, args.dropoutRate, "20x", args.workers, args.BN_reps))
			results.append(r)
		pool.close()
		pool.join()
		for r in results:
			r.wait()
			if not r.successful():
				# Raises an error when not successful
				r.get()
	
		sys.stdout.write('\r' + ' '*50)
		sys.stdout.flush()
		utilsModel.combinePredictions("20x", outputPath, args.ensemble, args.dropoutRate)



	################# Final Reports  #######################
	########################################################
	print("\n\tFinalizing results:", flush=True)
	print("\t\tCalculating multi-resolution prediction...", end='', flush=True)
	utilsModel.multiResolution(outputPath, args.ensemble, args.dropoutRate, threshold)
	print("done", flush=True)

	if args.predictionMasks:
		print("\t\tGenerating prediction masks for each sample...", end='', flush=True)
		if os.path.exists(os.path.join(outputPath, "probability_masks")):
			shutil.rmtree(os.path.join(outputPath, "probability_masks"))
		os.makedirs(os.path.join(outputPath, "probability_masks"))
		bestModel = utilsModel.selectBestModel(os.path.join(outputPath, "predictions_5x_n" + str(args.ensemble) + "_models_" + str(args.dropoutRate) + ".csv"))
		plotProbabilityMasks.multiprocess_plotMasks(tilePath, os.path.join(outputPath, bestModel, "ROI"), os.path.join(outputPath, bestModel, "feature_vectors_test_5x.tsv"), \
											os.path.join(outputPath, bestModel, "feature_vectors_test_20x.tsv"), os.path.join(args.projectPath, "objectiveInfo.txt"), \
											os.path.join(args.projectPath, "slideNumberToSampleName.txt"), outputPath, args.max_cpu)
		print("done", flush=True)

	if args.reportVerbose:
		print("\n\n")
		print("\t\t+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("\t\tSummary Report:")
		print("\t\t\t\t\t\tSample\t\t  DeepHRD-Prediction\tProbability\t\t    CI-95\t\t p-value\n")
		finalResults = pd.read_csv(os.path.join(outputPath, "DeepHRD_report_5x_20x_n" + str(args.ensemble) + "_dropout" + str(args.dropoutRate) + ".csv"), header=0, index_col=0)
		for sample in finalResults.index:
			if finalResults.loc[sample, 'HRD-prediction'] == 0:
				classification = 'HRP'
			else:
				classification = 'HRD'
			if finalResults.loc[sample, 'p-value'] > 0.01:
				classification = "Inconc."

			print("\t\t".join(["\t\t\t", sample.split("/")[-1].split(".")[0], classification, "  " + str(round(finalResults.loc[sample, 'Multi-Res-prediction'], 4)), "[" +str(round(finalResults.loc[sample, 'LowerCI'], 3)) + "-" + str(round(finalResults.loc[sample, 'UpperCI'], 3)) + "]", '{:.2E}'.format(finalResults.loc[sample, 'p-value'])]))
		print("\n\n")

	print("\t\tComplete final reports can be found under: ", outputPath)
	print("\t\t+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

	print("\n\n\n-----------------------------------------------", flush=True)

	print("Analysis complete. Thank you for using DeepHRD!\n", flush=True)


if __name__ == "__main__":
	main()