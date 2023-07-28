#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

import numpy as np
import torch
import PIL.Image as Image
import pandas as pd
import os
from scipy.stats import norm
import random
import time
import sys



def runMultiGpuTraining (i, iModels, pythonVersion, outputPath, batch_size, dropoutRate, resolution, workers, epochs, checkpointModel=None):
	for currentModel in iModels:
		if resolution == "5x":
			testCommand = pythonVersion + " base/train_mp.py --train_lib " + os.path.join(outputPath, "trainData.pt") + " --val_lib " + os.path.join(outputPath, "valData.pt") +  " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) +  " --batch_size " + str(batch_size) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) + " --epochs " + str(epochs)
		elif resolution == "20x":
			testCommand = pythonVersion + " base/train_mp.py --train_lib " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1), "trainData20x.pt") + " --val_lib " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1), "valData20x.pt") +  " --output " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1)) +  " --batch_size " + str(batch_size) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) + " --epochs " + str(epochs)
		else:
			print("Resolution " + resolution + " is not currently supported.")
			sys.exit()
		# time.sleep(random.randrange(0, 4))
		if checkpointModel:
			testCommand += " --model " + checkpointModel
		os.system(testCommand)
		torch.cuda.empty_cache()



def runMultiGpuInference (i, iModels, pythonVersion, outputPath, modelPath, batch_size, dropoutRate, resolution, workers, BN_reps):
	for currentModel in iModels:
		if resolution == '5x':
			# Non-dropout inference for extracting features of each tile.
			testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(i+1) + ".pth") +  " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers) 
			testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_" + resolution + ".tsv")
			
			# time.sleep(random.randrange(0, 4))
			os.system(testCommand)
			os.system(testCommand2)
			


			# Additional inference for all BN-reps with the specified dropout rate (default 0.2).
			if dropoutRate > 0:
				testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(i+1) + ".pth") + " --batch_size " + str(batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) 		
				testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions_" + resolution + ".csv")
				os.system(testCommand)
				os.system(testCommand3)
				torch.cuda.empty_cache()	

		else:
			# Non-dropout inference for extracting features of each tile.
			testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i+1), "ROI", "testData20x.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(i+1) + ".pth") +  " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers) 
			testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_" + resolution + ".tsv")
			# time.sleep(random.randrange(0, 4))
			os.system(testCommand)
			os.system(testCommand2)


			# Additional inference for all BN-reps with the specified dropout rate (default 0.2).
			if dropoutRate > 0:
				testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i+1), "ROI", "testData20x.pt")+ " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(i+1) + ".pth") + " --batch_size " + str(batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) 		
				testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions_" + resolution + ".csv")
				os.system(testCommand)
				os.system(testCommand3)
				torch.cuda.empty_cache()	
		


def generateFeatureVectorsUsingBestModels (i, iModels, project, projectPath, pythonVersion, outputPath, batch_size, dropoutRate, resolution, bestModels, checkpointModel=None):
	for l, currentModel in enumerate(iModels):
		modelPath = os.path.join(outputPath, "training_m" + str(currentModel+1))

		# Select best checkpoint or use the model number specified by the user
		existingCheckpointModels = [x for x in os.listdir(modelPath) if ".pth" in x]
		existingCheckpointModelNumbers = [int(x.split("checkpoint_best_5x_")[1].split(".")[0]) for x in existingCheckpointModels if ".pth" in x]
		if bestModels[l] != None:
			bestModel = os.path.join(modelPath, existingCheckpointModels[existingCheckpointModelNumbers.index(bestModels[currentModel])])
		else:
			bestModel = os.path.join(modelPath, existingCheckpointModels[existingCheckpointModelNumbers.index(max(existingCheckpointModelNumbers))])

		# Run Train, Validation, and test data through best checkpoint from above
		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "trainData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_train.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_train.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)

		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "valData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_val.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_val.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)

		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_test.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_test.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)	

		torch.cuda.empty_cache()



def runMultiGpuROIs (i, iModels, project, projectPath, pythonVersion, outputPath, maxROI, max_cpu, predict=False):
	for currentModel in iModels:
		if predict:
			roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " +  os.path.join(outputPath, "m" + str(currentModel+1), "ROI") + " --objectiveFile " + \
						os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath, project) + " --tileConv " + \
						os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_5x.tsv") + \
						" --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu) + " --predict"
		else:
			roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " +  os.path.join(outputPath, "training_20x_m" + str(currentModel+1)) + " --objectiveFile " + \
						os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath, project) + " --tileConv " + \
						os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_test.tsv") + \
						" --train_lib " + os.path.join(outputPath, "trainData.pt") + " --feature_vectors_train " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_train.tsv") + \
						" --val_lib " + os.path.join(outputPath, "valData.pt") + " --feature_vectors_val " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_val.tsv") + \
						" --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu)

		os.system(roiCommand)
		torch.cuda.empty_cache()



def selectBestModel (predictionsPath):
	predictions5x = pd.read_csv(predictionsPath, header=0, index_col=0)
	avgPredictions = predictions5x.loc[:,predictions5x.columns.str.endswith("AverageProb")]
	bestModels = pd.DataFrame(index=predictions5x.index, columns=avgPredictions.columns)

	for sample in predictions5x.index:
		bestModels.loc[sample] = abs(avgPredictions.loc[sample] - float(predictions5x.loc[sample, 'Ensemble-Probability']))

	bestModel = bestModels.mean(axis=0).astype(float).idxmin().split("-")[0]

	return(bestModel)

def collectSampleIndex (file, tileConversionMatrix):
	sample = file.split("/")[-1].split(".")[0]
	sampleIndex = int(tileConversionMatrix.loc[sample].iloc[0])
	if sampleIndex < 10:
		sampleIndex = "00" + str(sampleIndex)
	elif 100 > sampleIndex >= 10:
		sampleIndex = "0" + str(sampleIndex)
	else:
		sampleIndex = str(sampleIndex)
	return(sampleIndex)



def z_test (x, mu, sigma):
	'''
	Performs a z-test for statistical comparisons of simulated and original data.

	Parameters:
			x	->	observed number in original sample (mutation count; int)
		   mu	->	average number observed in simulations (average mutation count; float)
		sigma	->	standard deviation in simulations (float)

	Returns:
		z	->	z-score (float)
		p	->	associated p_value (float)
	'''
	z = (x-mu)/sigma
	p = 2*min(norm.cdf(z), 1-norm.cdf(z))
	return(z, p)


def multiResolution (outputPath, nModels, dropoutRate, threshold):
	mat5x = pd.read_csv(os.path.join(outputPath, "predictions_5x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"), index_col=0, header=0)
	mat20x = pd.read_csv(os.path.join(outputPath, "predictions_20x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"), index_col=0, header=0)
	# samples = list(set(mat5x.index & mat20x.index))
	samples = list(set.intersection(set(mat5x.index), set(mat20x.index)))
	mat5x = mat5x.loc[samples]
	mat20x = mat20x.loc[samples]

	finalMat = pd.DataFrame(columns=["file", "target", "HRD-prediction", "Multi-Res-prediction", "LowerCI", "UpperCI", "p-value"])
	finalMat['file'] = samples
	finalMat = finalMat.set_index('file')
	finalMat['target'] = list(mat20x['target'])
	finalMat['Multi-Res-prediction'] = list((mat5x['Ensemble-Probability'] + mat20x['Ensemble-Probability'])/2)
	finalMat['HRD-prediction'] = (finalMat['Multi-Res-prediction'] > threshold).astype(int)

	ensemblePredictions = pd.concat([mat5x.loc[:,mat5x.columns.str.endswith("Prob")], mat20x.loc[:,mat20x.columns.str.endswith("Prob")]], axis=1)

	finalMat['LowerCI'] = list(ensemblePredictions.quantile(0.025, axis=1))
	finalMat['UpperCI'] = list(ensemblePredictions.quantile(0.975, axis=1))
	for sample in finalMat.index:
		predictions = ensemblePredictions.loc[sample]
		z, p = z_test(threshold, np.mean(predictions), np.std(predictions))
		finalMat.loc[sample, 'p-value'] = p

	finalMat.to_csv(os.path.join(outputPath, "DeepHRD_report_5x_20x_n" + str(nModels) + "_dropout" + str(dropoutRate) + ".csv"))


def combinePredictions (resolution, outputPath, nModels, dropoutRate):

	stdev = 1
	# bestModel = None
	for i in range(nModels):
		# modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions_" + resolution + "_m" + str(i+1) + "_" + str(dropoutRate) + "_temp.csv")
		# modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions.csv")
		modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions_" + resolution + ".csv")
		newPredictions = pd.read_csv(modelPath, header=0, index_col=0)
		if i == 0:
			finalPredictions = newPredictions
			finalPredictions['Ensemble-LowerCI'] = 0
			finalPredictions['Ensemble-UpperCI'] = 0
			finalPredictions = finalPredictions.rename(columns={'probability':'Ensemble-Probability'})

		else:
			finalPredictions['Ensemble-Probability'] += newPredictions['probability']
			finalPredictions = pd.concat([finalPredictions, newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")]], axis=1)
		finalPredictions["m" + str(i+1) + "-AverageProb"] = newPredictions['probability']
		finalPredictions["m" + str(i+1) + "-LowerCI"] = newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")].quantile(0.025, axis=1)
		finalPredictions["m" + str(i+1) + "-UpperCI"] = newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")].quantile(0.975, axis=1)

		
	finalPredictions['Ensemble-Probability'] = finalPredictions['Ensemble-Probability']/nModels
	finalPredictions['Ensemble-LowerCI'] = finalPredictions.loc[:,finalPredictions.columns.str.startswith("BN_rep")].quantile(0.025, axis=1)
	finalPredictions['Ensemble-UpperCI'] = finalPredictions.loc[:,finalPredictions.columns.str.startswith("BN_rep")].quantile(0.975, axis=1)
	finalPredictions = finalPredictions.loc[:,~finalPredictions.columns.str.startswith("BN_rep")]
	finalPredictions.to_csv(os.path.join(outputPath, "predictions_" + resolution + "_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"))



def groupTopKtilesTesting (groups, data,k):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide.

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		k:			[int]	The number of top k tiles to consider for each slide. The default is 1; using 
							only the maximum predicted tile probabilite as the final probability for the entire 
							tissue slide (standard MIL assumption).

	Returns:
		1. The indeces for the top k tiles with hightest probabilites for each tissue slide. If k=1, this will
		   return a single tile index for each slide.
		2. The indeces to directly access the relevant slide indeces for each tile
		3. The probabilities each corresponding top tile

	'''
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-k:] = True
	index[:-k] = groups[k:] != groups[:-k]
	return (list(order[index]), list(groups[index]), list(data[index]))


def groupTopKtiles (groups, data,k=1):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide. 

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		k:			[int]	The number of top k tiles to consider for each slide. The default is 1; using 
							only the maximum predicted tile probabilite as the final probability for the entire 
							tissue slide (standard MIL assumption).

	Returns:
		The indeces for the top k tiles with hightest probabilites for each tissue slide. If k=1, this will
		return a single tile index for each slide.
	'''
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-k:] = True
	index[:-k] = groups[k:] != groups[:-k]
	return (list(order[index]))


def groupTopKtilesProbabilities (groups, data, nmax):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide.

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		nmax:		[int]	The number of tissue slides in the given dataset

	Returns:
		The maximum tile probabilities for each slide. 

	'''
	out = np.empty(nmax)
	out[:] = np.nan
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-1] = True
	index[:-1] = groups[1:] != groups[:-1]
	out[groups[index]] = data[index]
	return (out)



def calculateError (pred,real):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Calculates the error between the predicted and real values.

	Parameters:
		pred:		[list]	The predicted class labels after performing an inference pass over the validation set
		real:		[list]	The target class labels of the validation set

	Returns:
		The error, false positive rate, and false negative rate.
	'''
	real = [1 if x[1] >= 0.5 else 0 for x in real]
	pred = np.array(pred)
	real = np.array(real)
	neq = np.not_equal(pred, real)
	accuracy = float(neq.sum())/pred.shape[0]
	fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
	fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
	return (accuracy, fpr, fnr)



class MILdataset ():

	'''
	Class edited used from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).
	Instantiates a MIL dataset. 

	Attributes
		targets:		[list]	Contains the target value for all tissue slides (i.e. 1 or 0. Alternatively, can be a probability)
		grid:			[list]	Contains all of the tiles across all tissue slides
		slideIDX:		[list]	Contains corresponding slide index value for each tile
		transform:		[Class]	Transform class containing all of the relevant transformations to perform to a given tile
		mode:			[int]	Specifies whether to generate data in the inference (1) or training (2) mode 
	'''

	def __init__(self, libraryfile, transform):
		'''
		Initializes all class atributes.
		'''
		lib = torch.load(libraryfile)

		grid = []
		slideIDX = []
		for i,g in enumerate(lib['tiles']):
			grid.extend(g)
			slideIDX.extend([i]*len(g))

		self.slidenames = lib['slides']
		self.targets = lib['targets']
		self.grid = grid
		self.slideIDX = slideIDX
		self.transform = transform
		self.mode = None

	def modelState (self,mode):
		'''Changes the current mode either to inference or training'''
		self.mode = mode

	def maketraindata (self, idxs):
		'''Generates the training dataset using a list of indeces'''
		self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
		self.t_data = random.sample(self.t_data, len(self.t_data))		

	def __getitem__(self,index):
		'''
		Accesses a tile based upon the preset mode (inference or training) and returns the opened tile image and the target tensor
		'''

		# Inference mode
		if self.mode == 1:
			slideIDX = self.slideIDX[index]
			target = self.targets[slideIDX]
			tile = self.grid[index]
			img = Image.open(tile)
			img = self.transform(img)
			return (img, target)

		# Training mode
		elif self.mode == 2:
			slideIDX, coord, target = self.t_data[index]
			img = Image.open(coord)
			img = self.transform(img)
			return (img, target)

	def __len__(self):
		'''
		Returns the length of the given dataset, whether it's the training or inference sets
		'''
		if self.mode == 1:
			return (len(self.grid))
		elif self.mode == 2:
			return (len(self.t_data))


