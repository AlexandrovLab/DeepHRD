import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import argparse
from model import ResNet_dropout as RNN
from snorkel.classification import cross_entropy_with_probs
import utilsModel as ut


parser = argparse.ArgumentParser(description='Multi-Resolution biomarker classifier training script - 2022')
parser.add_argument('--train_lib', type=str, default='', help='Path to the training data structure. See README for more details on formatting')
parser.add_argument('--val_lib', type=str, default='', help='Path to the validation data structure. See README for more details on formatting')
parser.add_argument('--output', type=str, default='.', help='Path to the output where the checkpoints and training files are saved')
parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
parser.add_argument('--validation_interval', default=1, type=int, help='How often to run inference on the validation set.')
parser.add_argument('--k', default=1, type=int, help='The top k tiles based on predicted model probabilities used as representative features of the training classes for each slide.')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint model to restart a given training session or for transfer learning.')
parser.add_argument('--resolution', type=str, default='5x', help='Current magnification resolution')
parser.add_argument('--dropoutRate', default=0.2, type=float, help='Rate of dropout to be used within the fully connected layers.')
parser.add_argument('--weights', default=0.5, type=float, help='Unbalanced positive class weight.')
parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers.')
parser.add_argument('--gpu', default=0, type=int, help='Gpu device selection.')


best_acc = 0
def main ():
	'''
	Multi-resolution training main function to organize and execute all subsequent commands. This script will train a single model at a single
	resolution. To perform an ensemble stacking, this script will need to be re-run N times for an N-stacked model for each desired resolution.
	This version implements a probabilistic soft-labelling for all weak-labels.	The general worflow of the training follows the 
	Multiple-instance Learning (MIL) framework demonstrated and written by Campanella et al. 2019 (https://www.nature.com/articles/s41591-019-0508-1#Abs1).
	This function has been modified based upon (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019). 

	Parameters:
		Passed via command line arguments (See command "python3 train.py -h" for more details)

	Returns:
		None

	Outputs:
		Checkpoint model files:		Files that contain the weights and biases for checkpoints along the training epochs. A model is saved if
									the current accuracy of the model tested on the validation set is better than the previous best accuracy.
									All checkpoints are kept with the following extension: ([output_path]/checkpoint_best_[EPOCH_NUMBER].pth)

		Convergence statistics:		The training loss for each epoch are saved into [output_path]/convergence.csv for reference along with
									the validation loss, error, false positive rate, and false negative rate after every inference pass, which
									is determined using the --validation_interval argument.

		Loss values	:				The training and validation loss are saved in a separate file [output_path]/loss.csv for each epoch.

	'''


	# Instantiate the user arguments and accuract globally
	global args, best_acc, device
	args = parser.parse_args()

	resolution = args.resolution

	# Creates the output directory if it does not already exist
	if not os.path.exists(args.output):
		os.makedirs(args.output)
		

	# If a GPU is available, set the default device to "cuda", otherwise perform training on the "cpu" device. Assuming a GPU is available
	# move the current training to the desired GPU specified by the user (default is 0).
	gpu_available = torch.cuda.is_available()
	if gpu_available:
		torch.cuda.set_device(args.gpu)
		# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



	# Import the model from the model script. Instantiate model weights using a pretrained model if specified
	model = RNN(args.dropoutRate)
	if args.checkpoint:
		ch = torch.load(args.checkpoint)
		model.load_state_dict(ch['state_dict'])


	# Instantiate the loss function using balanced weights or user-specified weights along with the optimizer (Adam).
	# Moves the loss function and model to the available device (either GPU or CPU)
	model.to(device)
	if args.weights==0.5:
		criterion = nn.CrossEntropyLoss().to(device)
	else:
		w = torch.Tensor([1-args.weights,args.weights])
		criterion = nn.CrossEntropyLoss(w).to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	cudnn.benchmark = True # Input remains the same, so the computation graph remains constant


	# Specifies the normalization values and the desired transformations to apply to the input data. The normalization weights
	# are consistent with the weights used to pretrain the ResNet architectures using the ImageNet database. Transformations 
	# include random horizontal and vertical flips (probability=0.5), random rotations up to 180 degress, random color alterations
	# that affect the brightness with a factor of 0.5 and the contrast between 0.2-1.8. 
	normalize = transforms.Normalize(mean=[0.485,0.406,0.406],std=[0.229,0.224,0.225])
	trans = transforms.Compose([transforms.RandomHorizontalFlip(),
								transforms.RandomVerticalFlip(),
								transforms.RandomRotation(180),
								transforms.ColorJitter(brightness=0.5,contrast=[0.2,1.8], saturation=0, hue=0),
								transforms.ToTensor(), normalize])


	# Loads both the training and validation datasets if present. The transformations from above are applied to both datasets
	train_dset = ut.MILdataset(args.train_lib, trans)
	train_loader = torch.utils.data.DataLoader(
		train_dset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False)
	if args.val_lib:
		val_dset = ut.MILdataset(args.val_lib, trans)
		val_loader = torch.utils.data.DataLoader(
			val_dset,
			batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=False)


	# Creates the convergence output file and prints a header
	with open(os.path.join(args.output,'convergence_' + resolution + '.tsv'), 'w') as out:
		print("\t".join(["epoch", "metric", "value"]), file=out)

	# Creates the loss output file
	with open(os.path.join(args.output,'loss_' + resolution + '.tsv'), 'w') as out:
		print("\t".join(["Train_loss", "Validation_loss"]), file=out)


	# Begin training across the desired number of epochs
	for epoch in range(args.epochs):

		# Set the mode to inference
		train_dset.modelState(1)

		# Run an inference pass over the training dataset
		probs, loss = inference(train_loader, model)

		# Provides the indeces for the k top tiles with the maximum predicted probabilites for each tissue slide
		topk = ut.groupTopKtiles(np.array(train_dset.slideIDX), probs, args.k)

		# Generates a new trainind dataset after each inference pass within each epoch and shuffles the data.
		train_dset.maketraindata(topk)

		# Set the mode to inference and perform a training increment across the model
		train_dset.modelState(2)
		loss = train(epoch, train_loader, model, criterion, optimizer)

		# Save the resulting loss to the output convergence and loss files
		# print("Train: ", str(epoch+1) + "/" + str(args.epochs), "Loss: ", str(loss))
		with open(os.path.join(args.output, 'convergence_' + resolution + '.tsv'), 'a') as out:
			print("\t".join([str(epoch+1), str(loss)]), file=out)

		with open(os.path.join(args.output,'loss_' + resolution + '.tsv'), 'a') as floss:
			print(str(loss), file=floss, end="\t")



		# Perform an inference pass of the validation dataset based on the --validation_interval parameter. The default is every epoch.
		if (epoch+1) % args.validation_interval == 0:
			# Set the mode to inference and perform a pass over the validation set
			val_dset.modelState(1)
			probs, loss = inference(val_loader, model)

			# Collects the maximum tile probability for each tissue slide and generates the final prediction label
			maxs = ut.groupTopKtilesProbabilities(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
			pred = [1 if x >= 0.5 else 0 for x in maxs]

			# Calculates the accuracy, balanced error, false positive rate, and false negative rate and saves the values into the output files
			accuracy,fpr,fnr = ut.calculateError(pred, val_dset.targets)
			err = (fpr+fnr)/2.
			# print("Validation: ", str(epoch+1) + "/" + str(args.epochs), "Val_loss: ", str(loss))
			with open(os.path.join(args.output, 'convergence_' + resolution + '.tsv'), 'a') as out:
				print("\t".join([str(epoch+1), "Accuracy: ", str(accuracy)]), file=out)
				print("\t".join([str(epoch+1), "Error: ",  str(err)]), file=out)
				print("\t".join([str(epoch+1), "ValLoss: ", str(loss)]), file=out)

			with open(os.path.join(args.output,'loss_' + resolution + '.tsv'), 'a') as floss:
				print(str(loss), file=floss)

			# If the results from the current inference pass of the validation set are better than the previous best model
			# then save the current model weights and biases as a new checkpoint. This currently uses the balanced error 
			# calculated on the validation cohort.
			if 1-err > best_acc:
				best_acc = 1-err
				obj = {
					'epoch': epoch+1,
					'state_dict': model.state_dict(),
					'best_acc': best_acc,
					'optimizer' : optimizer.state_dict()
				}
				torch.save(obj, os.path.join(args.output,'checkpoint_best_' + resolution + "_" + str(epoch+1) + '.pth'))
				with open(os.path.join(args.output, 'convergence_' + resolution + '.tsv'), 'a') as out:
					print("Model saved (" + str(epoch+1) + ")", file=out)



def inference (loader, model):
	'''
	Performs an inference pass over the complete dataset.

	Parameters:
		loader:		[iterable over a given dataset] The specified dataset that is loaded in as a PyTorch dataloader object.
		model:		[PyTorch model] The current training model.

	Returns:
		Probabilities for each tile after passing through the model
		Loss aggregated across the dataset.
	'''
	model.eval()
	probs = torch.FloatTensor(len(loader.dataset))
	running_loss = 0.
	with torch.no_grad():
		for i, (input, target) in enumerate(loader):
			input = input.to(device)
			target = target.to(device)
			output = F.softmax(model(input), dim=1)
			loss = cross_entropy_with_probs(output, target)
			running_loss += loss.item()*input.size(0)
			probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()

	return (probs.cpu().numpy(), running_loss/len(loader.dataset))



def train (run, loader, model, criterion, optimizer):
	'''
	Performs an training pass over the training dataset, which changes after every epoch based on the inference pass
	over the complete training dataset.

	Parameters:
		run:			[integer] The current training epoch.
		loader:			[iterable over a given dataset] The specified dataset that is loaded in as a PyTorch dataloader object.
		model:			[PyTorch model] The current training model.
		criterion:		Loss function (cross entropy loss with probabilities)
		optimizer:		Optimizer function (Adam optimizer)

	Returns:
		Loss aggregated across the dataset.
	'''

	model.train()
	running_loss = 0.
	for i, (input, target) in enumerate(loader):
		input = input.to(device)
		target = target.to(device)
		output = model(input)
		loss = cross_entropy_with_probs(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item()*input.size(0)
	return (running_loss/len(loader.dataset))




if __name__ == '__main__':
	main()
