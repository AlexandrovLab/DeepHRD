import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights


class ResNet_dropout(nn.Module):
	def __init__(self, dropoutRate):
		super(ResNet_dropout, self).__init__()

		self.dropoutRate = dropoutRate

		# Import the pretrained ResNet18 architecture
		# self.resnet = torchvision.models.resnet.resnet18(pretrained=True)	
		self.resnet = torchvision.models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

		# Freezes the first 6 layers of the ResNet architecture
		count = 0
		for child in self.resnet.children():
			count += 1
			if count < 7:
				for param in child.parameters():
					param.requires_grad=False

		# Adds dropout into the fully connected layer of the ResNet model and changes the final output to 2 nodes
		self.resnet.fc = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(self.resnet.fc.in_features, 2))


	def forward(self, x):
		'''
		Performs a standard forward pass of the ResNet architecture using the modified fully connected layers

		Parameters:
			x:		[tensor] The input tile image

		Returns:
			The activation values from the final output layer. Softmax is performed subsequently, independent from 
			the model.
		'''

		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)

		x = self.resnet.layer1(x)
		x = self.resnet.layer2(x)
		x = self.resnet.layer3(x)
		x = self.resnet.layer4(x)

		x = self.resnet.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.resnet.fc(x)

		return (x)