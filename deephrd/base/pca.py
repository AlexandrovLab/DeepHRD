import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import random




def pcaCalc (features, saveFig, outputPath, slideID, epoch, slideName):
	'''
	Performs principal component analysis on a collection of feature vectors. Specifically, the features are extracted from the 
	penultimate layer of the fully connected layers of the MIL-ResNet model for each tile of a WSI. All tile feature vectors for 
	a single WSI are processed simultaneously.
	'''
	features = pd.DataFrame(features)
	a = list(features[4])
	features.drop(4, axis=1)
	pca_tiles = PCA(n_components=2)
	principalComponents = pca_tiles.fit_transform(features)
	principal_tiles_Df = pd.DataFrame(data = principalComponents
	             , columns = ['principal component 1', 'principal component 2'])


	# Perform Kmeans clustering across a range of solutions using the one with the maximum silhouette score. The minimum number of clusters
	# is 2, while the max is 5.
	sil = []
	kmax = min(5, max(3, len(a)))
	for k in range(2, kmax):
	  kmeans = KMeans(n_clusters = k, n_init='auto').fit(principal_tiles_Df)
	  labels = kmeans.labels_
	  if len(set(labels)) == 1:
	  	return(list(random.sample(np.arange(len(labels)), 10)))
	  elif len(set(labels)) == 2 and len(a) == 2:
	  	return([i for i,x in enumerate(a) if x ==max(a)])
	  sil.append(silhouette_score(principal_tiles_Df, labels, metric = 'euclidean'))
	bestK = [i for i,x in enumerate(sil) if x ==max(sil)][0] + 2
	kmeans = KMeans(init="random", n_clusters=bestK, n_init=10, max_iter=300, random_state=42)
	kmeans.fit(principal_tiles_Df)


	# Locate the cluster/tile with the maximum prediction probability.
	labels = kmeans.labels_
	sil_scores = silhouette_samples(principal_tiles_Df, labels)
	maxLabel = labels[[i for i, x in enumerate(a) if x == max(a)][0]]

	indeces = []
	# Calculate the 95% quantile of the silhouette scores
	npPercentileCutoff = np.percentile(sil_scores, 95)

	# Pull indeces of all tiles that belong to the cluster with the maximum prediction
	# probability and have a silhouette score greater than the 95% quantile cutoff.
	for n, (x,z)  in enumerate(zip(labels, sil_scores)):
		if x == maxLabel and z > npPercentileCutoff:
			indeces.append(n)
	if len(indeces) == 0:
		for n, x  in enumerate(labels):
			if x == maxLabel:
				indeces.append(n)

	# Options to save the PCA results as as plot for each sample. 
	if saveFig:
		labels = ['goldenrod' if x == max(a) else 'darkred' if x > 0.5 else 'teal' for x in a]
		edgecolor = ['darkred' if x > 0.5 else 'teal' for x in a]
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=14)
		plt.xlabel('Principal Component - 1',fontsize=20)
		plt.ylabel('Principal Component - 2',fontsize=20)
		plt.title("Principal Component Analysis",fontsize=20)
		targets = [0]
		colors = ['r']
		for target, color in zip(targets,colors):
		    plt.scatter(principal_tiles_Df['principal component 1']
		               , principal_tiles_Df['principal component 2'], c = a, cmap='hot', s = 50, alpha=0.6, edgecolor=edgecolor)

		if bestK == 2:
			plt.scatter([kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[1][0]],[kmeans.cluster_centers_[0][1],kmeans.cluster_centers_[1][1]], color='grey', s=100, alpha=0.6)
		elif bestK == 3:
			plt.scatter([kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[2][0]], [kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[2][1]], color='grey', s=100, alpha=0.6)
		elif bestK == 4 :
			plt.scatter([kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[3][0]], [kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[2][1], kmeans.cluster_centers_[3][1]], color='grey', s=100, alpha=0.6)
		elif bestK == 5:
			plt.scatter([kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[3][0], kmeans.cluster_centers_[4][0]], [kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[2][1], kmeans.cluster_centers_[3][1], kmeans.cluster_centers_[4][1]], color='grey', s=100, alpha=0.6)

		newPointsX, newPointsY = zip(*[[principal_tiles_Df['principal component 1'][x],principal_tiles_Df['principal component 2'][x]] for x in indeces])
		plt.scatter(newPointsX,newPointsY, alpha=0.6, edgecolor='black', marker="*")
		plt.savefig(outputPath + "slide_" + slideName + "_epoch_" + str(epoch) + ".pdf",  bbox_inches='tight')
		plt.close()
	return(indeces)

def readFeatures (file):
	features = pd.read_csv(file, sep="\t", header=None, usecols=[i for i in range(4, 517)])
	return(features)


