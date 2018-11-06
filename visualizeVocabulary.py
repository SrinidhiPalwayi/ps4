import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import misc
import scipy.misc
from skimage import color
import collections
import os, random
import scipy.io as sio
import pickle



def visualizeVocabulary():
    i = 0
    frames = set()
    while i < 900:
        mat = random.choice(os.listdir("sift/"))
        if mat not in frames:
            i+=1
            frames.add(mat)
    descriptors_sample = None
    #creating one-dimensional sift vector
    print("finished picking files")
    for f in frames:
        print(f)
        mat = sio.loadmat("sift/" + f)
        descriptors = np.array(mat["descriptors"])
        size = descriptors.shape[0]
        sample = random.sample(range(0, size), int(0.70*size))
        subset = descriptors[sample,:]
        if(descriptors_sample is None):
            descriptors_sample = np.array(subset)
        else:
            descriptors_sample = np.vstack((descriptors_sample,subset))
    print(descriptors_sample.shape)
    kmeans_cluster = cluster.KMeans(n_clusters=1100)
    kmeans_cluster.fit_predict(descriptors_sample)
    pickle.dump(kmeans_cluster , open("k_means_model", 'wb'))
    meanColors = kmeans_cluster.cluster_centers_
    print(meanColors.shape)

visualizeVocabulary()