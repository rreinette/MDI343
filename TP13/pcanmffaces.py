#  License: BSD
#  -*- coding: utf-8 -*-

#  Authors: Vlad Niculae, Alexandre Gramfort, Slim Essid


from time import time

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import RandomState

from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# -- Prepare data and define utility functions --------------------------------
  
    
    
n_row, n_col = 2, 5
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0, dtype=np.float64)

print("Dataset consists of %d faces" % n_samples)


def plot_gallery(title, images):
    """Plot images as gallery"""
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        comp = comp.reshape(image_shape)
        vmax = comp.max()
        vmin = comp.min()
        dmy = np.nonzero(comp < 0)
        if len(dmy[0]) > 0:
            yz, xz = dmy
        comp[comp < 0] = 0

        plt.imshow(comp, cmap=plt.cm.gray, vmax=vmax, vmin=vmin)

        if len(dmy[0]) > 0:
            plt.plot(xz, yz, 'r,')
            print(len(dmy[0]), "negative-valued pixels")

        plt.xticks(())
        plt.yticks(())

    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

# Plot a sample of the input data
plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# -- Decomposition methods ----------------------------------------------------

# List of the different estimators and whether to center the data

estimators = [
    ('pca', 'Eigenfaces - PCA',
     decomposition.PCA(n_components=n_components, whiten=True),
     True),

    ('nmf', 'Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init=None, tol=1e-6,
		        max_iter=1000),
     False)
]

# -- Transform and classify ---------------------------------------------------

labels = dataset.target
X = faces
X_ = faces_centered

for shortname, name, estimator, center in estimators:
    #if shortname != 'nmf':
     #   continue
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()

    data = X
    if center:
        data = X_

    data = estimator.fit_transform(data)

    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    components_ = estimator.components_

    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

plt.show()


### PARTIE MODIFIÉE
scores_mean_PCA = [] 
scores_mean_NMF = []
for i in range(1,50):
	dataPCA = X_
	dataNMF = X
	target = dataset.target
	PCA = decomposition.PCA(n_components=i)
	NMF = decomposition.NMF(n_components=i)
	dataPCA = PCA.fit_transform(dataPCA)
	dataNMF = NMF.fit_transform(dataNMF)
	LDA_PCA = LinearDiscriminantAnalysis()
	LDA_NMF = LinearDiscriminantAnalysis()
	LDA_PCA.fit(dataPCA,target)
	LDA_NMF.fit(dataNMF,target)
	scores_PCA = cross_val_score(LDA_PCA,dataPCA,target,cv=5)
	scores_NMF = cross_val_score(LDA_NMF,dataNMF,target,cv=5)
	scores_mean_PCA.append(np.mean(scores_PCA))
	scores_mean_NMF.append(np.mean(scores_NMF))
plt.close()	
plt.figure(figsize=(10,8))
plt.plot(np.arange(1,50),scores_mean_PCA,'r',label='PCA')
plt.plot(np.arange(1,50),scores_mean_NMF,'b',label='NMF')
plt.legend(loc='lower right')
plt.title('LDA score after PCA and NMF')
plt.xlabel('n_components')
plt.ylabel('score')
plt.show()	


















