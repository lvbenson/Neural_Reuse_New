import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns



dir = "./Combined/4T_2x5/Data"
files = glob.glob(os.path.join(dir, "perf_*.npy"))
files.sort()

#ip_all = []
X = []
for i, file in enumerate(files):
    fits = np.load(file)
    # if np.prod(fits) > 0.8:
    fits = fits**(1/4)
    if np.min(fits) > 0.8:
        ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
        ipp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_IP_" + str(ind) + ".npy") #size 10, 1 for each neuron
        sample = [(ipp[0], i),(ipp[1], i),(ipp[2], i),(ipp[3], i),(ipp[4], i),(ipp[5], i),(ipp[6], i),(ipp[7], i),(ipp[8], i),(ipp[9], i)]
        #print(sample)
        #ip_all.append(sample)
        X.append(sample)
        cpp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_CP_" + str(ind) + ".npy")
        lwp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_LW_" + str(ind) + ".npy")
        mcp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_MC_" + str(ind) + ".npy")

X = [item for sublist in X for item in sublist]
print(X[10])

#print(X)
#print(len(X))
#X = np.array(X_)
#X = np.insert(1,0)
#print(X)

from sklearn.cluster import KMeans
"""
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(ip_all)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
"""
#number of clusters: 4 or 5

km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
print(y_km)

plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    X[y_km == 3, 0], X[y_km == 3, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
