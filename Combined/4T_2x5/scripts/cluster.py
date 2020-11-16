

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
        ipp = np.load("./Combined/4T_2x5/Data/lesions_IP_" + str(ind) + ".npy") #size 10, 1 for each neuron
        cpp = np.load("./Combined/4T_2x5/Data/lesions_CP_" + str(ind) + ".npy")
        lwp = np.load("./Combined/4T_2x5/Data/lesions_LW_" + str(ind) + ".npy")
        mcp = np.load("./Combined/4T_2x5/Data/lesions_MC_" + str(ind) + ".npy")
        #reuse as second feature
        reused_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")
        if i <= len(reused_2x5):
            lesion_total = []
            for ip,cp,lw,mc in zip (ipp,cpp,lwp,mcp):
                add = ip+cp+lw+mc
                lesion_total.append(add)
            lesion_total = lesion_total/max(lesion_total)
            sample = (lesion_total[0], reused_2x5[i-1]),(lesion_total[1], reused_2x5[i-1]),(lesion_total[2], reused_2x5[i-1]),(lesion_total[3], reused_2x5[i-1]),(lesion_total[4], reused_2x5[i-1]),(lesion_total[5], reused_2x5[i-1]),(lesion_total[6], reused_2x5[i-1]),(lesion_total[7], reused_2x5[i-1]),(lesion_total[8], reused_2x5[i-1]),(lesion_total[9], reused_2x5[i-1])
            #print(sample)
            X.append(sample)
#print(len(X))


#X = [item for sublist in X for item in sublist]
X = [y for x in X for y in x]
X = np.array(X)

from sklearn.cluster import KMeans
"""
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
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
#print(y_km)

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
    s=50, c='yellow',
    marker='v', edgecolor='black',
    label='cluster 4'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1,loc=(1.05,1.0))
plt.ylabel('Prop Reused')
plt.xlabel('Neural Contribution')
plt.grid()
plt.show()
