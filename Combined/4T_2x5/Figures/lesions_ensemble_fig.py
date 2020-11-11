#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:06:51 2020

@author: lvbenson
"""


import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

dir = "../2x5"
files = glob.glob(os.path.join(dir, "perf_*.npy"))
files.sort()

all_categs = []
all_counts = []
for i, file in enumerate(files):
    fits = np.load(file)
        # if np.prod(fits) > 0.8:
    if np.min(fits) > 0.0:
        ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
        ipp = np.load("../2x5/lesions_IP_" + str(ind) + ".npy")
        cpp = np.load("../2x5/lesions_CP_" + str(ind) + ".npy")
        lwp = np.load("../2x5/lesions_LW_" + str(ind) + ".npy")

            # Stats on neurons for Ablations
        Threshold = 0.85
        count = np.zeros(8)
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ipp, cpp, lwp):
            if (
                ip_neuron > Threshold
                and cp_neuron > Threshold
                and lw_neuron > Threshold
            ):  # no task neurons
                count[0] += 1
            if (
                ip_neuron <= Threshold
                and cp_neuron > Threshold
                and lw_neuron > Threshold
            ):  # ip task neurons
                    count[1] += 1
            if (
                ip_neuron > Threshold
                and cp_neuron <= Threshold
                and lw_neuron > Threshold
            ):  # cp task neurons
                count[2] += 1
            if (
                ip_neuron > Threshold
                and cp_neuron > Threshold
                and lw_neuron <= Threshold
            ):  # lw task neurons
                count[3] += 1
            if (
                ip_neuron <= Threshold
                and cp_neuron <= Threshold
                and lw_neuron > Threshold
            ):  # ip + cp task neurons
                count[4] += 1
            if (
                ip_neuron <= Threshold
                and cp_neuron > Threshold
                and lw_neuron <= Threshold
            ):  # ip + lw task neurons
                count[5] += 1
            if (
                ip_neuron > Threshold
                and cp_neuron <= Threshold
                and lw_neuron <= Threshold
            ):  # cp + lw task neurons
                count[6] += 1
            if (
                ip_neuron <= Threshold
                and cp_neuron <= Threshold
                and lw_neuron <= Threshold
            ):  # all  task neurons
                count[7] += 1

            # making it dataframe ready
        all_counts.append(count)
        categs = ["None", "IP", "CP", "LW", "IP+CP", "IP+LW", "CP+LW", "All"]
        for cg, ct in zip(categs, count):
            all_categs.append([cg, ct, i])

    # plot specialization and reuse
plt.figure(figsize=[8, 4])
ax2 = plt.subplot2grid([1, 3], [0, 2], adjustable="box", aspect=1)
ax2.plot([-1.5, 20.5], [20.5, -1.5], "k", linewidth=0.7)
count_data = []
for count in all_counts:
    plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
    
    count_data.append([count[1] + count[2] + count[3], np.sum(count[4:])])
df = pd.DataFrame(
        count_data, columns=["No. of specialized neurons", "No. of reused neurons"]
        )
ax = sns.swarmplot(
        x="No. of specialized neurons",
        y="No. of reused neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(21)]),
        alpha=0.8,
        s=7,
    )
plt.xticks(np.arange(21), np.arange(21))
plt.yticks(np.arange(21), np.arange(21))
plt.xlim([-1.5, 20.5])
plt.ylim([-1.5, 20.5])

"""
    # plt.xlabel("Number of Specialized Neurons")
    # plt.ylabel("Number of Reused Neurons")

    # make df and plot
    plt.subplot2grid([1, 3], [1, 0], colspan=3)
    df = pd.DataFrame(
        all_categs, columns=["Category", "No. of Neurons", "network_id"]
    )
    ax = sns.swarmplot(x="Category", y="No. of Neurons", hue="network_id", data=df)
    ax.legend_.remove()
    
"""

plt.tight_layout()
#plt.savefig("figure_5.pdf")
plt.show()
