##################################################
# figure 5
# Lesion analysis
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def all_lesion_data():
    plt.figure(figsize=[4, 3])
    # load data
    dir = "../New"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()
    print("Found {} files in {}".format(len(files), dir))
    dat = []
    all_fits = []
    best_fits = []
    count = 0
    for i, file in enumerate(files):
        fits = np.load(file)
        # if np.prod(fits) > 0.8:
        if np.min(fits) > 0.85:
            run_num = file.split("/")[-1].split(".")[-2].split("_")[-1]
            lesion_data = np.load("../New/lesions_IP_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 11), lesion_data, s=5, alpha=0.7)
            lesion_data = np.load("../New/lesions_CP_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 11), lesion_data, s=5, alpha=0.7)
            lesion_data = np.load("../New/lesions_LW_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 11), lesion_data, s=5, alpha=0.7)

    # plt.plot([0.5,10.5],[0.8,0.8], "k--", alpha=0.7)
    plt.xticks(np.arange(1, 11))

    plt.xlim([0.5, 10.5])
    plt.xlabel("Neuron #")
    plt.ylabel("Fitness after lesion")
    plt.tight_layout()
    plt.show()

all_lesion_data()


def plot_lesion_analysis(run_num):
    plt.figure(figsize=[8, 2.5])

    plt.subplot2grid([1, 3], [0, 0])
    plt.plot([0.5, 20.5], [0.85, 0.85], "k--", alpha=0.7)
    for i in np.arange(1, 10):
        plt.plot([i + 0.5, i + 0.5], [-0.05, 1.1], "gray", alpha=0.5)

    lesion_data = np.load("../New/lesions_IP_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11) - 0.05,
        lesion_data,
        s=50,
        alpha=0.7,
        c="xkcd:tomato",
        label="IP",
    )
    lesion_data = np.load("../New/lesions_CP_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11), lesion_data, s=50, alpha=0.7, c="xkcd:azure", label="CP"
    )
    lesion_data = np.load("../New/lesions_LW_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11) + 0.05,
        lesion_data,
        s=50,
        alpha=0.7,
        c="xkcd:teal green",
        label="LW",
    )

    plt.legend()
    plt.xticks(np.arange(1, 11))
    plt.xlim([0.5, 10.5])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Neuron #")
    plt.ylabel("Fitness after lesion")

    dir = "../New"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()

    all_categs = []
    all_counts = []

    for i, file in enumerate(files):
        fits = np.load(file)
        # if np.prod(fits) > 0.8:
        if np.min(fits) > 0.85:
            ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
            ipp = np.load("../New/lesions_IP_" + str(ind) + ".npy")
            cpp = np.load("../New/lesions_CP_" + str(ind) + ".npy")
            lwp = np.load("../New/lesions_LW_" + str(ind) + ".npy")

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
    plt.figure(figsize=[8, 8])
    ax2 = plt.subplot2grid([1, 3], [0, 2], adjustable="box", aspect=1)
    ax2.plot([-0.5, 10.5], [10.5, -0.5], "k", linewidth=0.7)
    count_data = []
    for count in all_counts:
        # plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
        count_data.append([count[1] + count[2] + count[3], np.sum(count[4:])])
    df = pd.DataFrame(
        count_data, columns=["No. of specialized neurons", "No. of reused neurons"]
    )
    ax = sns.swarmplot(
        x="No. of specialized neurons",
        y="No. of reused neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(11)]),
        alpha=0.8,
        s=7,
    )
    plt.xticks(np.arange(11), np.arange(11))
    plt.yticks(np.arange(11), np.arange(11))
    plt.xlim([-0.5, 10.5])
    plt.ylim([-0.5, 10.5])
    # plt.xlabel("Number of Specialized Neurons")
    # plt.ylabel("Number of Reused Neurons")

    # make df and plot
    """plt.subplot2grid([1, 3], [1, 0], colspan=3)
    df = pd.DataFrame(
        all_categs, columns=["Category", "No. of Neurons", "network_id"]
    )
    ax = sns.swarmplot(x="Category", y="No. of Neurons", hue="network_id", data=df)
    ax.legend_.remove()"""

    plt.tight_layout()
    plt.show()


plot_lesion_analysis(65)
