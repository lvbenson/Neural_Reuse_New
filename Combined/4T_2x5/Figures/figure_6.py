##################################################
# figure 5
# mi analysis
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def all_mi_data():
    plt.figure(figsize=[6, 4])
    # load data
    dir = "./Combined/4T_2x5/Data"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()
    print("Found {} files in {}".format(len(files), dir))
    dat = []
    all_fits = []
    best_fits = []
    count = 0
    for i, file in enumerate(files):
        fits = np.load(file)
        fits = fits**(1/4)
        #if np.prod(fits)**(1/4) > 0.8:
        if np.min(fits)**(1/4) > 0.8:
            run_num = file.split("/")[-1].split(".")[-2].split("_")[-1]
            mi_data = np.load("./Combined/4T_2x5/Data/NormMI_IP_{}.npy".format(run_num))
            #print(mi_data)
            #print(mi_data.shape)
            plt.scatter(np.arange(1, 11), mi_data, c='blue', s=5, alpha=0.7)
            mi_data = np.load("./Combined/4T_2x5/Data/NormMI_CP_{}.npy".format(run_num))
            #print(mi_data)
            plt.scatter(np.arange(1, 11), mi_data, c='green', s=5, alpha=0.7)
            mi_data = np.load("./Combined/4T_2x5/Data/NormMI_LW_{}.npy".format(run_num))
            #print(mi_data)
            plt.scatter(np.arange(1, 11), mi_data, c='red', s=5, alpha=0.7)
            mi_data = np.load("./Combined/4T_2x5/Data/NormMI_MC_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 11), mi_data, c='yellow', s=5, alpha=0.7)

    # plt.plot([0.5,10.5],[0.8,0.8], "k--", alpha=0.7)
    plt.xticks(np.arange(1, 11))

    plt.xlim([0.5, 10.5])
    plt.ylim([-0.03,1.1])
    plt.xlabel("Neuron #")
    plt.ylabel("Contribution estimated by\nMutual Information")
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x5/Figures/figure_6_MI.pdf")
    plt.show()

all_mi_data()


def plot_mi_analysis(run_num):
    plt.figure(figsize=[8, 2.5])

    ### PANEL A
    plt.subplot2grid([2, 3], [0, 0])
    plt.plot([0.5, 10.5], [0.85, 0.85], "k--", alpha=0.7)
    for i in np.arange(1, 10):
        plt.plot([i + 0.5, i + 0.5], [-0.05, 1.1], "gray", alpha=0.5)

    # load and plot data
    mi_data = 1 - np.load("./Combined/4T_2x5/Data/NormMI_IP_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11) - 0.10, mi_data, s=50, alpha=0.7, c="xkcd:tomato", label="IP"
    )
    mi_data = 1 - np.load("./Combined/4T_2x5/Data/NormMI_CP_{}.npy".format(run_num))
    plt.scatter(np.arange(1, 11) - 0.05, mi_data, s=50, alpha=0.7, c="xkcd:azure", label="CP")
    mi_data = 1 - np.load("./Combined/4T_2x5/Data/NormMI_LW_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11),
        mi_data,
        s=50,
        alpha=0.7,
        c="xkcd:teal green",
        label="LW",
    )
    mi_data = 1 - np.load("./Combined/4T_2x5/Data/NormMI_MC_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 11) + 0.05,
        mi_data,
        s=50,
        alpha=0.7,
        c="xkcd:orange",
        label="MC",
    )

    plt.legend()
    plt.xticks(np.arange(1, 11))
    plt.xlim([0.5, 10.5])
    plt.ylim([-0.1, 1.05])
    plt.xlabel("Neuron #")
    plt.ylabel("Contribution estimated using\nMutual Information")
    plt.show()

    ### Prep for rest of tthe panelss
    dir = "./Combined/4T_2x5/Data"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()

    all_categs = []
    all_counts = []

    for i, file in enumerate(files):
        fits = np.load(file)
        # if np.prod(fits) > 0.8:
        fits = fits**(1/4)
        if np.min(fits) > 0.8:
            ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
            ipp = 1 - np.load("./Combined/4T_2x5/Data/NormMI_IP_" + str(ind) + ".npy")
            cpp = 1 - np.load("./Combined/4T_2x5/Data/NormMI_CP_" + str(ind) + ".npy")
            lwp = 1 - np.load("./Combined/4T_2x5/Data/NormMI_LW_" + str(ind) + ".npy")
            mcp = 1 - np.load("./Combined/4T_2x5/Data/NormMI_MC_" + str(ind) + ".npy")

            Threshold = 0.85
            count = np.zeros(12)
            for (ip_neuron, cp_neuron, lw_neuron, mc_neuron) in zip(ipp, cpp, lwp, mcp):
                if (
                    ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold and mc_neuron > Threshold
                ):  # no task neurons
                    count[0] += 1
                if (
                    ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold and mc_neuron > Threshold
                ):  # ip task neurons
                    count[1] += 1
                if (
                    ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold and mc_neuron > Threshold
                ):  # cp task neurons
                    count[2] += 1
                if (
                    ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold and mc_neuron > Threshold
                ):  # lw task neurons
                    count[3] += 1
                if (
                    ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold and mc_neuron <= Threshold
                ):  # mc task neurons
                    count[4] += 1
                if (
                    ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold and mc_neuron > Threshold
                ):  # ip and cp
                    count[5] += 1
                if (
                    ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold and mc_neuron > Threshold
                ):  # ip and lw
                    count[6] += 1
                if (
                    ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold and mc_neuron <= Threshold
                ):  # ip and mc
                    count[7] += 1
                if (
                    ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold and mc_neuron > Threshold
                ):  #cp and lw
                    count[8] += 1
                if (
                    ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold and mc_neuron <= Threshold
                ):  #cp and mc
                    count[9] += 1
                if (
                    ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold and mc_neuron <= Threshold
                ):  #lw and mc
                    count[10] += 1
                if (
                    ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold and mc_neuron <= Threshold
                ):  #all 
                    count[11] += 1

            # making it dataframe ready
            all_counts.append(count)
            categs = ["None","IP","CP","LW","MC","IP+CP","IP+LW","IP+MC","CP+LW","CP+MC","LW+MC","All"]
            #categs = ["None", "IP", "CP", "LW", "IP+CP", "IP+LW", "CP+LW", "All"]
            for cg, ct in zip(categs, count):
                all_categs.append([cg, ct, i])

    # plot specialization and reuse
    plt.figure(figsize=[8, 8])
    ax2 = plt.subplot2grid([1, 3], [0, 2], adjustable="box", aspect=1)
    ax2.plot([-0.5, 11.5], [11.5, -0.5], "k", linewidth=0.7)
    count_data = []
    for count in all_counts:
        # plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
        count_data.append([count[1] + count[2] + count[3] + count[4], np.sum(count[5:])])
    df = pd.DataFrame(
        count_data, columns=["No. of specialized neurons", "No. of reused neurons"]
    )
    ax = sns.swarmplot(
        x="No. of specialized neurons",
        y="No. of reused neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(12)]),
        alpha=0.8,
        s=7,
    )
    plt.xticks(np.arange(12), np.arange(12))
    plt.yticks(np.arange(12), np.arange(12))
    plt.xlim([-0.5, 11.5])
    plt.ylim([-0.5, 11.5])
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x5/Figures/figure_6_reuse.pdf")
    plt.show()

"""
        ### PANEL D
    plt.subplot2grid([2, 3], [1, 0], colspan=3)
    df = pd.DataFrame(
        all_categs, columns=["Category", "Number of Neurons", "network_id"]
    )
    ax = sns.stripplot(x="Category", y="Number of Neurons", hue="network_id", data=df)
    ax.legend_.remove()
"""
    #plt.tight_layout()
    #plt.savefig("./Combined/4T_2x5/Figures/figure_6_reuse.pdf")
    #plt.savefig("figure_6a_mi.png")
    #plt.show()

plot_mi_analysis(0)

"""
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

    ### PANEL C
    plt.subplot2grid([2, 3], [0, 2])
    count_data = []
    for count in all_counts:
        # plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
        count_data.append([count[1] + count[2] + count[3], np.sum(count[4:])])
    df = pd.DataFrame(
        count_data,
        columns=["Number of Specialized Neurons", "Number of Reused Neurons"],
    )
    ax = sns.swarmplot(
        x="Number of Specialized Neurons",
        y="Number of Reused Neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(11)]),
    )
    plt.xticks(np.arange(11), np.arange(11))
    plt.yticks(np.arange(11), np.arange(11))
    # plt.xlim([-0.5,10.5])
    # plt.ylim([-0.5,10.5])
    # plt.xlabel("Number of Specialized Neurons")
    # plt.ylabel("Number of Reused Neurons")

    ### PANEL D
    plt.subplot2grid([2, 3], [1, 0], colspan=3)
    df = pd.DataFrame(
        all_categs, columns=["Category", "Number of Neurons", "network_id"]
    )
    ax = sns.swarmplot(x="Category", y="Number of Neurons", hue="network_id", data=df)
    ax.legend_.remove()

    plt.tight_layout()
    plt.savefig("figure_6a_mi.pdf")
    plt.savefig("figure_6a_mi.png")
    plt.show()
    """