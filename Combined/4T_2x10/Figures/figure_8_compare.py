##################################################
# figure 6
# comparing lesion, mi and var analysis
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def dist(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def categorize(ipp, cpp, lwp, mcp, Threshold):
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
    return [count[1] + count[2] + count[3], count[4], np.sum(count[5:])], count


def compare_analyses(Threshold):
    plt.figure(figsize=[8, 3])
    # prep
    plt.subplot(131)
    plt.plot([0, 6.5], [0, 6.5], "k", linewidth=0.7)
    plt.subplot(132)
    plt.plot([0, 23.5], [0, 23.5], "k", linewidth=0.7)
    plt.subplot(133)
    plt.plot([0, 22], [0, 22], "k", linewidth=0.7)

    dir = "./Combined/4T_2x10/Data"
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

            l_ip = np.load("./Combined/4T_2x10/Data/lesions_IP_" + str(ind) + ".npy")
            l_cp = np.load("./Combined/4T_2x10/Data/lesions_CP_" + str(ind) + ".npy")
            l_lw = np.load("./Combined/4T_2x10/Data/lesions_LW_" + str(ind) + ".npy")
            l_mc = np.load("./Combined/4T_2x10/Data/lesions_MC_" + str(ind) + ".npy")
            lesion_data = np.concatenate([l_ip, l_cp, l_lw, l_mc])
            lesion_bin_categs, lesion_categs = categorize(l_ip, l_cp, l_lw, l_mc, Threshold)

            mi_ip = 1 - np.load("./Combined/4T_2x10/Data/NormMI_IP_" + str(ind) + ".npy")
            mi_cp = 1 - np.load("./Combined/4T_2x10/Data/NormMI_CP_" + str(ind) + ".npy")
            mi_lw = 1 - np.load("./Combined/4T_2x10/Data/NormMI_LW_" + str(ind) + ".npy")
            mi_mc = 1 - np.load("./Combined/4T_2x10/Data/NormMI_mc_" + str(ind) + ".npy")
            mi_data = np.concatenate([mi_ip, mi_cp, mi_lw, mi_mc])
            mi_bin_categs, mi_categs = categorize(mi_ip, mi_cp, mi_lw, mi_mc, Threshold)

            v_ip = 1 - np.load("./Combined/4T_2x10/Data/NormVar_IP_" + str(ind) + ".npy")
            v_cp = 1 - np.load("./Combined/4T_2x10/Data/NormVar_CP_" + str(ind) + ".npy")
            v_lw = 1 - np.load("./Combined/4T_2x10/Data/NormVar_LW_" + str(ind) + ".npy")
            v_mc = 1 - np.load("./Combined/4T_2x10/Data/NormVar_MC_" + str(ind) + ".npy")
            v_data = np.concatenate([v_ip, v_cp, v_lw, v_mc])
            var_bin_categs, var_categs = categorize(v_ip, v_cp, v_lw, v_mc, Threshold)

            plt.subplot(131)
            plt.scatter(
                dist(mi_data, lesion_data),
                dist(v_data, lesion_data),
                c="k" if ind == "1" else "C0",
                s=30,
                alpha=1 if ind == "1" else 0.8,
            )
            plt.xlabel("Estimated contribution\nMI vs Lesion")
            plt.ylabel("Estimated contribution\nNeural variability vs Lesion")

            plt.subplot(132)
            plt.scatter(
                dist(mi_bin_categs, lesion_bin_categs),
                dist(var_bin_categs, lesion_bin_categs),
                c="k" if ind == "1" else "C1",
                s=30,
                alpha=1 if ind == "1" else 0.8,
            )
            plt.xlabel("Estimated specialization vs reuse\nMI vs Lesion")
            plt.ylabel("Estimated specialization vs reuse\nNeural variability vs Lesion")

            plt.subplot(133)
            plt.scatter(
                dist(mi_categs, lesion_categs),
                dist(var_categs, lesion_categs),
                c="k" if ind == "1" else "C2",
                s=30,
                alpha=1 if ind == "1" else 0.8,
            )
            plt.xlabel("Estimated reuse categories\nMI vs Lesion")
            plt.ylabel("Estimated reuse categories\nNeural variability vs Lesion")

    """
    plt.subplot(131, aspect=1)
    plt.xlim([0, 3.5])
    plt.ylim([0, 3.5])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated contribution\nMI vs Lesion")
    plt.ylabel("Estimated contribution\nNeural variability vs Lesion")

    plt.subplot(132, aspect=1)
    plt.xlim([0, 13.5])
    plt.ylim([0, 13.5])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated specialization vs reuse\nMI vs Lesion")
    plt.ylabel("Estimated specialization vs reuse\nNeural variability vs Lesion")

    plt.subplot(133, aspect=1)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated reuse categories\nMI vs Lesion")
    plt.ylabel("Estimated reuse categories\nNeural variability vs Lesion")


    """

    plt.tight_layout()
    plt.savefig("./Combined/4T_2x10/Figures/compare_reuse.pdf")
    #plt.savefig("figure_6c.png")
    plt.show()


compare_analyses(0.85)
