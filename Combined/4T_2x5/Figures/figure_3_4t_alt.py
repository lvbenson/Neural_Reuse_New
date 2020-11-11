##################################################
# figure 3
# connected swarm of individual fitness of multifunc ensemble
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def connected_swarms(dir):
    plt.figure(figsize=[4.5, 3])

    # load data
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
        # if np.prod(fits) > 0.8:
        #prod_list = []
        if np.prod(fits) >= 0.8:
            count += 1
            fits = np.round(fits, decimals=4)
            all_fits.append(fits)
            #prod_list.append(np.prod(fits))
            
            if "perf_11.npy" in file:
                best_fits.append(["IP", fits[0], i])
                best_fits.append(["CP", fits[1], i])
                best_fits.append(["LW", fits[2], i])
                best_fits.append(["MC", fits[3], i])
            print(i, file, fits, np.prod(fits))
            dat.append(["IP", fits[0], i])
            dat.append(["CP", fits[1], i])
            dat.append(["LW", fits[2], i])
            dat.append(["MC", fits[3], i])
    print("Number of networks under considertaion ", count)
    all_fits = np.array(all_fits)
    print(np.shape(all_fits), np.min(all_fits,0), np.argmin(all_fits, 0), np.max(all_fits, 0), np.max(np.prod(all_fits,1)))
    #print(best_fits)

    # make DataFrame and plot
    df = pd.DataFrame(dat, columns=["Task", "Performance", "network_id"])
    ax = sns.stripplot(
        x="Task",
        y="Performance",
        # hue="network_id",
        data=df,
        alpha=0.8,
        palette={"IP": "xkcd:tomato", "CP": "xkcd:azure", "LW": "xkcd:teal green", "MC": "xkcd:yellow"},
    )
    # ax.legend_.remove()

    # make DataFram and plot best
    df = pd.DataFrame(best_fits, columns=["Task", "Performance", "network_id"])
    #print(df)
    ax = sns.stripplot(
        x="Task",
        y="Performance",
        data=df,
        # palette={"IP": "xkcd:tomato", "CP": "xkcd:azure", "LW": "xkcd:teal green"},
        palette={"IP": "k", "CP": "k", "LW": "k", "MC": "k"},
        marker="s",
    )
    # ax.set_yticks(ax.get_yticks()[::5])

    """
    # plot connecting lines
    x1, y1 = np.array(ax.collections[0].get_offsets()).T
    x2, y2 = np.array(ax.collections[1].get_offsets()).T
    x3, y3 = np.array(ax.collections[2].get_offsets()).T
    for xi, xj, xk, yi, yj, yk in zip(x1, x2, x3, y1, y2, y3):
        if yi == np.max(y1):  # best of the best
            print("Max == ", yi, yi * yj * yk)
            plt.plot([xi, xj], [yi, yj], "black")
            plt.plot([xj, xk], [yj, yk], "black")
        else:
            plt.plot([xi, xj], [yi, yj], "gray", alpha=0.3)
            plt.plot([xj, xk], [yj, yk], "gray", alpha=0.3)
    """

    plt.ylim([0.85, 1.01])
    plt.yticks(np.arange(0.85, 1.01, 0.05))
    plt.title('Best Performance on Ind Tasks: Agent 11')
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x5/Figures/figure_3_4T_new.pdf")
    plt.show()


connected_swarms("./Combined/4T_2x5/Data")