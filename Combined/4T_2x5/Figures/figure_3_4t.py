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
    plt.figure(figsize=[6, 4])

    # load data
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()
    print("Found {} files in {}".format(len(files), dir))
    dat = []
    for i, file in enumerate(files):
        fits = np.load(file)
        fits = fits**(1/4)
        #print(np.prod(fits),'prod fits')
        if np.prod(fits) >= 0.8:
            #print('found')
            #print(i)
            
            fits = np.round(fits, decimals=4)
            #print(i, file, fits, np.prod(fits))
            dat.append(np.concatenate([[i], ["IP"], [fits[0]]]))
            dat.append(np.concatenate([[i], ["CP"], [fits[1]]]))
            dat.append(np.concatenate([[i], ["LW"], [fits[2]]]))
            dat.append(np.concatenate([[i], ["MC"], [fits[3]]]))
    #print(dat,'dat')
    # make DataFrame and plot
    df = pd.DataFrame(dat, columns=["id","task", "fitness"])
    ax = sns.stripplot(
        x="task",
        y="fitness",
        hue="id",
        data=df,
    )
    #ax._legend.remove()
    ax.get_legend().remove()
    ax.set_yticks(ax.get_yticks()[::5])
    """
    # plot connecting lines
    x1, y1 = np.array(ax.collections[0].get_offsets()).T
    x2, y2 = np.array(ax.collections[1].get_offsets()).T
    x3, y3 = np.array(ax.collections[2].get_offsets()).T
    x4, y4 = np.array(ax.collections[3].get_offsets()).T
    ind = 0
    for xi, xj, xk, xl, yi, yj, yk, yl in zip(x1, x2, x3, x4, y1, y2, y3, y4):
        if yi == np.max(y1):  # best of the best
            #print("Max == ", yi, yi * yj * yk * yl)
            plt.plot([xi, xj], [yi, yj], "black")
            plt.plot([xj, xk], [yj, yk], "black")
            plt.plot([xk, xl], [yk, yl], "black")
        else:
            plt.plot([xi, xj], [yi, yj], "gray", alpha=0.3)
            plt.plot([xj, xk], [yj, yk], "gray", alpha=0.3)
            plt.plot([xk, xl], [yk, yl], "gray", alpha=0.3)
        ind += 1
    """
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x5/Figures/figure_3_4T_comb.pdf")
    plt.show()


connected_swarms("./Combined/4T_2x5/Data")
