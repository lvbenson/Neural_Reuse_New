import os
import glob
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
"""
def fitness_traces(dir, color, label):
    files = glob.glob(os.path.join(dir, "best_history*.npy"))
    print("Found {} files in {}".format(len(files), dir))
    bfs = np.zeros(len(files))
    for i, file in enumerate(files):
        bf = np.load(file)
        bf = bf**(1/4)
        bfs[i] = bf[-1]
        print(bfs[i])
        if dir == "./Combined/4T_2x10/Data":
            plt.plot(bf, color=color, label=label)
        else:
            plt.plot(bf, color=color, label=label)
        label = None

    #print("Number of runs > 0.2 = ", len(bfs[bfs >= 0.2]))
    #print("Best of all runs = ", np.max(bfs))
    #print("")


plt.figure(figsize=[4, 3])

#fitness_traces("./Combined/4T_2x10/Data", "xkcd:green", "2x5")
fitness_traces("./Combined/4T_2x5/Data", "xkcd:azure", "2x10")
fitness_traces("./Combined/4T_2x10/Data", "xkcd:green", "2x5")
#fitness_traces("./Combined/4T_2x20/Data", "xkcd:red", "2x20")

plt.legend()
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("fitness traces")
plt.savefig("./Combined/4T_2x10/Figures/figure_2_compare.pdf")
plt.show()
"""


####Distribution plot

def distributions(dir1, dir2, dir3):
    files1 = glob.glob(os.path.join(dir1, "best_history*.npy"))
    print("Found {} files in {}".format(len(files1), dir1))
    files2 = glob.glob(os.path.join(dir2, "best_history*.npy"))
    print("Found {} files in {}".format(len(files2), dir2))
    files3 = glob.glob(os.path.join(dir3, "best_history*.npy"))
    print("Found {} files in {}".format(len(files3), dir3))
    bfs1 = np.zeros(len(files1))
    fits1_list = []
    for i, file in enumerate(files1):
        bf1 = np.load(file)
        bf1 = bf1**(1/4)
        bfs1[i] = bf1[-1]
        fits1_list.append(bfs1[i])
    bfs2 = np.zeros(len(files1))
    fits2_list = []
    for i, file in enumerate(files2):
        bf2 = np.load(file)
        bf2 = bf2**(1/4)
        bfs2[i] = bf2[-1]
        fits2_list.append(bfs2[i])
    bfs3 = np.zeros(len(files3))
    fits3_list = []
    for i, file in enumerate(files3):
        bf3 = np.load(file)
        bf3 = bf3**(1/4)
        bfs3[i] = bf3[-1]
        fits3_list.append(bfs3[i])
    return fits1_list,fits2_list,fits3_list

exp1,exp2,exp3 = distributions("./Combined/4T_2x5/Data", "./Combined/4T_2x10/Data","./Combined/4T_2x20/Data")

colors = ['#E69F00', '#56B4E9','#009E73']
names = ['2x5','2x10','2x20']

plt.hist([exp1,exp2,exp3], bins = int(1/.1), color = colors, label=names)

plt.legend()
plt.xlabel('fitness')
plt.ylabel('# of agents')
plt.title('Fitness distributions')
plt.savefig("./Combined/4T_2x10/Figures/figure_2_distributions.pdf")
plt.show()
    
