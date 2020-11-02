import os
import glob

import numpy as np
import matplotlib.pyplot as plt

def fitness_traces(dir, color, label):
    files = glob.glob(os.path.join(dir, "best_history*.npy"))
    print("Found {} files in {}".format(len(files), dir))
    bfs = np.zeros(len(files))
    for i, file in enumerate(files):
        bf = np.load(file)
        bf = bf**(1/4)
        bfs[i] = bf[-1]
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
fitness_traces("./Combined/4T_2x20/Data", "xkcd:red", "2x20")

plt.legend()
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("fitness traces")
plt.savefig("./Combined/4T_2x10/Figures/figure_2_compare.pdf")
plt.show()