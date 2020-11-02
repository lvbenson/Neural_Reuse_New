##################################################
# figure 2
# gens vs fitnness for multifunctional runs
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def fitness_traces(dir):
    plt.figure(figsize=[4, 3])

    # load data
    files = glob.glob(os.path.join(dir, "best_history*.npy"))
    files.sort()
    print("Found {} files in {}".format(len(files), dir))
    best_final_fits = np.zeros(len(files))
    bfs = []
    for i, file in enumerate(files):
        fits = np.load(file)
        #print(fits)
        #print(fits.shape)
        #print(len(fits))
        fits = fits**(1/4) #elevate for the number of tasks
        #fits = fits[0:1000]
       
        #print(fits)
        bfs.append(fits)
        best_final_fits[i] = fits[-1]
        #print(file, fits[-1])
    print(
        "Number of runs with fitness >= 0.8 = ",
        len(best_final_fits[best_final_fits > 0.8]),
    )

    # get best run from data
    #print(len(bfs))
    bfs = np.array(bfs)
    #print(bfs.shape)
    #print(bfs)
    best_run = np.argmax(bfs[:, -1])
    #print(best_run, bfs[best_run, -1])

    # plot
    ok_label = r"$Fitness \geq 0.8$"
    ko_label = "Fitness < 0.8"
    best_label = "Best"
    for i in range(np.shape(bfs)[0]):
        if bfs[i][-1] < 0.8:
            plt.plot(bfs[i], "tab:olive", label=ko_label)
            ko_label = None
    for i in range(np.shape(bfs)[0]):
        if bfs[i][-1] >= 0.8:
            plt.plot(bfs[i], "sienna", label=ok_label)
            ok_label = None

    # Plotting best
    
    plt.plot(bfs[best_run], "xkcd:reddish", label=best_label)  # "xkcd:ultramarine")
    plt.title("Combined sensors/motors")
    #plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x10/Figures/figure_2_fitnesstraces.pdf")
    plt.show()
    


fitness_traces("./Combined/4T_2x10/Data")
"""
plt.plot(bfs[best_run], "xkcd:reddish", label=best_label)  # "xkcd:ultramarine")
plt.title("Combined sensors/motors")
    #plt.legend()
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.tight_layout()
plt.savefig("./Combined/4T_2x10/Figures/figure_2_fitnesstraces.pdf")
plt.show()
"""