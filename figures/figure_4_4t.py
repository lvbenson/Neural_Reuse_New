##################################################
# figure 4
# perf-map and behavior trials of the best
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def behavior_viz(dir, run_num):
    plt.figure(figsize=[8, 4])

    ## Perf maps
    plt.subplot(241)
    perfmap_file = os.path.join(dir, "perfmap_IP_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 2, 2], [-2.5, 0, 2.5])
    plt.yticks([0, 2, 2], [1, 0, -1])
    # plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("IP \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    plt.subplot(242)
    perfmap_file = os.path.join(dir, "perfmap_CP_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 2, 2], [-0.05, 0, 0.05])
    plt.yticks([0, 2, 2], [0.05, 0, -0.05])
    # plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("CP \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    plt.subplot(243)
    perfmap_file = os.path.join(dir, "perfmap_LW_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    #print(perfmap)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 2, 2], [-0.5, 0, 0.5])
    plt.yticks([0, 2, 2], [1, 0, -1])
    #plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("LW \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))
    
    
    plt.subplot(244)
    perfmap_file = os.path.join(dir, "perfmap_MC_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    #print(perfmap)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 2, 2], [-0.5, 0, 0.5])
    plt.yticks([0, 2, 2], [1, 0, -1])
    plt.colorbar()
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.title("MC \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    
    plt.tight_layout()
    
    


    ## Behaviors
    plt.subplot(245)
    IP_dat = np.load(os.path.join(dir, "theta_traces_IP_{}.npy".format(run_num)))
    for theta_trace in IP_dat:
        plt.plot(np.arange(0, 15, 0.05), theta_trace)
    plt.yticks(np.arange(-360, 361, 180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del IP_dat

    plt.subplot(246)
    CP_dat = np.load(os.path.join(dir, "theta_traces_CP_{}.npy".format(run_num)))[::15]
    for theta_trace in CP_dat:
        plt.plot(theta_trace)
    # plt.yticks(np.arange(-360,361,180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del CP_dat

    plt.subplot(247)
    LW_dat = np.load(os.path.join(dir, "theta_traces_LW_{}.npy".format(run_num)))
    for theta_trace in LW_dat:
        plt.plot(theta_trace)
    # plt.yticks(np.arange(-360,361,180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del LW_dat
    
    plt.subplot(248)
    MC_dat = np.load(os.path.join(dir, "position_traces_MC_{}.npy".format(run_num)))
    for position_trace in MC_dat:
        plt.plot(position_trace)
    plt.box(None)
    plt.ylabel("position")
    plt.xlabel("time")
    del MC_dat
    
    

    plt.tight_layout()
    """
    np.save(
        os.path.join(save_dir, "position_traces_MC_{}.npy".format(run_num)),
        position_traces_MC,
    )
    """
    #plt.savefig(
    #    os.path.join(dir, "figure_4_behav_{}.pdf".format(run_num))
    #)
    plt.savefig("./Combined/Experiments/Comb_4T_2x5_NEW/Figures/figure_4_behav_15.pdf")
    plt.show()


behavior_viz("./Combined/Experiments/Comb_4T_2x5_NEW/Data", 26)
