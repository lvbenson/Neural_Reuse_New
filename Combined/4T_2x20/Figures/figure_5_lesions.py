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

"""
def all_lesion_data():
    plt.figure(figsize=[6, 4])
    # load data
    #dir = "../New"
    dir = "./Combined/4T_2x10/Data"
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
        if np.min(fits) > 0.8:
            run_num = file.split("/")[-1].split(".")[-2].split("_")[-1]
            #lesion_data = np.load("../New/lesions_IP_{}.npy".format(run_num))
            lesion_data = np.load("./Combined/4T_2x10/Data/lesions_IP_{}.npy".format(run_num))
            #print(lesion_data.shape)
            plt.scatter(np.arange(1, 21), lesion_data, c='blue', s=5, alpha=0.7, label = 'IP' )
            lesion_data = np.load("./Combined/4T_2x10/Data/lesions_CP_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 21), lesion_data, c='green',s=5, alpha=0.7, label='CP')
            lesion_data = np.load("./Combined/4T_2x10/Data/lesions_LW_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 21), lesion_data, c='red',s=5, alpha=0.7,label='LW')
            lesion_data = np.load("./Combined/4T_2x10/Data/lesions_MC_{}.npy".format(run_num))
            plt.scatter(np.arange(1, 21), lesion_data, c='yellow',s=5, alpha=0.7,label='MC')

    # plt.plot([0.5,10.5],[0.8,0.8], "k--", alpha=0.7)
    plt.xticks(np.arange(1, 11))

    plt.xlim([0.5, 20.5])
    plt.ylim([-0.03,1.1])
    plt.xlabel("Neuron #")
    plt.ylabel("Fitness after lesion")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./Combined/4T_2x10/Figures/figure_5_lesions.pdf")
    plt.show()

all_lesion_data()
"""

def plot_lesion_analysis(run_num):
    """
    plt.figure(figsize=[8, 2.5])

    plt.subplot2grid([1, 3], [0, 0])
    plt.plot([0.5, 20.5], [0.85, 0.85], "k--", alpha=0.7)
    for i in np.arange(1, 10):
        plt.plot([i + 0.5, i + 0.5], [-0.05, 1.1], "gray", alpha=0.5)

    lesion_data = np.load("./Combined/4T_2x10/Data/lesions_IP_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 21) - 0.1,
        lesion_data,
        s=50,
        alpha=0.7,
        c="xkcd:azure",
        label="IP",
    )
    lesion_data = np.load("./Combined/4T_2x10/Data/lesions_CP_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 21)-0.05, lesion_data, s=50, alpha=0.7, c="xkcd:teal green", label="CP"
    )
    lesion_data = np.load("./Combined/4T_2x10/Data/lesions_LW_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 21),
        lesion_data,
        s=50,
        alpha=0.7,
        c="xkcd:tomato",
        label="LW",
    )
    lesion_data = np.load("./Combined/4T_2x10/Data/lesions_MC_{}.npy".format(run_num))
    plt.scatter(
        np.arange(1, 21) + 0.05,
        lesion_data,
        s=50,
        alpha=0.7,
        c="xkcd:yellow",
        label="MC",
    )

    plt.legend()
    plt.xticks(np.arange(1, 21))
    plt.xlim([0.5, 20.5])
    plt.ylim([-0.1, 1.05])
    plt.xlabel("Neuron #")
    plt.ylabel("Fitness after lesion")
    """
    dir = "./Combined/4T_2x20/Data/"
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
            ipp = np.load("./Combined/4T_2x20/Data/lesions_IP_" + str(ind) + ".npy")
            cpp = np.load("./Combined/4T_2x20/Data/lesions_CP_" + str(ind) + ".npy")
            lwp = np.load("./Combined/4T_2x20/Data/lesions_LW_" + str(ind) + ".npy")
            mcp = np.load("./Combined/4T_2x20/Data/lesions_LW_" + str(ind) + ".npy")

            # Stats on neurons for Ablations
            Threshold = 0.85
            count = np.zeros(16)
            for (ip_neuron, cp_neuron, lw_neuron, mc_neuron) in zip(ipp, cpp, lwp, mcp):
                if (
                    ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold and mc_neuron > Threshold
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
                    ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold and mc_neuron > Threshold
                ): #ip, cp, lw
                    count[11] += 1
                if ( 
                    ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold and mc_neuron <= Threshold
                ): #ip, cp, mc
                    count[12] += 1
                if (
                    ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold and mc_neuron <= Threshold
                ): #ip, lw, mc
                    count[13] += 1
                if (
                    ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold and mc_neuron <= Threshold
                ): #cp, lw, mc
                    count[14] += 1
                if (
                    ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold and mc_neuron <= Threshold
                ):  #all 
                    count[15] += 1
                

            # making it dataframe ready
            all_counts.append(count) #count is a 1x15 array for each agent. All_counts is 15xensemble size 
            categs = ["None","IP","CP","LW","MC","IP+CP","IP+LW","IP+MC","CP+LW","CP+MC","LW+MC","IP+CP+LW","IP+CP+MC","IP+LW+MC","CP+LW+MC","All"]
            #categs = ["None", "IP", "CP", "LW", "IP+CP", "IP+LW", "CP+LW", "All"]
            for cg, ct in zip(categs, count): #15 categories, 15 slots in count, all_categs keeps track of categories for each agent
                all_categs.append([cg, ct, i])
    #print(all_counts)
    #Pairwise data
    #ip_inv = []
    #cp_inv = []
    #lw_inv = []
    mc_inv = []
    #ip_cp = []
    #ip_lw = []
    ip_mc = []
    #ip_mc_lab = []
    #cp_lw = []
    cp_mc = []
    #cp_mc_lab = []
    lw_mc = []
    ip_cp_mc = []
    ip_lw_mc = []
    cp_lw_mc = []

    all_tasks = []
    no_tasks = []
    task_labels = []
    most_pop = []

    for count in all_counts: #each set of categories, in the ensemble (all_counts)
        #set of categories for each agent

        #ip_cp.append(count[5])
        #ip_lw.append(count[6])
        mc_inv.append(count[4]) #number of MC-exclusive neurons in each agent
        ip_mc.append(count[7])
        cp_mc.append(count[9])
        lw_mc.append(count[10])
        ip_cp_mc.append(count[12])
        ip_lw_mc.append(count[13])
        cp_lw_mc.append(count[14])
        all_tasks.append(count[15])
        no_tasks.append(count[0])
        count = list(count)
        pop = count.index(max(count)) #calculate highest value in count
        cat_pop = categs[pop] #get corresponding category 
        most_pop.append(cat_pop)
    #print(most_pop)
        
    
     

    """
        all_tasks.append(count[15])
        no_tasks.append(count[0])
        if count[4] > 0:
            task_labels[0] = 'MC'
        if count[7] > 0:
            task_labels[1] = 'MC'
        if count[9] > 0:
            task_labels[2] = 'CPMC'
        if count[10] > 0:
            task_labels[3] = 'LWMC'
        if count[12] > 0:
            task_labels[4] = 'IPCPMC'
        if count[13] > 0:
            task_labels[5] = 'IPLWMC'
        if count[14] > 0:
            task_labels[6] = 'CPLWMC'
        if count[15] > 0:
            task_labels[7] = 'All'
        if count[0] > 0:
            task_labels[8] = 'None'
        else:
            pass
        print(len(task_labels))
        ensemble_labels.append(task_labels)
    #print(mc_inv)
    #print(lw_mc)
    #print(ensemble_labels)
    """
    np.save("./Combined/4T_2x20/Data"+"/MC"+".npy",mc_inv)
    np.save("./Combined/4T_2x20/Data"+"/ip_mc"+".npy",ip_mc)
    np.save("./Combined/4T_2x20/Data"+"/cp_mc"+".npy",cp_mc)
    np.save("./Combined/4T_2x20/Data"+"/lw_mc"+".npy",lw_mc)
    np.save("./Combined/4T_2x20/Data"+"/ip_cp_mc"+".npy",ip_cp_mc)
    np.save("./Combined/4T_2x20/Data"+"/ip_lw_mc"+".npy",ip_lw_mc)
    np.save("./Combined/4T_2x20/Data"+"/cp_lw_mc"+".npy",cp_lw_mc)
    np.save("./Combined/4T_2x20/Data"+"/all"+".npy",all_tasks)
    np.save("./Combined/4T_2x20/Data"+"/none"+".npy",no_tasks)
    print(most_pop)
    np.save("./Combined/4T_2x20/Data"+"/most_pop_cat"+".npy",most_pop)
    
    """
    # plot specialization and reuse
    plt.figure(figsize=[4, 4])
    #ax2 = plt.subplot2grid([1, 3], [0, 2], adjustable="box", aspect=1)
    #ax2.plot([-0.5, 21.5], [21.5, -0.5], "k", linewidth=0.7)
    #ax2 = plt.subplot2grid([1, 4], [0, 4], adjustable="box", aspect=1)
    plt.plot([0.0, 1.0], [0.0, 1.0], "k", linewidth=0.7)
    """
    reused_count = []
    reused_count_2 = []
    reused_count_3 = []
    reused_count_4 = []
    special_count = []
    #count_data_prop = []
    for count in all_counts:
        # plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
        #count_data.append([count[1] + count[2] + count[3] + count[4], np.sum(count[5:])])
        #count_data_prop.append([((count[1] + count[2] + count[3] + count[4])/20), ((np.sum(count[5:]))/20)])
        #count_data_prop.append([((np.sum(count[:4]))/20), ((np.sum(count[5:]))/20)])
        reused_count.append((np.sum(count[5:]))/40)
        reused_count_2.append((count[5]+count[6]+count[7]+count[8]+count[9]+count[10])/40)
        reused_count_3.append((count[1]+count[12]+count[13]+count[14])/40)
        reused_count_4.append((count[15])/40)
        #special_count.append((np.sum(count[1:5]))/20)
        special_count.append((count[1]+count[2]+count[3]+count[4])/40)
        #print(len(count_data_prop))
    
    print(reused_count)
    print(reused_count_2)
    print(reused_count_3)
    print(reused_count_4)
    print(special_count)

    np.save("./Combined/4T_2x20/Data"+"/reused_prop"+".npy",reused_count)
    np.save("./Combined/4T_2x20/Data"+"/reused_prop_2"+".npy",reused_count_2)
    np.save("./Combined/4T_2x20/Data"+"/reused_prop_3"+".npy",reused_count_3)
    np.save("./Combined/4T_2x20/Data"+"/reused_prop_4"+".npy",reused_count_4)
    np.save("./Combined/4T_2x20/Data"+"/special_prop"+".npy",special_count)



#NEW REUSE PLOT: PROPORTION OF REUSED NEURONS
"""

    plt.scatter(reused_count,special_count)
    plt.title("Proportion of Neural Reuse, 2x20")
    plt.ylabel("Prop. of specialized neurons")
    plt.xlabel("prop of reused neurons")
    plt.savefig("./Combined/4T_2x20/Figures"+"/reuse_proportions.png")
    plt.show()



    



    
    df = pd.DataFrame(
        count_data, columns=["No. of specialized neurons", "No. of reused neurons"]
    )
    
    df = pd.DataFrame(
        count_data_prop, columns=["prop. of specialized neurons", "prop. of reused neurons"]
    )
    
    ax = sns.stripplot(
        x="prop. of specialized neurons",
        y="prop. of reused neurons",
        data=df,
        #palette=
        #palette=dict([(i, "xkcd:velvet") for i in range(10)]),
        alpha=0.8,
        s=7,
    )
    
    #plt.xticks(np.arange(22), np.arange(22))
    plt.xticks(np.arange(10), np.arange(10))
    plt.yticks(np.arange(10), np.arange(10))
    #plt.yticks(np.arange(22), np.arange(22))
    #plt.xticks(0.0,1.0)
    #plt.xlim([-0.5, 21.5])
    plt.xlim([0, 10])
    plt.ylim([0, 1])
    plt.xlabel("prop of Specialized Neurons")
    plt.ylabel("prop of Reused Neurons")

    # make df and plot
   plt.subplot2grid([1, 3], [1, 0], colspan=3)
    df = pd.DataFrame(
        all_categs, columns=["Category", "No. of Neurons", "network_id"]
    )
    ax = sns.swarmplot(x="Category", y="No. of Neurons", hue="network_id", data=df)
    ax.legend_.remove()

    plt.tight_layout()
    plt.savefig("./Combined/4T_2x10/Figures/figure_5_reuse_prop.pdf")
    plt.show()
    """


plot_lesion_analysis(11)