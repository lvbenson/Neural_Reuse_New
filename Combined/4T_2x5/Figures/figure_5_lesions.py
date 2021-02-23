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



def plot_lesion_analysis():
    
    dir = "./Combined/4T_3x20/Data/"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    #agentlen = 40
    files.sort()

    all_categs = []
    all_counts = []
    
    for i, file in enumerate(files):
        fits = np.load(file)
        # if np.prod(fits) > 0.8:
        fits = fits**(1/4)
        if 1 + 1 == 2:
            ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
            ipp = np.load(dir + "lesions_IP_" + str(ind) + ".npy")
            cpp = np.load(dir + "lesions_CP_" + str(ind) + ".npy")
            lwp = np.load(dir + "lesions_LW_" + str(ind) + ".npy")
            mcp = np.load(dir + "lesions_MC_" + str(ind) + ".npy")
            #ipp = np.load("./Combined/4T_3x5/Data/lesions_IP_" + str(ind) + ".npy") #10 values, one for each neuron in a circuit
            #cpp = np.load("./Combined/4T_3x5/Data/lesions_CP_" + str(ind) + ".npy")
            #lwp = np.load("./Combined/4T_3x5/Data/lesions_LW_" + str(ind) + ".npy")
            #mcp = np.load("./Combined/4T_3x5/Data/lesions_MC_" + str(ind) + ".npy")

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
            #np.save(dir + "NEWSTATS_" + str(i) + ".npy", count)
            #np.save("C:/Users/benso/Desktop/Projects/Neural_Reuse/Neural_Reuse_New/Combined/4T_3x5/Data"+"/NEWSTATS_" + str(i) + ".npy",count)
            all_counts.append(count) #count is a 1x15 array for each agent. All_counts is 15xensemble size 
            #all categories: reuse and specialization
            categs = ["None","IP","CP","LW","MC","IP+CP","IP+LW","IP+MC","CP+LW","CP+MC","LW+MC","IP+CP+LW","IP+CP+MC","IP+LW+MC","CP+LW+MC","All"]

        




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
    np.save("./Combined/4T_3x5/Data"+"/MC"+".npy",mc_inv)
    np.save("./Combined/4T_3x5/Data"+"/ip_mc"+".npy",ip_mc)
    np.save("./Combined/4T_3x5/Data"+"/cp_mc"+".npy",cp_mc)
    np.save("./Combined/4T_3x5/Data"+"/lw_mc"+".npy",lw_mc)
    np.save("./Combined/4T_3x5/Data"+"/ip_cp_mc"+".npy",ip_cp_mc)
    np.save("./Combined/4T_3x5/Data"+"/ip_lw_mc"+".npy",ip_lw_mc)
    np.save("./Combined/4T_3x5/Data"+"/cp_lw_mc"+".npy",cp_lw_mc)
    np.save("./Combined/4T_3x5/Data"+"/all"+".npy",all_tasks)
    np.save("./Combined/4T_3x5/Data"+"/none"+".npy",no_tasks)
    #print(most_pop25
    np.save("./Combined/4T_3x5/Data"+"/most_pop_cat"+".npy",most_pop)
    """
    
    #count_data = []
    neuron_nums_reuse = []
    neuron_nums_2 = []
    neuron_nums_3 = []
    neuron_nums_4 = []
    neuron_nums_special = []
    
    reused_count = []
    reused_count_2 = []
    reused_count_3 = []
    reused_count_4 = []
    special_count = []

    IP_reuse = []
    CP_reuse = []
    LW_reuse = []
    MC_reuse = []


    #count_data_prop = []
    for count in all_counts: #for every agent in the ensemble...

        # plt.scatter(count[1]+count[2]+count[3], np.sum(count[4:]), c="C0")
        #count_data.append([count[1] + count[2] + count[3] + count[4], np.sum(count[5:])])
        #count_data_prop.append([((count[1] + count[2] + count[3] + count[4])/20), ((np.sum(count[5:]))/20)])
        #count_data_prop.append([((np.sum(count[:4]))/20), ((np.sum(count[5:]))/20)])
        reused_count.append((np.sum(count[5:]))/15)
        neuron_nums_reuse.append(np.sum(count[5:]))

        reused_count_2.append((count[5]+count[6]+count[7]+count[8]+count[9]+count[10])/15)
        neuron_nums_2.append(count[5]+count[6]+count[7]+count[8]+count[9]+count[10])

        reused_count_3.append((count[1]+count[12]+count[13]+count[14])/15)
        neuron_nums_3.append(count[1]+count[12]+count[13]+count[14])
        
        reused_count_4.append((count[15])/15)
        neuron_nums_4.append(count[15])

        special_count.append((count[1]+count[2]+count[3]+count[4])/15)
        neuron_nums_special.append(count[1]+count[2]+count[3]+count[4])

        #MC: 4, 7, 9, 10, 12, 13, 14, 15
            #LW: 3, 6, 8, 10, 11, 13, 14, 15
            #CP: 2, 5, 8, 9, 11, 12, 14, 15
            #IP: 1, 5, 6, 7, 11, 12, 13, 15
        
        IP_reuse.append((count[5]+count[6]+count[7]+count[11]+count[12]+count[13]+count[15])/(count[1]+count[5]+count[6]+count[7]+count[11]+count[12]+count[13]+count[15]))
        CP_reuse.append((count[5]+count[8]+count[9]+count[11]+count[12]+count[14]+count[15])/(count[2]+count[5]+count[8]+count[9]+count[11]+count[12]+count[14]+count[15]))
        LW_reuse.append((count[6]+count[8]+count[10]+count[11]+count[13]+count[14]+count[15])/(count[3]+count[6]+count[8]+count[10]+count[11]+count[13]+count[14]+count[15]))
        MC_reuse.append((count[7]+count[9]+count[10]+count[12]+count[13]+count[14]+count[15])/(count[4]+count[7]+count[9]+count[10]+count[12]+count[13]+count[14]+count[15]))

    np.save(dir + "IP_reuse" + ".npy", IP_reuse)
    np.save(dir + "CP_reuse" + ".npy", CP_reuse)
    np.save(dir + "LW_reuse" + ".npy", LW_reuse)
    np.save(dir + "MC_reuse" + ".npy", MC_reuse)

    #print(IP_reuse)
    #print(CP_reuse)
    #print(LW_reuse)
    print(LW_reuse)
    print(len(MC_reuse))


    """
    #print(reuse_neuron_nums/ensemble)
    np.save(dir + "neuron_nums_reuse"+".npy",neuron_nums_reuse)
    np.save(dir + "neuron_nums_reuse_2"+".npy",neuron_nums_2)
    np.save(dir + "neuron_nums_reuse_3"+".npy",neuron_nums_3)
    np.save(dir + "neuron_nums_reuse_4"+".npy",neuron_nums_4)
    np.save(dir + "neuron_nums_special"+".npy",neuron_nums_special)
    
    #print(neuron_nums_reuse)
    #print(neuron_nums_4)
    #print(neuron_nums_3)
    print(neuron_nums_2)
    
    
    np.save("./Combined/4T_3x5/Data"+"/reused_prop"+".npy",reused_count)
    np.save("./Combined/4T_3x5/Data"+"/reused_prop_2"+".npy",reused_count_2)
    np.save("./Combined/4T_3x5/Data"+"/reused_prop_3"+".npy",reused_count_3)
    np.save("./Combined/4T_3x5/Data"+"/reused_prop_4"+".npy",reused_count_4)
    np.save("./Combined/4T_3x5/Data"+"/special_prop"+".npy",special_count)
    """


#NEW REUSE PLOT: PROPORTION OF REUSED NEURONS
    """
    plt.scatter(reused_count,special_count)
    plt.title("Proportion of Neural Reuse, 2x5")
    plt.ylabel("Prop. of specialized neurons")
    plt.xlabel("prop of reused neurons")
    plt.savefig("./Combined/4T_2x5/Figures"+"/reuse_proportions.png")
    plt.show()
    """

plot_lesion_analysis()
