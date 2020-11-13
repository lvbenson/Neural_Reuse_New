
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns



dir = "./Combined/4T_2x5/Data"
files = glob.glob(os.path.join(dir, "perf_*.npy"))
files.sort()


all_FTV = []
n1 = []
n2 = []
n3 = []
n4 = []
n5 = []
n6 = []
n7 = []
n8 = []
n9 = []
n10 = []
for i, file in enumerate(files):
    fits = np.load(file)
    # if np.prod(fits) > 0.8:
    fits = fits**(1/4)
    if np.min(fits) > 0.8:
        ind = file.split("/")[-1].split(".")[-2].split("_")[-1]
        ipp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_IP_" + str(ind) + ".npy") #size 10, 1 for each neuron
        cpp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_CP_" + str(ind) + ".npy")
        lwp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_LW_" + str(ind) + ".npy")
        mcp = 1 - np.load("./Combined/4T_2x5/Data/NormVar_MC_" + str(ind) + ".npy")

        ###########################################################################
        #Fractional Task variance for one agent#
        ###########################################################################
        FTV_ip_cp = []
        for ip,cp in zip(lwp,mcp):
            FTV_ip_cp.append((ip - cp)/(ip + cp)) #a FTV for neuron i that is -1 or 1 means that neuron is primarily selective in one of the tasks
        #print(FTV_ip_cp[0])
        n1.append(FTV_ip_cp[0])
        n2.append(FTV_ip_cp[1])
        n3.append(FTV_ip_cp[2])
        n4.append(FTV_ip_cp[3])
        n5.append(FTV_ip_cp[4])
        n6.append(FTV_ip_cp[5])
        n7.append(FTV_ip_cp[6])
        n8.append(FTV_ip_cp[7])
        n9.append(FTV_ip_cp[8])
        n10.append(FTV_ip_cp[9])
    

colors = ['gray', 'lightcoral', 'sienna', 'darkorange', 'yellow','lawngreen','lightseagreen','royalblue','blueviolet','deeppink']
names = ['neuron1', 'neuron2', 'neuron3', 'neuron4','neuron5','neuron6','neuron7','neuron8','neuron9','neuron10']

plt.hist([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10], bins = int(180/20),
         color = colors, label=names)

# Plot formatting
plt.legend()
plt.xlabel('Fractional Task Variance')
plt.ylabel('Neurons in ensemble')
plt.title('FTV: LW and MC')
plt.savefig("./Combined/4T_2x5/Figures/FTV_lw_mc.pdf")
plt.show()