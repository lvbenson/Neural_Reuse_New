import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

IP_2x3 = np.load("./Combined/4T_2x3/Data/" + "IP_reuse" + ".npy")
CP_2x3 = np.load("./Combined/4T_2x3/Data/" + "CP_reuse" + ".npy")
LW_2x3 = np.load("./Combined/4T_2x3/Data/" + "LW_reuse" + ".npy")
MC_2x3 = np.load("./Combined/4T_2x3/Data/" + "MC_reuse" + ".npy")
    
IP_2x5 = np.load("./Combined/4T_2x5/Data/" + "IP_reuse" + ".npy")
CP_2x5 = np.load("./Combined/4T_2x5/Data/" + "CP_reuse" + ".npy")
LW_2x5 = np.load("./Combined/4T_2x5/Data/" + "LW_reuse" + ".npy")
MC_2x5 = np.load("./Combined/4T_2x5/Data/" + "MC_reuse" + ".npy")

IP_2x10 = np.load("./Combined/4T_2x10/Data/" + "IP_reuse" + ".npy")
CP_2x10 = np.load("./Combined/4T_2x10/Data/" + "CP_reuse" + ".npy")
LW_2x10 = np.load("./Combined/4T_2x10/Data/" + "LW_reuse" + ".npy")
MC_2x10 = np.load("./Combined/4T_2x10/Data/" + "MC_reuse" + ".npy")

IP_2x20 = np.load("./Combined/4T_2x20/Data/" + "IP_reuse" + ".npy")
CP_2x20 = np.load("./Combined/4T_2x20/Data/" + "CP_reuse" + ".npy")
LW_2x20 = np.load("./Combined/4T_2x20/Data/" + "LW_reuse" + ".npy")
MC_2x20 = np.load("./Combined/4T_2x20/Data/" + "MC_reuse" + ".npy")

IP_3x5 = np.load("./Combined/4T_3x5/Data/" + "IP_reuse" + ".npy")
CP_3x5 = np.load("./Combined/4T_3x5/Data/" + "CP_reuse" + ".npy")
LW_3x5 = np.load("./Combined/4T_3x5/Data/" + "LW_reuse" + ".npy")
MC_3x5 = np.load("./Combined/4T_3x5/Data/" + "MC_reuse" + ".npy")

IP_3x10 = np.load("./Combined/4T_3x10/Data/" + "IP_reuse" + ".npy")
CP_3x10 = np.load("./Combined/4T_3x10/Data/" + "CP_reuse" + ".npy")
LW_3x10 = np.load("./Combined/4T_3x10/Data/" + "LW_reuse" + ".npy")
MC_3x10 = np.load("./Combined/4T_3x10/Data/" + "MC_reuse" + ".npy")


df = pd.DataFrame({"IP": [np.average(IP_2x3),np.average(IP_2x5),np.average(IP_2x10),np.average(IP_2x20),np.average(IP_3x5),np.average(IP_3x10)],
"CP": [np.average(CP_2x3),np.average(CP_2x5),np.average(CP_2x10),np.average(CP_2x20),np.average(CP_3x5),np.average(CP_3x10)],
"LW": [np.average(LW_2x3),np.average(LW_2x5),np.average(LW_2x10),np.average(LW_2x20),np.average(LW_3x5),np.average(LW_3x10)],
"MC": [np.average(MC_2x3),np.average(MC_2x5),np.average(MC_2x10),np.average(MC_2x20),np.average(MC_3x5),np.average(MC_3x10)],
"Size": ['2x3','2x5','2x10','2x20','3x5','3x10']})

dfnew = pd.DataFrame({"2x3": [np.average(IP_2x3),np.average(CP_2x3),np.average(LW_2x3),np.average(MC_2x3)],
"2x5": [np.average(IP_2x5),np.average(CP_2x5),np.average(LW_2x5),np.average(MC_2x5)],
"2x10": [np.average(IP_2x10),np.average(CP_2x10),np.average(LW_2x10),np.average(MC_2x10)],
"2x20": [np.average(IP_2x20),np.average(CP_2x20),np.average(LW_2x20),np.average(MC_2x20)],
"3x5": [np.average(IP_3x5),np.average(CP_3x5),np.average(LW_3x5),np.average(MC_3x5)],
"3x10": [np.average(IP_3x10),np.average(CP_3x10),np.average(LW_3x10),np.average(MC_3x10)],
"Task": ["IP","CP","LW","MC"]
})

df2 = pd.DataFrame({'Network Size': ['2x3','2x3','2x3','2x3','2x5','2x5','2x5','2x5','2x10','2x10','2x10','2x10',
'2x20','2x20','2x20','2x20','3x5','3x5','3x5','3x5','3x10','3x10','3x10','3x10'],
'Reuse Proportion': [np.average(IP_2x3),np.average(CP_2x3),np.average(LW_2x3),np.average(MC_2x3),
np.average(IP_2x5),np.average(CP_2x5),np.average(LW_2x5),np.average(MC_2x5),
np.average(IP_2x10),np.average(CP_2x10),np.average(LW_2x10),np.average(MC_2x10),
np.average(IP_2x20),np.average(CP_2x20),np.average(LW_2x20),np.average(MC_2x20),
np.average(IP_3x5),np.average(CP_3x5),np.average(LW_3x5),np.average(MC_3x5),
np.average(IP_3x10),np.average(CP_3x10),np.average(LW_3x10),np.average(MC_3x10)],
'Task': ["IP","CP","LW","MC","IP","CP","LW","MC","IP","CP","LW","MC","IP","CP","LW","MC","IP","CP","LW","MC","IP","CP","LW","MC"]
})

#print(dfnew.head())

sns.catplot(x='Network Size',y='Reuse Proportion',data=df2,hue='Task',s=8,aspect=1.5,legend_out=True)
plt.title("Task-Specific Reuse For Different Network Sizes")
#plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("C:/Users/benso/Desktop/Projects/Neural_Reuse/Neural_Reuse_New/Combined/4T_2x5/Figures/REUSE_TASKS_NETSIZES.png")
plt.show()

#sns.catplot(x=index, y="total_bill", data=df)
