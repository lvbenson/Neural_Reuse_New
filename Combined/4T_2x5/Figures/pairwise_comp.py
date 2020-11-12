import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Reuse data
reused_2x5 = np.load("./Combined/4T_2x5/Data/reused_prop.npy")
reused_2x3 = np.load("./Combined/4T_2x3/Data/reused_prop.npy")
reused_2x10 = np.load("./Combined/4T_2x10/Data/reused_prop.npy")
reused_2x20 = np.load("./Combined/4T_2x20/Data/reused_prop.npy")

#Specialized data
special_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")
special_2x3 = np.load("./Combined/4T_2x3/Data/special_prop.npy")
special_2x10 = np.load("./Combined/4T_2x10/Data/special_prop.npy")
special_2x20 = np.load("./Combined/4T_2x20/Data/special_prop.npy")

#Pairwise task involvement, MC 2x3
ipmc_4 = np.load("./Combined/4T_2x3/Data/ip_mc.npy")
cpmc_4 = np.load("./Combined/4T_2x3/Data/cp_mc.npy")
lwmc_4 = np.load("./Combined/4T_2x3/Data/lw_mc.npy")
all_4 = np.load("./Combined/4T_2x3/Data/all.npy")
none_4 = np.load("./Combined/4T_2x3/Data/none.npy")


#Pairwise task involvement, MC 2x5
ipmc = np.load("./Combined/4T_2x5/Data/ip_mc.npy")
cpmc = np.load("./Combined/4T_2x5/Data/cp_mc.npy")
lwmc = np.load("./Combined/4T_2x5/Data/lw_mc.npy")
all_ = np.load("./Combined/4T_2x5/Data/all.npy")
none_ = np.load("./Combined/4T_2x5/Data/none.npy")

#Pairwise task involvement, MC 2x10
ipmc_2 = np.load("./Combined/4T_2x10/Data/ip_mc.npy")
cpmc_2 = np.load("./Combined/4T_2x10/Data/cp_mc.npy")
lwmc_2 = np.load("./Combined/4T_2x10/Data/lw_mc.npy")
all_2 = np.load("./Combined/4T_2x10/Data/all.npy")
none_2 = np.load("./Combined/4T_2x10/Data/none.npy")

#Pairwise task involvement, MC 2x20
ipmc_3 = np.load("./Combined/4T_2x20/Data/ip_mc.npy")
cpmc_3 = np.load("./Combined/4T_2x20/Data/cp_mc.npy")
lwmc_3 = np.load("./Combined/4T_2x20/Data/lw_mc.npy")
all_3 = np.load("./Combined/4T_2x20/Data/all.npy")
none_3 = np.load("./Combined/4T_2x20/Data/none.npy")




#Pairwise task involvement labels
labels = np.load("./Combined/4T_2x5/Data/task_labels.npy")
labels2 = np.load("./Combined/4T_2x10/Data/task_labels.npy")
labels3 = np.load("./Combined/4T_2x20/Data/task_labels.npy")
labels4 = np.load("./Combined/4T_2x3/Data/task_labels.npy")

"""
mc_pairs = {'Task': labels,
'IP+MC': ipmc,
'CP+MC': cpmc,
'LW+MC': lwmc,
'All': all_,
'None': none_,
'PropReused': reused_2x5,
'PropSpecialized': special_2x5}

df = pd.DataFrame(mc_pairs,columns=['Task','IP+MC','CP+MC','LW+MC','All','None','PropReused','PropSpecialized'])

sns.pairplot(df,hue='Task')
plt.savefig("./Combined/4T_2x5/Figures/pairwise_mc.png")
plt.show()
#plt.close()


mc_pairs_2 = {'Task': labels2,
'IP+MC': ipmc_2,
'CP+MC': cpmc_2,
'LW+MC': lwmc_2,
'All': all_2,
'None': none_2,
'PropReused': reused_2x10,
'PropSpecialized': special_2x10}

df_2 = pd.DataFrame(mc_pairs_2,columns=['Task','IP+MC','CP+MC','LW+MC','All','None','PropReused','PropSpecialized'])

sns.pairplot(df_2,hue='Task')
plt.savefig("./Combined/4T_2x10/Figures/pairwise_mc.png")
#plt.show()
"""

mc_pairs_3 = {'Task': labels3,
'IP+MC': ipmc_3,
'CP+MC': cpmc_3,
'LW+MC': lwmc_3,
'All': all_3,
'None': none_3,
'PropReused': reused_2x20,
'PropSpecialized': special_2x20}

df_3 = pd.DataFrame(mc_pairs_3,columns=['Task','IP+MC','CP+MC','LW+MC','All','None','PropReused','PropSpecialized'])

sns.pairplot(df_3,hue='Task', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             size = 1.2)
#plt.savefig("./Combined/4T_2x20/Figures/pairwise_mc.png")
plt.show()

"""
mc_pairs_4 = {'Task': labels4,
'IP+MC': ipmc_4,
'CP+MC': cpmc_4,
'LW+MC': lwmc_4,
'All': all_4,
'None': none_4,
'PropReused': reused_2x3,
'PropSpecialized': special_2x3}

df_4 = pd.DataFrame(mc_pairs_4,columns=['Task','IP+MC','CP+MC','LW+MC','All','None','PropReused','PropSpecialized'])

sns.pairplot(df_4,hue='Task')
plt.savefig("./Combined/4T_2x3/Figures/pairwise_mc.png")
#plt.show()
"""