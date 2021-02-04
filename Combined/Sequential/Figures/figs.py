import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

reused_2x5 = np.load("./Combined/4T_2x5/Data/reused_prop.npy")
reused_seq = np.load("./Combined/Sequential/Data/reused_prop.npy")

special_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")
special_seq = np.load("./Combined/Sequential/Data/special_prop.npy")

#Pairwise task involvement, MC 2x5
mc = np.load("./Combined/4T_2x5/Data/MC.npy")
ipmc = np.load("./Combined/4T_2x5/Data/ip_mc.npy")
cpmc = np.load("./Combined/4T_2x5/Data/cp_mc.npy")
lwmc = np.load("./Combined/4T_2x5/Data/lw_mc.npy")

ipcpmc = np.load("./Combined/4T_2x5/Data/ip_cp_mc.npy")
iplwmc = np.load("./Combined/4T_2x5/Data/ip_lw_mc.npy")
cplwmc = np.load("./Combined/4T_2x5/Data/cp_lw_mc.npy")

all_ = np.load("./Combined/4T_2x5/Data/all.npy")
none_ = np.load("./Combined/4T_2x5/Data/none.npy")

#Pairwise task involvement, Sequential (MC)

mc4 = np.load("./Combined/Sequential/Data/MC.npy")
ipmc4 = np.load("./Combined/Sequential/Data/ip_mc.npy")
cpmc4 = np.load("./Combined/Sequential/Data/cp_mc.npy")
lwmc4 = np.load("./Combined/Sequential/Data/lw_mc.npy")

ipcpmc4 = np.load("./Combined/Sequential/Data/ip_cp_mc.npy")
iplwmc4 = np.load("./Combined/Sequential/Data/ip_lw_mc.npy")
cplwmc4 = np.load("./Combined/Sequential/Data/cp_lw_mc.npy")

all_4 = np.load("./Combined/Sequential/Data/all.npy")
none_4 = np.load("./Combined/Sequential/Data/none.npy")

#most pop cat
most_pop = np.load("./Combined/4T_2x5/Data/most_pop_cat.npy")
most_pop4 = np.load("./Combined/Sequential/Data/most_pop_cat.npy")

#2x5 dataframe

mc_pairs = {'Most Popular Cat': most_pop,
'MC': mc,
'IP+MC': ipmc,
'CP+MC': cpmc,
'LW+MC': lwmc,
'IP+CP+MC': ipcpmc,
'IP+LW+MC': iplwmc,
'CP+LW+MC': cplwmc,
'All': all_,
'None': none_,
'PropReused': reused_2x5,
'PropSpecialized': special_2x5,}

df = pd.DataFrame(mc_pairs,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])

fig, ax = plt.subplots(2,2,figsize=(6,7),sharey=True)

ax1 = sns.kdeplot(ax=ax[0,0],data=df, x="PropReused", hue="Most Popular Cat")
ax1.legend_.remove()
#ax1.set_title('Reuse: 2x5')
#
ax2 = sns.kdeplot(ax=ax[0,1],data=df, x="PropSpecialized", hue="Most Popular Cat")
#ax2.set_title('Specialization: 2x5')
plt.tight_layout()

#sequential dataframe (2x5)

mc_pairs4 = {'Most Popular Cat': most_pop4,
'MC': mc4,
'IP+MC': ipmc4,
'CP+MC': cpmc4,
'LW+MC': lwmc4,
'IP+CP+MC': ipcpmc4,
'IP+LW+MC': iplwmc4,
'CP+LW+MC': cplwmc4,
'All': all_4,
'None': none_4,
'PropReused': reused_seq,
'PropSpecialized': special_seq}

df4 = pd.DataFrame(mc_pairs4,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])


ax8 = sns.kdeplot(ax=ax[1,0],data=df4, x="PropReused", hue="Most Popular Cat")
ax8.legend_.remove()

#ax8.set_title('Sequential, Reused: 2x5')

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax9 = sns.kdeplot(ax=ax[1,1],data=df4, x="PropSpecialized", hue="Most Popular Cat")
plt.tight_layout()
#ax8.legend_.remove()
#ax2.set_title('Specialization')
#ax9.set_title('Sequential, Special: 2x5')




plt.suptitle('Simultaneous vs Sequential: 2x5')

plt.tight_layout()
plt.savefig("./Combined/Sequential/Figures/density_reuse.png")
plt.show()