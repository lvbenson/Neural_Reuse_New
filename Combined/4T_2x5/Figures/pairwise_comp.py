import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
#print(os.getcwd())
#Reuse data

reused_2x5 = np.load("./Combined/4T_2x5/Data/reused_prop.npy")
#print(reused_2x5)
reused_2x3 = np.load("./Combined/4T_2x3/Data/reused_prop.npy")
reused_2x10 = np.load("./Combined/4T_2x10/Data/reused_prop.npy")
reused_2x20 = np.load("./Combined/4T_2x20/Data/reused_prop.npy")
reused_seq = np.load("./Combined/Sequential/Data/reused_prop.npy")

#Specialized data
special_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")
special_2x3 = np.load("./Combined/4T_2x3/Data/special_prop.npy")
special_2x10 = np.load("./Combined/4T_2x10/Data/special_prop.npy")
special_2x20 = np.load("./Combined/4T_2x20/Data/special_prop.npy")
special_seq = np.load("./Combined/Sequential/Data/special_prop.npy")

#Pairwise task involvement, MC 2x3
mc3 = np.load("./Combined/4T_2x3/Data/MC.npy")
ipmc3 = np.load("./Combined/4T_2x3/Data/ip_mc.npy")
cpmc3 = np.load("./Combined/4T_2x3/Data/cp_mc.npy")
lwmc3 = np.load("./Combined/4T_2x3/Data/lw_mc.npy")

ipcpmc3 = np.load("./Combined/4T_2x3/Data/ip_cp_mc.npy")
iplwmc3 = np.load("./Combined/4T_2x3/Data/ip_lw_mc.npy")
cplwmc3 = np.load("./Combined/4T_2x3/Data/cp_lw_mc.npy")

all_3 = np.load("./Combined/4T_2x3/Data/all.npy")
none_3 = np.load("./Combined/4T_2x3/Data/none.npy")


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


#Pairwise task involvement, MC 2x10
mc1 = np.load("./Combined/4T_2x10/Data/MC.npy")
ipmc1 = np.load("./Combined/4T_2x10/Data/ip_mc.npy")
cpmc1 = np.load("./Combined/4T_2x10/Data/cp_mc.npy")
lwmc1 = np.load("./Combined/4T_2x10/Data/lw_mc.npy")

ipcpmc1 = np.load("./Combined/4T_2x10/Data/ip_cp_mc.npy")
iplwmc1 = np.load("./Combined/4T_2x10/Data/ip_lw_mc.npy")
cplwmc1 = np.load("./Combined/4T_2x10/Data/cp_lw_mc.npy")

all_1 = np.load("./Combined/4T_2x10/Data/all.npy")
none_1 = np.load("./Combined/4T_2x10/Data/none.npy")

#Pairwise task involvement, MC 2x20
mc2 = np.load("./Combined/4T_2x20/Data/MC.npy")
ipmc2 = np.load("./Combined/4T_2x20/Data/ip_mc.npy")
cpmc2 = np.load("./Combined/4T_2x20/Data/cp_mc.npy")
lwmc2 = np.load("./Combined/4T_2x20/Data/lw_mc.npy")

ipcpmc2 = np.load("./Combined/4T_2x20/Data/ip_cp_mc.npy")
iplwmc2 = np.load("./Combined/4T_2x20/Data/ip_lw_mc.npy")
cplwmc2 = np.load("./Combined/4T_2x20/Data/cp_lw_mc.npy")

all_2 = np.load("./Combined/4T_2x20/Data/all.npy")
none_2 = np.load("./Combined/4T_2x20/Data/none.npy")

#Pairwise task involvement, MC Sequential (2x5)
#Pairwise task involvement, MC 2x20
mc4 = np.load("./Combined/Sequential/Data/MC.npy")
ipmc4 = np.load("./Combined/Sequential/Data/ip_mc.npy")
cpmc4 = np.load("./Combined/Sequential/Data/cp_mc.npy")
lwmc4 = np.load("./Combined/Sequential/Data/lw_mc.npy")

ipcpmc4 = np.load("./Combined/Sequential/Data/ip_cp_mc.npy")
iplwmc4 = np.load("./Combined/Sequential/Data/ip_lw_mc.npy")
cplwmc4 = np.load("./Combined/Sequential/Data/cp_lw_mc.npy")

all_4 = np.load("./Combined/Sequential/Data/all.npy")
none_4 = np.load("./Combined/Sequential/Data/none.npy")


#Most Popular Category

most_pop = np.load("./Combined/4T_2x5/Data/most_pop_cat.npy")
most_pop1 = np.load("./Combined/4T_2x10/Data/most_pop_cat.npy")
most_pop2 = np.load("./Combined/4T_2x20/Data/most_pop_cat.npy")
most_pop3 = np.load("./Combined/4T_2x3/Data/most_pop_cat.npy")
most_pop4 = np.load("./Combined/Sequential/Data/most_pop_cat.npy")

#2x5
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

#############################################
#2x5
#############################################

df = pd.DataFrame(mc_pairs,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])

fig, ax = plt.subplots(5,2,figsize=(10,10),sharey=True)

ax1 = sns.kdeplot(ax=ax[0,0],data=df, x="PropReused", hue="Most Popular Cat")
ax1.legend_.remove()
ax1.set_title('Reuse: 2x5')

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax2 = sns.kdeplot(ax=ax[0,1],data=df, x="PropSpecialized", hue="Most Popular Cat")
ax2.set_title('Specialization: 2x5')
#plt.savefig("./Combined/4T_2x5/Figures/density_reuse.png")


#############################################
#2x10
#############################################

mc_pairs1 = {'Most Popular Cat': most_pop1,
'MC': mc1,
'IP+MC': ipmc1,
'CP+MC': cpmc1,
'LW+MC': lwmc1,
'IP+CP+MC': ipcpmc1,
'IP+LW+MC': iplwmc1,
'CP+LW+MC': cplwmc1,
'All': all_1,
'None': none_1,
'PropReused': reused_2x10,
'PropSpecialized': special_2x10}

df1 = pd.DataFrame(mc_pairs1,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])


ax3 = sns.kdeplot(ax=ax[1,0],data=df1, x="PropReused", hue="Most Popular Cat")
ax3.set_title('2x10')
ax3.legend_.remove()

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax4 = sns.kdeplot(ax=ax[1,1],data=df1, x="PropSpecialized", hue="Most Popular Cat")
#x4.legend_.remove()
#ax2.set_title('Specialization')
#plt.savefig("./Combined/4T_2x5/Figures/density_reuse.png")
ax4.set_title('2x10')

#############################################
#2x20
#############################################

mc_pairs2 = {'Most Popular Cat': most_pop2,
'MC': mc2,
'IP+MC': ipmc2,
'CP+MC': cpmc2,
'LW+MC': lwmc2,
'IP+CP+MC': ipcpmc2,
'IP+LW+MC': iplwmc2,
'CP+LW+MC': cplwmc2,
'All': all_2,
'None': none_2,
'PropReused': reused_2x20,
'PropSpecialized': special_2x20}

df2 = pd.DataFrame(mc_pairs2,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])


ax5 = sns.kdeplot(ax=ax[2,0],data=df2, x="PropReused", hue="Most Popular Cat")
ax5.set_title('2x20')
ax5.legend_.remove()

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax6 = sns.kdeplot(ax=ax[2,1],data=df2, x="PropSpecialized", hue="Most Popular Cat")
#x4.legend_.remove()
#ax2.set_title('Specialization')
#plt.savefig("./Combined/4T_2x5/Figures/density_reuse.png")
ax6.set_title('2x20')

#############################################
#2x3
#############################################
mc_pairs3 = {'Most Popular Cat': most_pop3,
'MC': mc3,
'IP+MC': ipmc3,
'CP+MC': cpmc3,
'LW+MC': lwmc3,
'IP+CP+MC': ipcpmc3,
'IP+LW+MC': iplwmc3,
'CP+LW+MC': cplwmc3,
'All': all_3,
'None': none_3,
'PropReused': reused_2x3,
'PropSpecialized': special_2x3}

df3 = pd.DataFrame(mc_pairs3,columns=['Most Popular Cat','MC','IP+MC','CP+MC','LW+MC','IP+CP+MC','IP+LW+MC','CP+LW+MC','All','None','PropReused','PropSpecialized'])


ax7 = sns.kdeplot(ax=ax[3,0],data=df3, x="PropReused", hue="Most Popular Cat")
ax7.set_title('2x3')
ax7.legend_.remove()

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax8 = sns.kdeplot(ax=ax[3,1],data=df3, x="PropSpecialized", hue="Most Popular Cat")
#x4.legend_.remove()
#ax2.set_title('Specialization')
ax8.set_title('2x3')


#############################################
#SEQUENTIAL (2x5)
#############################################
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


ax8 = sns.kdeplot(ax=ax[4,0],data=df4, x="PropReused", hue="Most Popular Cat")
ax8.set_title('Sequential')
ax8.legend_.remove()

#sns.pairplot(df,hue='Most Popular')
#sns.pairplot(df,hue='Most Popular', diag_kind='kde',plot_kws={'alpha': 0.5, 's': 70, 'edgecolor': 'k'},
             #size = 1.2)

ax8 = sns.kdeplot(ax=ax[4,1],data=df3, x="PropSpecialized", hue="Most Popular Cat")
#x4.legend_.remove()
#ax2.set_title('Specialization')
ax8.set_title('Sequential')






plt.tight_layout()
plt.savefig("./Combined/4T_2x5/Figures/density_reuseNEW.png")
plt.show()
