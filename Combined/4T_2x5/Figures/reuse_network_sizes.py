import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

#reuse proportions calculated in figure_5_lesions.py

reused_2x5 = np.load("./Combined/4T_2x5/Data/reused_prop.npy")
reused_2x5_2 = np.load("./Combined/4T_2x5/Data/reused_prop_2.npy")
reused_2x5_3 = np.load("./Combined/4T_2x5/Data/reused_prop_3.npy")
reused_2x5_4 = np.load("./Combined/4T_2x5/Data/reused_prop_4.npy")
special_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")

num_reuse_2x5 = np.load("./Combined/4T_2x5/Data/neuron_nums_reuse.npy")
num_reuse_2_2x5 = np.load("./Combined/4T_2x5/Data/neuron_nums_reuse_2.npy")
num_reuse_3_2x5 = np.load("./Combined/4T_2x5/Data/neuron_nums_reuse_3.npy")
num_reuse_4_2x5 = np.load("./Combined/4T_2x5/Data/neuron_nums_reuse_4.npy")
num_special_2x5 = np.load("./Combined/4T_2x5/Data/neuron_nums_special.npy")


reused_2x3 = np.load("./Combined/4T_2x3/Data/reused_prop.npy")
reused_2x3_2 = np.load("./Combined/4T_2x3/Data/reused_prop_2.npy")
reused_2x3_3 = np.load("./Combined/4T_2x3/Data/reused_prop_3.npy")
reused_2x3_4 = np.load("./Combined/4T_2x3/Data/reused_prop_4.npy")
special_2x3 = np.load("./Combined/4T_2x3/Data/special_prop.npy")

num_reuse_2x3 = np.load("./Combined/4T_2x3/Data/neuron_nums_reuse.npy")
num_reuse_2_2x3 = np.load("./Combined/4T_2x3/Data/neuron_nums_reuse_2.npy")
num_reuse_3_2x3 = np.load("./Combined/4T_2x3/Data/neuron_nums_reuse_3.npy")
num_reuse_4_2x3 = np.load("./Combined/4T_2x3/Data/neuron_nums_reuse_4.npy")
num_special_2x3 = np.load("./Combined/4T_2x3/Data/neuron_nums_special.npy")

reused_2x10 = np.load("./Combined/4T_2x10/Data/reused_prop.npy")
reused_2x10_2 = np.load("./Combined/4T_2x10/Data/reused_prop_2.npy")
reused_2x10_3 = np.load("./Combined/4T_2x10/Data/reused_prop_3.npy")
reused_2x10_4 = np.load("./Combined/4T_2x10/Data/reused_prop_4.npy")
special_2x10 = np.load("./Combined/4T_2x10/Data/special_prop.npy")

num_reuse_2x10 = np.load("./Combined/4T_2x10/Data/neuron_nums_reuse.npy")
num_reuse_2_2x10 = np.load("./Combined/4T_2x10/Data/neuron_nums_reuse_2.npy")
num_reuse_3_2x10 = np.load("./Combined/4T_2x10/Data/neuron_nums_reuse_3.npy")
num_reuse_4_2x10 = np.load("./Combined/4T_2x10/Data/neuron_nums_reuse_4.npy")
num_special_2x10 = np.load("./Combined/4T_2x10/Data/neuron_nums_special.npy")

reused_2x20 = np.load("./Combined/4T_2x20/Data/reused_prop.npy")
reused_2x20_2 = np.load("./Combined/4T_2x20/Data/reused_prop_2.npy")
reused_2x20_3 = np.load("./Combined/4T_2x20/Data/reused_prop_3.npy")
reused_2x20_4 = np.load("./Combined/4T_2x20/Data/reused_prop_4.npy")
special_2x20 = np.load("./Combined/4T_2x20/Data/special_prop.npy")

num_reuse_2x20 = np.load("./Combined/4T_2x20/Data/neuron_nums_reuse.npy")
num_reuse_2_2x20 = np.load("./Combined/4T_2x20/Data/neuron_nums_reuse_2.npy")
num_reuse_3_2x20 = np.load("./Combined/4T_2x20/Data/neuron_nums_reuse_3.npy")
num_reuse_4_2x20 = np.load("./Combined/4T_2x20/Data/neuron_nums_reuse_4.npy")
num_special_2x20 = np.load("./Combined/4T_2x20/Data/neuron_nums_special.npy")

reused_3x5 = np.load("./Combined/4T_3x5/Data/reused_prop.npy")
reused_3x5_2 = np.load("./Combined/4T_3x5/Data/reused_prop_2.npy")
reused_3x5_3 = np.load("./Combined/4T_3x5/Data/reused_prop_3.npy")
reused_3x5_4 = np.load("./Combined/4T_3x5/Data/reused_prop_4.npy")
special_3x5 = np.load("./Combined/4T_3x5/Data/special_prop.npy")

num_reuse_3x5 = np.load("./Combined/4T_3x5/Data/neuron_nums_reuse.npy")
num_reuse_2_3x5 = np.load("./Combined/4T_3x5/Data/neuron_nums_reuse_2.npy")
num_reuse_3_3x5 = np.load("./Combined/4T_3x5/Data/neuron_nums_reuse_3.npy")
num_reuse_4_3x5 = np.load("./Combined/4T_3x5/Data/neuron_nums_reuse_4.npy")
#print(num_reuse_4_3x5)
num_special_3x5 = np.load("./Combined/4T_3x5/Data/neuron_nums_special.npy")

reused_3x10 = np.load("./Combined/4T_3x10/Data/reused_prop.npy")
reused_3x10_2 = np.load("./Combined/4T_3x10/Data/reused_prop_2.npy")
reused_3x10_3 = np.load("./Combined/4T_3x10/Data/reused_prop_3.npy")
reused_3x10_4 = np.load("./Combined/4T_3x10/Data/reused_prop_4.npy")
special_3x10 = np.load("./Combined/4T_3x10/Data/special_prop.npy")

num_reuse_3x10 = np.load("./Combined/4T_3x10/Data/neuron_nums_reuse.npy")
num_reuse_2_3x10 = np.load("./Combined/4T_3x10/Data/neuron_nums_reuse_2.npy")
num_reuse_3_3x10 = np.load("./Combined/4T_3x10/Data/neuron_nums_reuse_3.npy")
num_reuse_4_3x10 = np.load("./Combined/4T_3x10/Data/neuron_nums_reuse_4.npy")
num_special_3x10 = np.load("./Combined/4T_3x10/Data/neuron_nums_special.npy")

"""
colors = ['#E69F00', '#56B4E9','#009E73','#D55E00']
names = ['2x5','2x10','2x20','2x3']

plt.figure(figsize=(6,4))
plt.hist([reused_2x5,reused_2x3,reused_2x10,reused_2x20], bins = int(1/.1), color = colors, label=names)

plt.legend()
plt.xlabel('Proportion reused')
plt.ylabel('# of agents')
plt.title('Neural Reuse due to Lesions')
#plt.savefig("./Combined/4T_2x10/Figures/figure_2_distributions.pdf")
plt.show()

plt.figure(figsize=(6,4))
plt.hist([special_2x5,special_2x3,special_2x10,special_2x20], bins = int(1/.1), color = colors, label=names)

plt.legend()
plt.xlabel('Proportion Specialized')
plt.ylabel('# of agents')
plt.title('Neural Specialization due to Lesions')
#plt.savefig("./Combined/4T_2x10/Figures/figure_2_distributions.pdf")
plt.show()
"""
data = []
data2 = []
data3 = []
data4 = []
data.append([reused_2x3,special_2x3])
data2.append([reused_2x5,special_2x5])
data3.append([reused_2x10,special_2x10])
data4.append([reused_2x20,special_2x20])
"""
data = {'name': ['Prop Reused','Prop Specialized'],
'2x3': [reused_2x3,special_2x3],
'2x5': [reused_2x5,special_2x5],
'2x10': [reused_2x10,special_2x10],
'2x20': [reused_2x20,special_2x20]}
"""
"""
df = pd.DataFrame({'Exp': ['2x3','2x5','2x10','2x20'],
'Prop Reused': [reused_2x3,reused_2x5,reused_2x10,reused_2x20],
'Prop Specialized': [special_2x3,special_2x5,special_2x10,special_2x20]})

"""
size3 = []
size5 = []
size10 = []
size20 = []
for r in list(reused_2x3):
    size3.append('2x3')
for r in list(reused_2x5):
    size5.append('2x5')
for r in list(reused_2x10):
    size10.append('2x10')
for r in list(reused_2x20):
    size20.append('2x20')



dfnew = pd.DataFrame({"Reused": [sum(reused_2x3)/len(reused_2x3),sum(reused_2x5)/len(reused_2x5),sum(reused_2x10)/len(reused_2x10),sum(reused_2x20)/len(reused_2x20)],
"Reused,2": [sum(reused_2x3_2)/len(reused_2x3_2),sum(reused_2x5_2)/len(reused_2x5_2),sum(reused_2x10_2)/len(reused_2x10_2),sum(reused_2x20_2)/len(reused_2x20_2)],
"Reused,3": [sum(reused_2x3_3)/len(reused_2x3_3),sum(reused_2x5_3)/len(reused_2x5_3),sum(reused_2x10_3)/len(reused_2x10_3),sum(reused_2x20_3)/len(reused_2x20_3)],
"Reused,4": [sum(reused_2x3_4)/len(reused_2x3_4),sum(reused_2x5_4)/len(reused_2x5_4),sum(reused_2x10_4)/len(reused_2x10_4),sum(reused_2x20_4)/len(reused_2x20_4)],
"Specialized": [sum(special_2x3)/len(special_2x3),sum(special_2x5)/len(special_2x5),sum(special_2x10)/len(special_2x10),sum(special_2x20)/len(special_2x20)],
"NetworkSize": ["2x3","2x5","2x10","2x20"]})
index=(['2x3'],['2x5'],['2x10'],['2x20'])

df = pd.DataFrame({"Reused": [reused_2x3,reused_2x5,reused_2x10,reused_2x20,reused_3x10,reused_3x10],
"Reused,2": [reused_2x3_2,reused_2x5_2,reused_2x10_2,reused_2x20_2,reused_3x5_2,reused_3x10_2],
"Reused,3": [reused_2x3_3,reused_2x5_3,reused_2x10_3,reused_2x20_3,reused_3x5_3,reused_3x10_3],
"Reused,4": [reused_2x3_4,reused_2x5_4,reused_2x10_4,reused_2x20_4,reused_3x5_4,reused_3x10_4],
"Specialized": [special_2x3,special_2x5,special_2x10,special_2x20,special_3x5,special_3x10],
"NetworkSize": ["2x3","2x5","2x10","2x20","3x5","3x10"]},
index=['2x3','2x5','2x10','2x20','3x5','3x10'])
#print(df)

dfnum = pd.DataFrame({"Reused": [np.average(num_reuse_2x3),np.average(num_reuse_2x5),np.average(num_reuse_2x10),np.average(num_reuse_2x20),np.average(num_reuse_3x5),np.average(num_reuse_3x10)],
"Reused,2": [np.average(num_reuse_2_2x3),np.average(num_reuse_2_2x5),np.average(num_reuse_2_2x10),np.average(num_reuse_2_2x20),np.average(num_reuse_2_3x5),np.average(num_reuse_2_3x10)],
"Reused,3": [np.average(num_reuse_3_2x3),np.average(num_reuse_3_2x5),np.average(num_reuse_3_2x10),np.average(num_reuse_3_2x20),np.average(num_reuse_3_3x5),np.average(num_reuse_3_3x10)],
"Reused,4": [np.average(num_reuse_4_2x3),np.average(num_reuse_4_2x5),np.average(num_reuse_4_2x10),np.average(num_reuse_4_2x20),np.average(num_reuse_4_3x5),np.average(num_reuse_4_3x10)],
"Specialized": [np.average(num_special_2x3),np.average(num_special_2x5),np.average(num_special_2x10),np.average(num_special_2x20),np.average(num_special_3x5),np.average(num_special_3x10)],
"NetworkSize": ["2x3","2x5","2x10","2x20","3x5","3x10"]},
index=['2x3','2x5','2x10','2x20','3x5','3x10'])



ax = dfnum.plot.bar(rot=0, stacked=True)
plt.ylabel("Number of Neurons")
plt.title("4-Task Agents: Reuse Categories")
plt.savefig("C:/Users/benso/Desktop/Projects/Neural_Reuse/Neural_Reuse_New/Combined/4T_2x5/Figures/reuse_categories_networks.png")
plt.show()








"""
dfnum[['Reused','Reused,2','Reused,3','Reused,4','Specialized']].applymap(lambda x: x[0]).plot.bar(rot=0)
plt.title('# Neurons in Reuse Categories')
plt.xlabel('Experiment')
plt.ylabel('Proportions')
#plt.savefig("./Combined/4T_2x5/Figures/REUSE_extent.pdf")
plt.show()




fig, ax = plt.subplots(2, 1, figsize=(6, 5))

dfnum[['Reused','Reused,2','Reused,3','Reused,4','Specialized']].applymap(lambda x: x[0]).plot.bar(ax=ax[0],rot=0)
ax[0].set_title('# of neurons per reuse category')

df[['Reused','Reused,2','Reused,3','Reused,4','Specialized']].applymap(lambda x: x[0]).plot.bar(ax=ax[1],rot=0)
ax[1].set_title('Proportion of neurons per reuse category')
ax[1].get_legend().remove()
plt.tight_layout()
plt.savefig("./Combined/4T_2x5/Figures/reuse_categories_networks.pdf")
plt.show()
"""     