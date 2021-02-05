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

reused_2x3 = np.load("./Combined/4T_2x3/Data/reused_prop.npy")
reused_2x3_2 = np.load("./Combined/4T_2x3/Data/reused_prop_2.npy")
reused_2x3_3 = np.load("./Combined/4T_2x3/Data/reused_prop_3.npy")
reused_2x3_4 = np.load("./Combined/4T_2x3/Data/reused_prop_4.npy")
special_2x3 = np.load("./Combined/4T_2x3/Data/special_prop.npy")

reused_2x10 = np.load("./Combined/4T_2x10/Data/reused_prop.npy")
reused_2x10_2 = np.load("./Combined/4T_2x10/Data/reused_prop_2.npy")
reused_2x10_3 = np.load("./Combined/4T_2x10/Data/reused_prop_3.npy")
reused_2x10_4 = np.load("./Combined/4T_2x10/Data/reused_prop_4.npy")
special_2x10 = np.load("./Combined/4T_2x10/Data/special_prop.npy")

reused_2x20 = np.load("./Combined/4T_2x20/Data/reused_prop.npy")
reused_2x20_2 = np.load("./Combined/4T_2x20/Data/reused_prop_2.npy")
reused_2x20_3 = np.load("./Combined/4T_2x20/Data/reused_prop_3.npy")
reused_2x20_4 = np.load("./Combined/4T_2x20/Data/reused_prop_4.npy")
special_2x20 = np.load("./Combined/4T_2x20/Data/special_prop.npy")
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

data = {'name': ['Prop Reused','Prop Specialized'],
'2x3': [reused_2x3,special_2x3],
'2x5': [reused_2x5,special_2x5],
'2x10': [reused_2x10,special_2x10],
'2x20': [reused_2x20,special_2x20]}

#df = pd.DataFrame({'Exp': ['2x3','2x5','2x10','2x20'],
#'Prop Reused': [reused_2x3,reused_2x5,reused_2x10,reused_2x20],
#'Prop Specialized': [special_2x3,special_2x5,special_2x10,special_2x20]})

df = pd.DataFrame({'Reused': [reused_2x3,reused_2x5,reused_2x10,reused_2x20],
'Reused,2': [reused_2x3_2,reused_2x5_2,reused_2x10_2,reused_2x20_2],
'Reused,3': [reused_2x3_3,reused_2x5_3,reused_2x10_3,reused_2x20_3],
'Reused,4': [reused_2x3_4,reused_2x5_4,reused_2x10_4,reused_2x20_4],
'Specialized': [special_2x3,special_2x5,special_2x10,special_2x20]},
index=(['2x3'],['2x5'],['2x10'],['2x20']))

#print(df)
#sns.pairplot(df)
#plt.show()


#df_lists = df[['Prop Reused','Prop Specialized']].unstack().apply(pd.Series)
#df_lists.plot.bar(rot=0, cmap=plt.cm.jet, fontsize=8, width=0.7, figsize=(8,4))

#, color=list('br')
#SHOWS DISCRETE CATEGORIES OF REUSE AND SPECIALIZATION
df[['Reused','Reused,2','Reused,3','Reused,4','Specialized']].applymap(lambda x: x[0]).plot.bar(rot=0)
plt.title('Reuse Proportions')
plt.xlabel('Experiment')
plt.ylabel('Proportions')
plt.savefig("./Combined/4T_2x5/Figures/REUSE_extent.pdf")
plt.show()


#SHOWS MULTIPLE CATEGORIES OF REUSE AND SPECIALIZATION (2 neurons reused, 3 neurons reused, 4)