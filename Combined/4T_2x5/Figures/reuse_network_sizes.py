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


dfnew = pd.DataFrame({"Reused2x3": list(reused_2x3),
"Reused2x5":list(reused_2x5),
"Reused2x10":list(reused_2x10),
"Reused2x20":list(reused_2x20)})

"""
dfnew = pd.DataFrame({"Reused": [sum(reused_2x3)/len(reused_2x3),sum(reused_2x5)/len(reused_2x5),sum(reused_2x10)/len(reused_2x10),sum(reused_2x20)/len(reused_2x20)],
"Reused,2": [sum(reused_2x3_2)/len(reused_2x3_2),sum(reused_2x5_2)/len(reused_2x5_2),sum(reused_2x10_2)/len(reused_2x10_2),sum(reused_2x20_2)/len(reused_2x20_2)],
"Reused,3": [sum(reused_2x3_3)/len(reused_2x3_3),sum(reused_2x5_3)/len(reused_2x5_3),sum(reused_2x10_3)/len(reused_2x10_3),sum(reused_2x20_3)/len(reused_2x20_3)],
"Reused,4": [sum(reused_2x3_4)/len(reused_2x3_4),sum(reused_2x5_4)/len(reused_2x5_4),sum(reused_2x10_4)/len(reused_2x10_4),sum(reused_2x20_4)/len(reused_2x20_4)],
"Specialized": [sum(special_2x3)/len(special_2x3),sum(special_2x5)/len(special_2x5),sum(special_2x10)/len(special_2x10),sum(special_2x20)/len(special_2x20)],
"NetworkSize": ["2x3","2x5","2x10","2x20"]})
#index=(['2x3'],['2x5'],['2x10'],['2x20'])
"""

print(dfnew)
#dfnew.to_csv("./Combined/4T_2x5/Data/dataframe.csv")
#df = pd.read_csv("./Combined/4T_2x5/Data/dataframe.csv")
import joypy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm


"""
df[['Reused','Reused,2','Reused,3','Reused,4','Specialized']].applymap(lambda x: x[0]).plot.bar(rot=0)
plt.title('Reuse Proportions')
plt.xlabel('Experiment')
plt.ylabel('Proportions')
plt.savefig("./Combined/4T_2x5/Figures/REUSE_extent.pdf")
plt.show()
"""


#plt.figure(figsize=(10,8), dpi= 50)
fig, axes = joypy.joyplot(dfnew, ylim='own', figsize=(4,5))

# Decoration
#plt.title('Reuse Extent Over Multiple Network Sizes', fontsize=14)
plt.show()
#SHOWS MULTIPLE CATEGORIES OF REUSE AND SPECIALIZATION (2 neurons reused, 3 neurons reused, 4)