import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

dir = 'Eduardo/Multi_Data'

dir1 = 'Eduardo/Ind_Data'

files = glob.glob(os.path.join(dir, "perf_MCLW5_MC*.npy"))
files.sort()

files1 = glob.glob(os.path.join(dir, "perf_MCLW5_LW*.npy"))
files1.sort()

files2 = glob.glob(os.path.join(dir1, "perf_MC5_MC*.npy"))
files2.sort()

files3 = glob.glob(os.path.join(dir1, "perf_LW5_LW*.npy"))
files3.sort()

MCmult = []
LWmult = []
MCind = []
LWind = []

for i,y,x,z in zip(files,files1,files2,files3):
    fits = np.load(i)
    #plt.plot(MCmult,fits,'bo',label='MCLW_MC')
    MCmult.append(fits)

    fits2 = np.load(x)
    MCind.append(fits2)
    
    #plt.plot(MCind,fits2,'b*',label='Ind_MC')

    fits1 = np.load(y)
    #plt.plot(LWmult,fits1,'ro',label='MCLW_LW')
    LWmult.append(fits1)

    fits3 = np.load(z)
    #print(fits3)
    #plt.plot(LWind,fits3,'r*',label='Ind_LW') 
    LWind.append(fits3)

data = {
    'MC_Multi': MCmult,
    'MC_Ind': MCind,
    'LW_Multi': LWmult,
    'LW_Ind': LWind
}

df = pd.DataFrame(data)


sns.swarmplot(x="variable", y="value", data=df.melt(),size=4)
plt.xlabel("Condition")
plt.ylabel("Performance")
plt.axvline(x=1.5,color='k', linestyle='--')

plt.title("Task performance on individual vs multifunctional networks")
plt.show()

