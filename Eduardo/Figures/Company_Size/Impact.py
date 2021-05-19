
"""
Given the set of company projects and a fixed total employee count, how many important employees are required for the success of those projects?
Importance defined as: removing the employee would cause project failure to some extent
Options
Option 0 -- None of the employees are important 

Option 1 -- All of the employees are important

Option 2 -- a very small fraction of the employees are important

Option 3 -- a large fraction of the employees are important
Hypothesis
Hypothesis for what we expect to see as company size goes from small to large
Hypothesis 1 -- the option to choose depends on the company size
	Hypothesis 1a -- small company will go for option 1
	Hypothesis 1b -- large company will go for option 2
"""

#participation for MCLW for 2x3, 2x5

import numpy as np
import matplotlib.pyplot as plt

"""
Multifunctional 1B. 
XXX --- Figure: Impact vs neuron sorted (MCLW_LW, MCLW_MC)  
Figure: Participation vs neuron sorted.
"""

reps = 100
nn5 = 2*5
################################
#MCLW_LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))

##############################
#MCLW_MC
##############################

b5xmc = np.zeros((reps,nn5))
b5xerrsmc = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))
    b5xmc[i] = np.sort(f)[::-1]
    b5xerrsmc[i] = np.std(f)


x5 = list(range(0,10))
err5mc = []
for i in b5xmc.T:
    err5mc.append(np.std(i))

################################
#MCLW - COMBINED
################################

combim = np.zeros((reps,nn5))
combimerr = np.zeros((reps,nn5))


for i in range(reps):
    mc = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))
    lw = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    nn = mc+lw
    #print(len(mc))


    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    combim[i] = nn
    combimerr[i] = np.std(f)


x5 = list(range(0,10))
combimerr = []
for i in combim.T:
    combimerr.append(np.std(i))


import math

#Multifunctional - impact

fig, axs = plt.subplots(2,2, constrained_layout=True)
axs[0,0].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[0,0].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0,0].plot(np.mean(b5xmc,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[0,0].fill_between(x5, np.mean(b5xmc,axis=0)-(np.divide(err5mc,math.sqrt(10))), np.mean(b5xmc,axis=0)+(np.divide(err5mc,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')
"""
axs[0].plot(np.mean(combim,axis=0),'o-',markersize=2,label="MCLW_COMB",color='#507ad0')
axs[0].fill_between(x5, np.mean(combim,axis=0)-(np.divide(combimerr,math.sqrt(10))), np.mean(combim,axis=0)+(np.divide(combimerr,math.sqrt(10))),alpha=0.2, edgecolor='#507ad0', facecolor='#507ad0')
"""

axs[0,0].set_title("2x5: Impact")
axs[0,0].legend()
axs[0,0].set_xlabel("Neuron (sorted)")

v5x = np.zeros((reps,nn5))
nI = 4
nH = 10

######################################
#LW
#####################################

for i in range(reps):
    nn = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    v5x[i] = nn

#################################
#MC
################################

mcv5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    mcv5x[i] = nn


##################################
#COMBINED
##################################

comb = np.zeros((reps,nn5))

for i in range(reps):
    mc = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
    lw = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    nn = np.concatenate((mc, lw), axis=0)
    #print(len(mc))


    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    comb[i] = nn
    



err5 = []

for i in v5x.T:
    err5.append(np.std(i))

mcerr5 = []

for i in mcv5x.T:
    mcerr5.append(np.std(i))

comberr = []
for i in comb.T:
    comberr.append(np.std(i))


#print('single',np.mean(v5x,axis=0))
#print('double',np.mean(comb,axis=0))

axs[1,0].plot(np.mean(v5x,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[1,0].fill_between(x5, np.mean(v5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(v5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1,0].plot(np.mean(mcv5x,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[1,0].fill_between(x5, np.mean(mcv5x,axis=0)-(np.divide(mcerr5,math.sqrt(10))), np.mean(mcv5x,axis=0)+(np.divide(mcerr5,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[1,0].plot(np.mean(comb,axis=0),'o-',markersize=2,label="MCLW",color='#507ad0')
axs[1,0].fill_between(x5, np.mean(comb,axis=0)-(np.divide(comberr,math.sqrt(10))), np.mean(comb,axis=0)+(np.divide(comberr,math.sqrt(10))),alpha=0.2, edgecolor='#507ad0', facecolor='#507ad0')

axs[1,0].set_title("Participation")
axs[1,0].legend()


#########################################################################
#2x3
#########################################################################


reps = 100
nn3 = 2*3
################################
#MCLW_LW, 2x3
###############################

b3xlw = np.zeros((reps,nn3))
b3xerrslw = np.zeros((reps,nn3))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Data3/lesions_MCLW3_LW_40_"+str(i)+".npy"))
    b3xlw[i] = np.sort(f)[::-1]
    b3xerrslw[i] = np.std(f)


x3 = list(range(0,6))
err3 = []
for i in b3xlw.T:
    err3.append(np.std(i))

##############################
#MCLW_MC
##############################

b3xmc = np.zeros((reps,nn3))
b3xerrsmc = np.zeros((reps,nn3))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Data3/lesions_MCLW3_MC_40_"+str(i)+".npy"))
    b3xmc[i] = np.sort(f)[::-1]
    b3xerrsmc[i] = np.std(f)


x3 = list(range(0,6))
err3mc = []
for i in b3xmc.T:
    err3mc.append(np.std(i))
"""
################################
#MCLW - COMBINED
################################

combim = np.zeros((reps,nn5))
combimerr = np.zeros((reps,nn5))


for i in range(reps):
    mc = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))
    lw = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    nn = mc+lw
    #print(len(mc))


    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    combim[i] = nn
    combimerr[i] = np.std(f)


x5 = list(range(0,10))
combimerr = []
for i in combim.T:
    combimerr.append(np.std(i))
"""

import math

#Multifunctional - impact

axs[0,1].plot(np.mean(b3xlw,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[0,1].fill_between(x3, np.mean(b3xlw,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(b3xlw,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0,1].plot(np.mean(b3xmc,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[0,1].fill_between(x3, np.mean(b3xmc,axis=0)-(np.divide(err3mc,math.sqrt(6))), np.mean(b3xmc,axis=0)+(np.divide(err3mc,math.sqrt(6))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')
"""
axs[0].plot(np.mean(combim,axis=0),'o-',markersize=2,label="MCLW_COMB",color='#507ad0')
axs[0].fill_between(x5, np.mean(combim,axis=0)-(np.divide(combimerr,math.sqrt(10))), np.mean(combim,axis=0)+(np.divide(combimerr,math.sqrt(10))),alpha=0.2, edgecolor='#507ad0', facecolor='#507ad0')
"""

axs[0,1].set_title("2x3: Impact")
axs[0,1].legend()
axs[0,1].set_xlabel("Neuron (sorted)")

v3x = np.zeros((reps,nn3))
nI = 4
nH = 6

######################################
#LW
#####################################

for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW3_LW_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v3x[i] = nn/max

#################################
#MC
################################

mcv3x = np.zeros((reps,nn3))

for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW3_MC_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv3x[i] = nn/max


##################################
#COMBINED
##################################

comb = np.zeros((reps,nn3))

for i in range(reps):
    mc = np.load("Eduardo/Data3/state_MCLW3_MC_"+str(i)+".npy")
    lw = np.load("Eduardo/Data3/state_MCLW3_LW_"+str(i)+".npy")
    nn = np.concatenate((mc, lw), axis=0)
    #print(len(mc))


    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    comb[i] = nn/max
    



err3 = []

for i in v3x.T:
    err3.append(np.std(i))

mcerr3 = []

for i in mcv3x.T:
    mcerr3.append(np.std(i))

comberr = []
for i in comb.T:
    comberr.append(np.std(i))



axs[1,1].plot(np.mean(v3x,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[1,1].fill_between(x3, np.mean(v3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(v3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1,1].plot(np.mean(mcv3x,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[1,1].fill_between(x3, np.mean(mcv3x,axis=0)-(np.divide(mcerr3,math.sqrt(6))), np.mean(mcv3x,axis=0)+(np.divide(mcerr3,math.sqrt(6))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[1,1].plot(np.mean(comb,axis=0),'o-',markersize=2,label="MCLW",color='#507ad0')
axs[1,1].fill_between(x3, np.mean(comb,axis=0)-(np.divide(comberr,math.sqrt(6))), np.mean(comb,axis=0)+(np.divide(comberr,math.sqrt(10))),alpha=0.2, edgecolor='#507ad0', facecolor='#507ad0')

axs[1,1].set_title("2x3: Participation (normalized)")
axs[1,1].legend()




fig.suptitle("Sizes 2x3, 2x5: Pairwise Multifunctional: MCLW")
plt.savefig("Eduardo/Figures/Company_Size/PairwiseMulti_2x5_2x3.png")
plt.show()