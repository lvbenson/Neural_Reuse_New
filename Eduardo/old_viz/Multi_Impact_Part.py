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




import math

#Multifunctional - impact

fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[0].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0].plot(np.mean(b5xmc,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[0].fill_between(x5, np.mean(b5xmc,axis=0)-(np.divide(err5mc,math.sqrt(10))), np.mean(b5xmc,axis=0)+(np.divide(err5mc,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[0].set_title("Impact")
axs[0].legend()
axs[0].set_xlabel("Neuron (sorted)")

v5x = np.zeros((reps,nn5))
nI = 4
nH = 10

for i in range(reps):
    nn = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    v5x[i] = nn
    #nn = nn.T[4:-1]
    #nn = np.mean(np.abs(np.diff(nn)),axis=1)
    #nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    #v5x[i] = nn/max


mcv5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    mcv5x[i] = nn
    #nn = nn.T[4:-1]
    #nn = np.mean(np.abs(np.diff(nn)),axis=1)
    #nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    #mcv5x[i] = nn/max

#clean this up: load and sort data altogether, plot separately
#one script per *perfected* figure

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


print('single',np.mean(v5x,axis=0))
print('double',np.mean(comb,axis=0))

axs[1].plot(np.mean(v5x,axis=0),'o-',markersize=2,label="MCLW_LW",color='#3F7F4C')
axs[1].fill_between(x5, np.mean(v5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(v5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1].plot(np.mean(mcv5x,axis=0),'o-',markersize=2,label="MCLW_MC",color='#CC4F1B')
axs[1].fill_between(x5, np.mean(mcv5x,axis=0)-(np.divide(mcerr5,math.sqrt(10))), np.mean(mcv5x,axis=0)+(np.divide(mcerr5,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[1].plot(np.mean(comb,axis=0),'o-',markersize=2,label="MCLW",color='#507ad0')
axs[1].fill_between(x5, np.mean(comb,axis=0)-(np.divide(comberr,math.sqrt(10))), np.mean(comb,axis=0)+(np.divide(comberr,math.sqrt(10))),alpha=0.2, edgecolor='#507ad0', facecolor='#507ad0')

axs[1].set_title("Participation")
axs[1].legend()




fig.suptitle("Size: 2x5, Pairwise Multifunctional: MCLW")
plt.savefig("Eduardo/PairwiseMulti_2x5.png")
plt.show()
