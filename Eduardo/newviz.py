import numpy as np
import matplotlib.pyplot as plt

"""
Individual tasks1A. 
How are resources allocated in a circuit solving a task? 
Do some tasks use more resources than other ones? 
How does the circuit solve this task? 
(how many neurons are necessary and how many are participating) 
Figure: Impact vs neuron sorted (LW, MC)  
Figure: Participation vs neuron sorted.
"""

############################################################
#Individual Tasks
#########################################################

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Data/lesions_LW5_40_"+str(i)+".npy"))
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))

##############################
#MC
##############################

b5xmc = np.zeros((reps,nn5))
b5xerrsmc = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Data/lesions_MC5_40_"+str(i)+".npy"))
    b5xmc[i] = np.sort(f)[::-1]
    b5xerrsmc[i] = np.std(f)


x5 = list(range(0,10))
err5mc = []
for i in b5xmc.T:
    err5mc.append(np.std(i))




import math

#Individual - impact

fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="LW",color='#3F7F4C')
axs[0].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0].plot(np.mean(b5xmc,axis=0),'o-',markersize=2,label="MC",color='#CC4F1B')
axs[0].fill_between(x5, np.mean(b5xmc,axis=0)-(np.divide(err5mc,math.sqrt(10))), np.mean(b5xmc,axis=0)+(np.divide(err5mc,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[0].set_title("Impact")
axs[0].legend()
axs[0].set_xlabel("Neuron (sorted)")

#Individual - participation

#LW

v5x = np.zeros((reps,nn5))
nI = 4
nH = 10

for i in range(reps):
    nn = np.load("Eduardo/Data/state_LW5_"+str(i)+".npy")
    #print(len(nn))
    #variance
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v5x[i] = nn/max
    #print('lw',v5x[i])

mcv5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Data/state_MC5_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv5x[i] = nn/max
    #print('mc',mcv5x[i])


err5 = []

for i in v5x.T:
    err5.append(np.std(i))

mcerr5 = []

for i in mcv5x.T:
    mcerr5.append(np.std(i))


axs[1].plot(np.mean(v5x,axis=0),'o-',markersize=2,label="LW",color='#3F7F4C')
axs[1].fill_between(x5, np.mean(v5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(v5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1].plot(np.mean(mcv5x,axis=0),'o-',markersize=2,label="MC",color='#CC4F1B')
axs[1].fill_between(x5, np.mean(mcv5x,axis=0)-(np.divide(mcerr5,math.sqrt(10))), np.mean(mcv5x,axis=0)+(np.divide(mcerr5,math.sqrt(10))),alpha=0.2, edgecolor='#c88700', facecolor='#c88700')

axs[1].set_title("Participation")
axs[1].legend()

fig.suptitle("Size: 2x5, Individual Tasks")
plt.savefig("Eduardo/Individual_2x5.png")
plt.show()




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
    f = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
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
    f = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy"))
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

#multifunctional - participation

#MCLW_LW

v5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v5x[i] = nn/max
    #nn = nn.T[4:-1]
    #nn = np.mean(np.abs(np.diff(nn)),axis=1)
    #nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    #v5x[i] = nn/max


mcv5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW5_MC_"+str(i)+".npy")
    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv5x[i] = nn/max
    #nn = nn.T[4:-1]
    #nn = np.mean(np.abs(np.diff(nn)),axis=1)
    #nn = np.sort(nn)[::-1]
    #max = np.max(nn)
    #mcv5x[i] = nn/max


comb = np.zeros((reps,nn5))

for i in range(reps):
    mc = np.load("Eduardo/Data3/state_MCLW5_MC_"+str(i)+".npy")
    lw = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
    nn = np.concatenate((mc, lw), axis=0)
    #print(len(mc))


    nn = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    comb[i] = nn/max
    



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




############################################################
#1C. XXX 
#Figure: Histogram. Distance of impact. I_MCLW_LW – I_ MCLW_MC.
#Figure: Histogram. Distance between participation. P_MCLW_LW – P_ MCLW_MC.
##############################################################
#HISTOGRAMS

impact5 = np.zeros((reps,nn5))
part5 = np.zeros((reps,nn5))

newimpact = []
#impact histogram

for i in range(reps):
    f1 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    f2 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy"))

    #print('before',len(f1))
    for p, (x,y) in enumerate(zip(f1,f2)):
        if x < 0.75 and y < 0.75:
            #print(x)
            #indx = np.where(f1 == x)
            #print(indx)
            f1[p] = 200
            f2[p] = 100

    impact5[i] = (f1-f2)
    imp = impact5[i]
    imp = imp[imp != 100]
    newimpact.append(imp)
    


#print(newimpact)
#arr = np.array(newimpact, dtype="object")
#print(arr[arr != 2])

fig, axs = plt.subplots(2, sharex=True, sharey=True, constrained_layout=True)
#newimpact = np.array(newimpact)
flat_list = [item for sublist in newimpact for item in sublist]

axs[0].hist(flat_list,100,density=False,alpha=0.5,label="2x5")
axs[0].set_title("Distance of Impact")

#participation histogram

newpart = []
for i in range(reps):
    nnlw = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
    nnlw = nnlw.T[4:-1]
    nnlw = np.mean(np.abs(np.diff(nnlw)),axis=1)
    nnlw = np.sort(nnlw)[::-1]
    maxlw = np.max(nnlw)
    f1 = nnlw/maxlw

    nnmc = np.load("Eduardo/Data3/state_MCLW5_MC_"+str(i)+".npy")
    nnmc = nnmc.T[4:-1]
    nnmc = np.mean(np.abs(np.diff(nnmc)),axis=1)
    nnmc = np.sort(nnmc)[::-1]
    maxmc = np.max(nnmc)
    f2 = nnmc/maxmc

    for p, (x,y) in enumerate(zip(f1,f2)):
        if x < 0.75 and y < 0.75:
            f1[p] = 200
            f2[p] = 100
            #print(x)

    part5[i] = f1 - f2
    partimp = part5[i]
    partimp = partimp[partimp != 100]
    newpart.append(partimp)

flat_list_part = [item for sublist in newpart for item in sublist]
#print(len(flat_list_part))

axs[1].hist(flat_list_part,100,density=False,alpha=0.5,label="2x5")
axs[1].set_title("Distance of Participation")

fig.suptitle("Size: 2x5, Pairwise Multifunctional: MCLW")
plt.savefig("Eduardo/Hist_PairwiseMulti_2x5.png")
plt.show()