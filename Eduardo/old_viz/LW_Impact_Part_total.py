#plot LW individual with LW multi together for each impact and participation
import numpy as np 
import matplotlib.pyplot as plt
import math 

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Ind_Data/lesions_LW5_LW_40_"+str(i)+".npy"))
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))


reps = 100
nn5 = 2*5
################################
#MCLW_LW, 2x5
###############################

b5xlwm = np.zeros((reps,nn5))
b5xerrslwm = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    b5xlwm[i] = np.sort(f)[::-1]
    b5xerrslwm[i] = np.std(f)


x5 = list(range(0,10))
err5m = []
for i in b5xlwm.T:
    err5m.append(np.std(i))


#Plot


fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="LW_Ind",color='#ff9269')
axs[0].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#ff9269', facecolor='#ff9269')

axs[0].plot(np.mean(b5xlwm,axis=0),'o-',markersize=2,label="LW_Multi",color='#ff0000')
axs[0].fill_between(x5, np.mean(b5xlwm,axis=0)-(np.divide(err5m,math.sqrt(10))), np.mean(b5xlwm,axis=0)+(np.divide(err5m,math.sqrt(10))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')

axs[0].set_title("LW: Ind vs Multi")
axs[0].legend()
axs[0].set_xlabel("Neuron (sorted)")



#MC

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Ind_Data/lesions_MC5_MC_40_"+str(i)+".npy"))
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))


reps = 100
nn5 = 2*5
################################
#MCLW_LW, 2x5
###############################

b5xlwm = np.zeros((reps,nn5))
b5xerrslwm = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))
    b5xlwm[i] = np.sort(f)[::-1]
    b5xerrslwm[i] = np.std(f)


x5 = list(range(0,10))
err5m = []
for i in b5xlwm.T:
    err5m.append(np.std(i))


#Plot


#fig, axs = plt.subplots(2, constrained_layout=True)
axs[1].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="MC_Ind",color='#02b7dd')
axs[1].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#02b7dd', facecolor='#02b7dd')

axs[1].plot(np.mean(b5xlwm,axis=0),'o-',markersize=2,label="MC_Multi",color='#022dfe')
axs[1].fill_between(x5, np.mean(b5xlwm,axis=0)-(np.divide(err5m,math.sqrt(10))), np.mean(b5xlwm,axis=0)+(np.divide(err5m,math.sqrt(10))),alpha=0.2, edgecolor='#022dfe', facecolor='#022dfe')

axs[1].set_title("MC: Ind vs Multi")
axs[1].legend()
axs[1].set_xlabel("Neuron (sorted)")


fig.suptitle("2-Task Impact: 2x5")
plt.savefig("Eduardo/LWMC_Impact_2x5.png")
plt.show()

#####################################################################################################
#plot LW individual with LW multi together for each impact and participation
import numpy as np 
import matplotlib.pyplot as plt
import math 

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))
nI = 4
nH = 10

for i in range(reps):
    #f = 1 - (np.load("Eduardo/Ind_Data/lesions_LW5_LW_40_"+str(i)+".npy"))
    f = np.load("Eduardo/Ind_Data/state_LW5_LW_"+str(i)+".npy")
    f = np.var(f[:,nI:nI+nH],axis=0)
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))


reps = 100
nn5 = 2*5
################################
#MCLW_LW, 2x5
###############################

b5xlwm = np.zeros((reps,nn5))
b5xerrslwm = np.zeros((reps,nn5))

for i in range(reps):
    #f = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    f = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    f = np.var(f[:,nI:nI+nH],axis=0)
    b5xlwm[i] = np.sort(f)[::-1]
    b5xerrslwm[i] = np.std(f)


x5 = list(range(0,10))
err5m = []
for i in b5xlwm.T:
    err5m.append(np.std(i))


#Plot


fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="LW_Ind",color='#ff9269')
axs[0].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#ff9269', facecolor='#ff9269')

axs[0].plot(np.mean(b5xlwm,axis=0),'o-',markersize=2,label="LW_Multi",color='#ff0000')
axs[0].fill_between(x5, np.mean(b5xlwm,axis=0)-(np.divide(err5m,math.sqrt(10))), np.mean(b5xlwm,axis=0)+(np.divide(err5m,math.sqrt(10))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')

axs[0].set_title("LW: Ind vs Multi")
axs[0].legend()
axs[0].set_xlabel("Neuron (sorted)")



#MC

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))
b5xerrslw = np.zeros((reps,nn5))

for i in range(reps):
    f = np.load("Eduardo/Ind_Data/state_MC5_MC_"+str(i)+".npy")
    f = np.var(f[:,nI:nI+nH],axis=0)
    b5xlw[i] = np.sort(f)[::-1]
    b5xerrslw[i] = np.std(f)


x5 = list(range(0,10))
err5 = []
for i in b5xlw.T:
    err5.append(np.std(i))


reps = 100
nn5 = 2*5
################################
#MCLW_LW, 2x5
###############################

b5xlwm = np.zeros((reps,nn5))
b5xerrslwm = np.zeros((reps,nn5))

for i in range(reps):
    f = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
    f = np.var(f[:,nI:nI+nH],axis=0)
    b5xlwm[i] = np.sort(f)[::-1]
    b5xerrslwm[i] = np.std(f)


x5 = list(range(0,10))
err5m = []
for i in b5xlwm.T:
    err5m.append(np.std(i))


#Plot


#fig, axs = plt.subplots(2, constrained_layout=True)
axs[1].plot(np.mean(b5xlw,axis=0),'o-',markersize=2,label="MC_Ind",color='#02b7dd')
axs[1].fill_between(x5, np.mean(b5xlw,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5xlw,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#02b7dd', facecolor='#02b7dd')

axs[1].plot(np.mean(b5xlwm,axis=0),'o-',markersize=2,label="MC_Multi",color='#022dfe')
axs[1].fill_between(x5, np.mean(b5xlwm,axis=0)-(np.divide(err5m,math.sqrt(10))), np.mean(b5xlwm,axis=0)+(np.divide(err5m,math.sqrt(10))),alpha=0.2, edgecolor='#022dfe', facecolor='#022dfe')

axs[1].set_title("MC: Ind vs Multi")
axs[1].legend()
axs[1].set_xlabel("Neuron (sorted)")


fig.suptitle("2-Task Participation: 2x5")
plt.savefig("Eduardo/LWMC_Participation_2x5.png")
plt.show()