import numpy as np
import matplotlib.pyplot as plt

reps = 100
nn5 = 2*5

impact5 = np.zeros((reps,nn5))
part5 = np.zeros((reps,nn5))

newimpact = []
#impact histogram

for i in range(reps):
    f1 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    f2 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))

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
    nnlw = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    nnlw = nnlw.T[4:-1]
    nnlw = np.mean(np.abs(np.diff(nnlw)),axis=1)
    nnlw = np.sort(nnlw)[::-1]
    maxlw = np.max(nnlw)
    f1 = nnlw/maxlw

    nnmc = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
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