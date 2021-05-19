import numpy as np 
import matplotlib.pyplot as plt

#plt.figure()
all_f1s = []
all_f2s = []

reps = 100

nI = 4
nH = 10

for i in range(reps):
    #f1 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    #f2 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))

    f1 = np.load("Eduardo/Multi_Data/state_MCLW5_LW_"+str(i)+".npy")
    f1 = np.var(f1[:,nI:nI+nH],axis=0)

    f2 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))


    f1[f1<0]=0
    f2[f2<0]=0

    #print(len(f1))
    #print(len(f2))

    """
    for p, (x,y) in enumerate(zip(f1,f2)):
        if x < 0.1 and y < 0.1:
            #print(x)
            #indx = np.where(f1 == x)
            #print(indx)
            f1[p] = 200
            f2[p] = 200
        
        if x < 0.1:
            f1[p] = 200
            f2[p] = 200
        if y < 0.1:
            f1[p] = 200
            f2[p] = 200
        
    """


    #impact5[i] = (f1-f2)
    #imp = impact5[i]
    #f1 = f1[f1 != 200]
    #f2 = f2[f2 != 200]

    #newimpact.append(imp)

    #plt.plot(f1,f2,'o',color='blue',alpha=0.5)
    
    for n in f1:
        if n > 1.0:
            n = 1.0
        if n < 0.0:
            n = 0.0
    for n in f2:
        if n > 1.0:
            n = 1.0
        if n < 0.0:
            n = 0.0
    
    all_f1s.append(f1)
    all_f2s.append(f2)

    inds = np.logical_and(f1>0.0, f2>0.0)
    all_f1s.append(f1[inds])
    all_f2s.append(f2[inds])


#plt.title("All neuron impact")
#plt.xlabel("Impact MCLW_LW")
#plt.ylabel("Impact MCLW_MC")


all_f1s = np.concatenate(all_f1s)
all_f2s = np.concatenate(all_f2s)

bins = np.linspace(0.0,1.2,12)
#print(bins)
#print(bins)

H, _, _ = np.histogram2d(all_f1s,all_f2s, bins=(bins,bins))

#ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ticks = np.arange(0.0,1.2,0.1)

#plt.figure()

#fig, axs = plt.subplots(2, constrained_layout=True)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))


axs[0].set_xticks(ticks)
axs[0].set_yticks(ticks)
axs[0].set_title("Impact vs Participation: LW")
axs[0].imshow(H,origin='lower',aspect="equal",extent=[0.0,1.2,0.0,1.2])
#plt.xticks(bins)
#plt.yticks(bins)
#axs[0].set_colorbar()
axs[0].set_xlabel("Participation")
axs[0].set_ylabel("Impact")


all_f1s = []
all_f2s = []

reps = 100

nI = 4
nH = 10

for i in range(reps):
    #f1 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    #f2 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))

    f1 = np.load("Eduardo/Multi_Data/state_MCLW5_MC_"+str(i)+".npy")
    f1 = np.var(f1[:,nI:nI+nH],axis=0)

    f2 = 1 - (np.load("Eduardo/Multi_Data/lesions_MCLW5_MC_40_"+str(i)+".npy"))


    f1[f1<0]=0
    f2[f2<0]=0

    #print(len(f1))
    #print(len(f2))

    """
    for p, (x,y) in enumerate(zip(f1,f2)):
        if x < 0.1 and y < 0.1:
            #print(x)
            #indx = np.where(f1 == x)
            #print(indx)
            f1[p] = 200
            f2[p] = 200
        
        if x < 0.1:
            f1[p] = 200
            f2[p] = 200
        if y < 0.1:
            f1[p] = 200
            f2[p] = 200
        
    """


    #impact5[i] = (f1-f2)
    #imp = impact5[i]
    #f1 = f1[f1 != 200]
    #f2 = f2[f2 != 200]

    #newimpact.append(imp)

    #plt.plot(f1,f2,'o',color='blue',alpha=0.5)
    
    for n in f1:
        if n > 1.0:
            n = 1.0
        if n < 0.0:
            n = 0.0
    for n in f2:
        if n > 1.0:
            n = 1.0
        if n < 0.0:
            n = 0.0
    
    all_f1s.append(f1)
    all_f2s.append(f2)

    inds = np.logical_and(f1>0.0, f2>0.0)
    all_f1s.append(f1[inds])
    all_f2s.append(f2[inds])


#plt.title("All neuron impact")
#plt.xlabel("Impact MCLW_LW")
#plt.ylabel("Impact MCLW_MC")


all_f1s = np.concatenate(all_f1s)
all_f2s = np.concatenate(all_f2s)

bins = np.linspace(0.0,1.2,12)
#print(bins)
#print(bins)

H, _, _ = np.histogram2d(all_f1s,all_f2s, bins=(bins,bins))

#ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ticks = np.arange(0.0,1.2,0.1)


axs[1].set_xticks(ticks)
axs[1].set_yticks(ticks)
axs[1].set_title("Impact vs Participation: MC")
im = axs[1].imshow(H,origin='lower',aspect="equal",extent=[0.0,1.2,0.0,1.2])
#plt.xticks(bins)
#plt.yticks(bins)
#axs[1].colorbar()
axs[1].set_xlabel("Participation")
axs[1].set_ylabel("Impact")

fig.suptitle("Multiple Tasks")
fig.colorbar(im,ax=axs)
plt.savefig("Eduardo/MULTI_HIST_ALL.png")
plt.show()

