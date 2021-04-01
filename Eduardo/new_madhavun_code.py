import numpy as np 
import matplotlib.pyplot as plt

plt.figure()
all_f1s = []
all_f2s = []

reps = 100

for i in range(reps):
    f1 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    f2 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy"))

    f1[f1<0]=0
    f2[f2<0]=0

    plt.plot(f1,f2,'o',color='blue',alpha=0.5)

    all_f1s.append(f1)
    all_f2s.append(f2)


plt.xlabel("Impact MCLW_LW")
plt.ylabel("Impact MCLW_MC")

all_f1s = np.concatenate(all_f1s)
all_f2s = np.concatenate(all_f2s)

bins = np.arange(0,1,0.05)
print(bins)
#print(bins)

H, _, _ = np.histogram2d(all_f1s,all_f2s, bins=(bins,bins))

plt.figure()
#plt.xticks(bins)
#plt.yticks(bins)
plt.imshow(H,aspect="equal")
plt.colorbar()
plt.xlabel("Impact MCLW_LW")
plt.ylabel("Impact MCLW_MC")

print("total # in plot:", np.sum(H))
plt.show()