import numpy as np 
import matplotlib.pyplot as plt



plt.figure()
all_f1s = []
all_f2s = []
​
# load all data
for i in range(reps):
    f1 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    f2 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy"))
​
    # cases that improved fitness upon leasioning
    f1[f1<0]=0
    f2[f2<0]=0
​
    plt.plot(f1,f2,'o',color='blue',alpha=0.5)
​
    all_f1s.append(f1)
    all_f2s.append(f2)
​
plt.xlabel("Impact MCLW_LW")
plt.ylabel("Impact MCLW_MC")
​
# 2D hist of the data
all_f1s = np.concatenate(all_f1s)
all_f2s = np.concatenate(all_f2s)
H, _, _ = np.histogram2d(all_f1s, all_f2s, bins=[np.linspace(0,1,0.05)])
​
plt.figure()
plt.imshow(H, aspect="equal")
plt.colorbar()
plt.xlabel("Impact MCLW_LW")
plt.ylabel("Impact MCLW_MC")
​
print("Total number of neurons included in plot: ", np.sum(H))
plt.show()