import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reps = 100
nn5 = 2*5
################################
#LW, 2x5
###############################

b5xlw = np.zeros((reps,nn5))

for i in range(reps):
    f = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
    b5xlw[i] = f


v5x = np.zeros((reps,nn5))

for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v5x[i] = nn/max

"""
plt.plot(b5xlw[0],v5x[0],'o-')
plt.show()
"""

coeffs = []

for (x,y) in zip(b5xlw,v5x):

    r = np.corrcoef(x, y)
    coeffs.append(r[0, 1])


plt.hist(coeffs,15,density=False,alpha=0.5)
plt.title("MCLW_LW correlation coeff")
plt.xlabel("Pearson Correlation Coefficients")
plt.ylabel("Number of networks")
plt.show()



"""
for (x,y) in zip(b5xlw,v5x):
    plt.plot(x,y,'o')
plt.xlabel('Impact')
plt.ylabel('Participation')
plt.title('MC-individual')
plt.show()
"""



"""
meanx = []
meany = []
for (x,y) in zip(b5xlw,b5xmc):
    meanx.append(np.mean(x))
    meany.append(np.mean(y))

plt.plot(np.arange(100),meanx)
plt.plot(np.arange(100),meany)

plt.show()
"""