import numpy as np
import matplotlib.pyplot as plt

reps = 100
p20 = np.zeros((reps,2))
p10 = np.zeros((reps,2))
p5 = np.zeros((reps,2))
p3 = np.zeros((reps,2))
for i in range(reps):
    p20[i][0] = np.load("Data3/perf_MCLW20_LW_"+str(i)+".npy")
    p20[i][1] = np.load("Data3/perf_MCLW20_MC_"+str(i)+".npy")
    p10[i][0] = np.load("Data3/perf_MCLW10_LW_"+str(i)+".npy")
    p10[i][1] = np.load("Data3/perf_MCLW10_MC_"+str(i)+".npy")
    p5[i][0] = np.load("Data3/perf_MCLW5_LW_"+str(i)+".npy")
    p5[i][1] = np.load("Data3/perf_MCLW5_MC_"+str(i)+".npy")
    p3[i][0] = np.load("Data3/perf_MCLW3_LW_"+str(i)+".npy")
    p3[i][1] = np.load("Data3/perf_MCLW3_MC_"+str(i)+".npy")
# plt.plot(p20.T[0],p20.T[1],'o',alpha=0.5,label="2x20")
# plt.plot(p10.T[0],p10.T[1],'o',alpha=0.5,label="2x10")
# plt.plot(p5.T[0],p5.T[1],'o',alpha=0.5,label="2x5")
# plt.plot(p3.T[0],p3.T[1],'o',alpha=0.5,label="2x3")
# plt.xlabel("LW")
# plt.xlabel("MC")
# plt.title("Final Performance (MCLW)")
# plt.legend()
# plt.show()

m20=p20.T[0]*p20.T[1]
m10=p10.T[0]*p10.T[1]
m5=p5.T[0]*p5.T[1]
m3=p3.T[0]*p3.T[1]
# plt.hist(m20,alpha=0.5,label="2x20")
# plt.hist(m10,alpha=0.5,label="2x10")
# plt.hist(m5,alpha=0.5,label="2x5")
# plt.hist(m3,alpha=0.5,label="2x3")
# plt.legend()
# plt.show()
print(np.argmax(m20))
print(np.argmax(m10))
print(np.argmax(m5))
print(np.argmax(m3))
