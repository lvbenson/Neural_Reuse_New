import numpy as np
import matplotlib.pyplot as plt

####################
# EVOLUTION
####################
reps = 100
gens = 1000
b3 = np.zeros((reps,gens))
b5 = np.zeros((reps,gens))
b10 = np.zeros((reps,gens))
b20 = np.zeros((reps,gens))
for i in range(reps):
    b3[i] = np.load("Data3/best_historyMCLW3_"+str(i)+".npy")
    b5[i] = np.load("Data3/best_historyMCLW5_"+str(i)+".npy")
    b10[i]= np.load("Data3/best_historyMCLW10_"+str(i)+".npy")
    b20[i] = np.load("Data3/best_historyMCLW20_"+str(i)+".npy")

plt.plot(b20.T,label="2x20")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("MC-LW (2x20)")
plt.show()

plt.plot(b10.T,label="2x10")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("MC-LW (2x10)")
plt.show()

plt.plot(b5.T,label="2x5")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("MC-LW (2x5)")
plt.show()

plt.plot(b3.T,label="2x3")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("MC-LW (2x3)")
plt.show()

plt.plot(np.mean(b20,axis=0)**(1/2),label="2x20")
plt.plot(np.mean(b10,axis=0)**(1/2),label="2x10")
plt.plot(np.mean(b5,axis=0)**(1/2),label="2x5")
plt.plot(np.mean(b3,axis=0)**(1/2),label="2x3")
plt.xlabel("Generations")
plt.ylabel("Sqrt Fitness (on avg)")
plt.title("MC (sqrt)")
plt.legend()
plt.show()

plt.hist(b20.T[-1],alpha=0.5,label="2x20")
plt.hist(b10.T[-1],alpha=0.5,label="2x10")
plt.hist(b5.T[-1],alpha=0.5,label="2x5")
plt.hist(b3.T[-1],alpha=0.5,label="2x3")
plt.xlabel("Final fitness")
plt.ylabel("Number of solutions")
plt.title("MC")
plt.legend()
plt.show()
