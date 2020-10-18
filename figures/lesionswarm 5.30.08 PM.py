import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns

dir = str(sys.argv[1])
reps = int(sys.argv[2])
data = []
categories = 8
catmat = np.zeros((27,8))
k=0
for ind in range(reps):
    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    if bf[-1]>0.80:
        ip_lesion = np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
        cp_lesion = np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
        lw_lesion = np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))
        Threshold = 0.99
        cat = np.zeros(categories)
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_lesion,cp_lesion,lw_lesion):
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
                cat[0] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
                cat[1] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
                cat[2] += 1
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
                cat[3] += 1
            if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
                cat[4] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
                cat[5] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neuron
                cat[6] += 1
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                cat[7] += 1
        catmat[k]=cat
        k+=1
        for c in range(categories):
            data.append([ind,c,cat[c]])
print(np.median(catmat,axis=0))
datanumpy = np.array(data)
dataframe = pd.DataFrame({'id': datanumpy[:, 0], 'cat': datanumpy[:, 1], 'num': datanumpy[:, 2]})
g = sns.catplot(x="cat", y="num", hue="id", kind="swarm", data=dataframe, aspect=1.61803398875);
#g._legend.remove()
plt.ylabel("Number of neurons per neural network")
plt.xticks(np.arange(categories),('None','IP','CP','LW','IP+CP','IP+LW','CP+LW','All'))
plt.savefig("NormVar_swarmplot.eps",format='eps')
plt.show()
plt.close()
