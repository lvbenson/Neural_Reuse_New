import numpy as np
import matplotlib.pyplot as plt


# #####################
# # LESIONS
# #####################
reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20

b3x = np.zeros((reps,nn3))
b3xerrs = np.zeros((reps,nn3))
b5x = np.zeros((reps,nn5))
b5xerrs = np.zeros((reps,nn5))
b10x = np.zeros((reps,nn10))
b10xerrs = np.zeros((reps,nn3))
b20x = np.zeros((reps,nn20))
b20xerrs = np.zeros((reps,nn3))
for i in range(reps):
    f = np.load("Eduardo/Data3/lesions_MCLW3_LW_40_"+str(i)+".npy")
    b3x[i] = np.sort(f)
    b3xerrs[i] = np.std(f)
    f = np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy")
    b5x[i] = np.sort(f)
    b5xerrs[i] = np.std(f)
    f = np.load("Eduardo/Data3/lesions_MCLW10_LW_40_"+str(i)+".npy")
    b10x[i] = np.sort(f)
    b10xerrs[i] = np.std(f)
    f = np.load("Eduardo/Data3/lesions_MCLW20_LW_40_"+str(i)+".npy")
    b20x[i] = np.sort(f)
    b20xerrs[i] = np.std(f)

print(np.mean(b20x,axis=0).shape)
x20 = list(range(0,40))
x10 = list(range(0,20))
x5 = list(range(0,10))
x3 = list(range(0,6))

err20 = []
err10 = []
err5 = []
err3 = []

for i in b20x.T:
    err20.append(np.std(i))
for i in b10x.T:
    err10.append(np.std(i))
for i in b5x.T:
    err5.append(np.std(i))
for i in b3x.T:
    err3.append(np.std(i))


import math
#############################
fig, axs = plt.subplots(2,sharex=True, constrained_layout=True)
#plt.fill_between(x, y-yerr, y+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
axs[0].plot(np.mean(b20x,axis=0),'o-',markersize=2,label="2x20",color='#c88700')
axs[0].fill_between(x20, np.mean(b20x,axis=0)-(np.divide(err20,math.sqrt(40))), np.mean(b20x,axis=0)+(np.divide(err20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0].plot(np.mean(b10x,axis=0),'o-',markersize=2,label="2x10",color='#1B2ACC')
axs[0].fill_between(x10, np.mean(b10x,axis=0)-(np.divide(err10,math.sqrt(20))), np.mean(b10x,axis=0)+(np.divide(err10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

axs[0].plot(np.mean(b5x,axis=0),'o-',markersize=2,label="2x5",color='#3F7F4C')
axs[0].fill_between(x5, np.mean(b5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0].plot(np.mean(b3x,axis=0),'o-',markersize=2,label="2x3",color='#ff0000')
axs[0].fill_between(x3, np.mean(b3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(b3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')
#ff0000
#plt.fill_between(x10, np.mean(b10x,axis=0)-np.std(b10x,axis=0), np.mean(b10x,axis=0)+np.std(b10x,axis=0),facecolor='r',alpha=0.2)

#gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1, random_start=100)

#plt.plot(np.mean(b20x,axis=0),label="2x20",color='b')
#plt.errorbar(x20,np.mean(b20x,axis=0),yerr=np.std(b20x,axis=0), fmt='o',color='blue',ecolor='lightgray',elinewidth=3,capsize=0,label="2x20")


#plt.errorbar(x10,np.mean(b10x,axis=0),yerr=np.std(b10x,axis=0),label="2x10")
#plt.errorbar(x5,np.mean(b5x,axis=0),yerr=np.std(b5x,axis=0),label="2x5")
#plt.errorbar(x3,np.mean(b3x,axis=0),yerr=np.std(b3x,axis=0),label="2x3")

#ax0.errorbar(x, y, yerr=error, fmt='-o')
#plt.plot(np.mean(b10x,axis=0),label="2x10")
#plt.plot(np.mean(b5x,axis=0),'o-', label="2x5")
#plt.plot(np.mean(b3x,axis=0),'o-', label="2x3")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Impact")
axs[0].set_title("Average over Ensemble (MCLW_LW)")
axs[0].legend()
#plt.show()

b3x = np.zeros((reps,nn3))
b5x = np.zeros((reps,nn5))
b10x = np.zeros((reps,nn10))
b20x = np.zeros((reps,nn20))
for i in range(reps):
    f = np.load("Eduardo/Data3/lesions_MCLW3_MC_40_"+str(i)+".npy")
    b3x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy")
    b5x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW10_MC_40_"+str(i)+".npy")
    b10x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW20_MC_40_"+str(i)+".npy")
    b20x[i] = np.sort(f)

err20 = []
err10 = []
err5 = []
err3 = []

for i in b20x.T:
    err20.append(np.std(i))
for i in b10x.T:
    err10.append(np.std(i))
for i in b5x.T:
    err5.append(np.std(i))
for i in b3x.T:
    err3.append(np.std(i))

axs[1].plot(np.mean(b20x,axis=0),'o-',markersize=2,label="2x20",color='#c88700')
axs[1].fill_between(x20, np.mean(b20x,axis=0)-(np.divide(err20,math.sqrt(40))), np.mean(b20x,axis=0)+(np.divide(err20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1].plot(np.mean(b10x,axis=0),'o-',markersize=2,label="2x10",color='#1B2ACC')
axs[1].fill_between(x10, np.mean(b10x,axis=0)-(np.divide(err10,math.sqrt(20))), np.mean(b10x,axis=0)+(np.divide(err10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

axs[1].plot(np.mean(b5x,axis=0),'o-',markersize=2,label="2x5",color='#3F7F4C')
axs[1].fill_between(x5, np.mean(b5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1].plot(np.mean(b3x,axis=0),'o-',markersize=2,label="2x3",color='#ff0000')
axs[1].fill_between(x3, np.mean(b3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(b3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')


"""
axs[1].plot(np.mean(b20x,axis=0),'o-',markersize=2,label="2x20",color='#c88700')
axs[1].fill_between(x20, np.mean(b20x,axis=0)-np.std(b20x,axis=0), np.mean(b20x,axis=0)+np.std(b20x,axis=0),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1].plot(np.mean(b10x,axis=0),'o-',markersize=2,label="2x10",color='#1B2ACC')
axs[1].fill_between(x10, np.mean(b10x,axis=0)-np.std(b10x,axis=0), np.mean(b10x,axis=0)+np.std(b10x,axis=0),alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

axs[1].plot(np.mean(b5x,axis=0),'o-',markersize=2,label="2x5",color='#3F7F4C')
axs[1].fill_between(x5, np.mean(b5x,axis=0)-np.std(b5x,axis=0), np.mean(b5x,axis=0)+np.std(b5x,axis=0),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1].plot(np.mean(b3x,axis=0),'o-',markersize=2,label="2x3",color='#ff0000')
axs[1].fill_between(x3, np.mean(b3x,axis=0)-np.std(b3x,axis=0), np.mean(b3x,axis=0)+np.std(b3x,axis=0),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')
"""
#plt.plot(np.mean(b20x,axis=0),'o-',label="2x20")
#plt.plot(np.mean(b10x,axis=0),'o-',label="2x10")
#plt.plot(np.mean(b5x,axis=0),'o-',label="2x5")
#plt.plot(np.mean(b3x,axis=0),'o-',label="2x3")
#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
fig.tight_layout()
plt.title("Average over Ensemble (MCLW_MC)")
plt.legend()
plt.savefig("Eduardo/pairwise_avg_ensemble")
plt.show()

"""
######################################################################################################
######################################################################################################
#LAURENS EDITS
######################################################################################################
######################################################################################################


####################
#LESIONS
####################
reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20

b3x = np.zeros((reps,nn3))
#b3y = np.zeros((reps,nn3))
b5x = np.zeros((reps,nn5))
b5y = np.zeros((reps,nn5))
b10x = np.zeros((reps,nn10))
b10y = np.zeros((reps,nn10))
b20x = np.zeros((reps,nn20))
b20y = np.zeros((reps,nn20))

#Eduardo\Data3
b3y = []
b5y = []
b10y = []
b20y = []

for i in range(reps):
    f1 = np.load("Eduardo/Data3/lesions_MCLW3_LW_40_"+str(i)+".npy")
    f2 = np.load("Eduardo/Data3/lesions_MCLW3_MC_40_"+str(i)+".npy")
    b3x[i] = (f1-f2) #finds difference between task lesions
    #b3y[i] = (f1*f2) #multiplies lesions. If 1 - value is high, 
    #heavily involved in both tasks
    b3y.append([f1,f2])
    f1 = np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy")
    f2 = np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy")
    b5x[i] = (f1-f2)
    #b5y[i] = (f1*f2)
    b5y.append([f1,f2])
    f1 = np.load("Eduardo/Data3/lesions_MCLW10_LW_40_"+str(i)+".npy")
    f2 = np.load("Eduardo/Data3/lesions_MCLW10_MC_40_"+str(i)+".npy")
    b10x[i] = (f1-f2)
    #b10y[i] = (f1*f2)
    b10y.append([f1,f2])
    f1 = np.load("Eduardo/Data3/lesions_MCLW20_LW_40_"+str(i)+".npy")
    f2 = np.load("Eduardo/Data3/lesions_MCLW20_MC_40_"+str(i)+".npy")
    b20x[i] = (f1-f2)
    #b20y[i] = (f1*f2)
    b20y.append([f1,f2])


x3c = 0
x13c = 0
x213c = 0
x3213c = 0
x5c = 0
x15c = 0
x215c = 0
x3215c = 0
x10c = 0
x110c = 0
x2110c = 0
x32110c = 0
x20c = 0
x120c = 0
x2120c = 0
x32120c = 0
none_dual3 = 0
none_dual5 = 0
none_dual10 = 0
none_dual20 = 0

for i in b3y:
    
    for l,m in zip(i[0],i[1]):
        if l and m < 0.1:
            x3c += 1
        elif (0.1 < l < 0.2) and (0.1 < m < 0.2):
            x13c += 1
        elif (0.2 < l < 0.5) and (0.2 < m < 0.5):
            x213c += 1
        elif (0.5 < l < 0.9) and (0.5 < m < 0.9):
            x3213c += 1
        elif (l > 0.99) and (m > 0.99):
            none_dual3 += 1

for i in b5y:
    
    for l,m in zip(i[0],i[1]):
        if l and m < 0.1:
            x5c += 1
        elif (0.1 < l < 0.2) and (0.1 < m < 0.2):
            x15c += 1
        elif (0.2 < l < 0.5) and (0.2 < m < 0.5):
            x215c += 1
        elif (0.5 < l < 0.9) and (0.5 < m < 0.9):
            x3215c += 1
        elif (l > 0.99) and (m > 0.99):
            none_dual5 += 1

for i in b10y:
    
    for l,m in zip(i[0],i[1]):
        if l and m < 0.1:
            x10c += 1
        elif (0.1 < l < 0.2) and (0.1 < m < 0.2):
            x110c += 1
        elif (0.2 < l < 0.5) and (0.2 < m < 0.5):
            x2110c += 1
        elif (0.5 < l < 0.9) and (0.5 < m < 0.9):
            x32110c += 1
        elif (l > 0.99) and (m > 0.99):
            none_dual10 += 1

for i in b20y:
    
    for l,m in zip(i[0],i[1]):
        if l and m < 0.1:
            x20c += 1
        elif (0.1 < l < 0.2) and (0.1 < m < 0.2):
            x120c += 1
        elif (0.2 < l < 0.5) and (0.2 < m < 0.5):
            x2120c += 1
        elif (0.5 < l < 0.9) and (0.5 < m < 0.9):
            x32120c += 1
        elif (l > 0.99) and (m > 0.99):
            none_dual20 += 1


T = 0.90
T2 = 0.1

T_1 = 0.8
T_12 = 0.2

T_2 = 0.5
T_22 = 0.5

T_3 = 0.1
T_33 = 0.9

x3a = np.count_nonzero(b3x < -T)
x3b = np.count_nonzero(b3x > T)
#x3c = np.count_nonzero(b3y < T2)
x5a = np.count_nonzero(b5x < -T)
x5b = np.count_nonzero(b5x > T)
#x5c = np.count_nonzero(b5y < T2)
x10a = np.count_nonzero(b10x < -T)
x10b = np.count_nonzero(b10x > T)
#x10c = np.count_nonzero(b10y < T2)
x20a = np.count_nonzero(b20x < -T)
x20b = np.count_nonzero(b20x > T)
#x20c = np.count_nonzero(b20y < T2)

x13a = (np.count_nonzero(b3x < -T_1) - x3a)
#print(x3a)
#print(np.count_nonzero(b3x < -T_1) - x3a)
x13b = (np.count_nonzero(b3x > T_1) - x3b)
#x13c = (np.count_nonzero(b3y < T_12) - x3c)
x15a = (np.count_nonzero(b5x < -T_1) - x5a)
x15b = (np.count_nonzero(b5x > T_1) - x5b)
#x15c = (np.count_nonzero(b5y < T_12) - x5c)
x110a = (np.count_nonzero(b10x < -T_1) - x10a)
x110b = (np.count_nonzero(b10x > T_1) - x10b)
#x110c = (np.count_nonzero(b10y < T_12) - x10c)
x120a = (np.count_nonzero(b20x < -T_1) - x20a)
x120b = (np.count_nonzero(b20x > T_1) - x20b)
#x120c = (np.count_nonzero(b20y < T_12) - x20c)

x213a = (np.count_nonzero(b3x < -T_2) - x13a)
#print(x213a)
x213b = (np.count_nonzero(b3x > T_2) - x13b)
#x213c = (np.count_nonzero(b3y < T_22) - x13c)
x215a = (np.count_nonzero(b5x < -T_2) - x15a)
x215b = (np.count_nonzero(b5x > T_2) - x15b)
#x215c = (np.count_nonzero(b5y < T_22) - x15c)
x2110a = (np.count_nonzero(b10x < -T_2) - x110a)
x2110b = (np.count_nonzero(b10x > T_2) - x110b)
#x2110c = (np.count_nonzero(b10y < T_22) - x110c)
x2120a = (np.count_nonzero(b20x < -T_2) - x120a)
x2120b = (np.count_nonzero(b20x > T_2) - x120b)
#x2120c = (np.count_nonzero(b20y < T_22) - x120c)

x3213a = (np.count_nonzero(b3x < -T_3) - x213a)
#print(x3213a)
x3213b = (np.count_nonzero(b3x > T_3) - x213b)
#x3213c = (np.count_nonzero(b3y < T_33) - x213c)
x3215a = (np.count_nonzero(b5x < -T_3) - x215a)
x3215b = (np.count_nonzero(b5x > T_3) - x215b)
#x3215c = (np.count_nonzero(b5y < T_33) - x215c)
x32110a = (np.count_nonzero(b10x < -T_3) - x2110a)
x32110b = (np.count_nonzero(b10x > T_3) - x2110b)
#x32110c = (np.count_nonzero(b10y < T_33) - x2110c)
x32120a = (np.count_nonzero(b20x < -T_3) - x2120a)
x32120b = (np.count_nonzero(b20x > T_3) - x2120b)
#x32120c = (np.count_nonzero(b20y < T_33) - x2120c)

fig, axs = plt.subplots(2,2)
#custom_ylim = (0,525)

# Setting the values for all axes.
#plt.setp(axs, ylim=custom_ylim)

axs[0,0].set_title("Threshold: 0.9")
axs[0,0].plot([-1,0,1],[x20a,x20c,x20b],'o',color='blue',label="2x20")
axs[0,0].plot([-1,0,1],[x10a,x10c,x10b],'o',color='orange',label="2x10")
axs[0,0].plot([-1,0,1],[x5a,x5c,x5b],'o',color='green',label="2x5")
axs[0,0].plot([-1,0,1],[x3a,x3c,x3b],'o',color='red',label="2x3")


axs[0,1].set_title("Threshold: 0.8<t<0.9")
axs[0,1].plot([-1,0,1],[x120a,x120c,x120b],'*',color='blue',label="2x20")
axs[0,1].plot([-1,0,1],[x110a,x110c,x110b],'*',color='orange',label="2x10")
axs[0,1].plot([-1,0,1],[x15a,x15c,x15b],'*',color='green',label="2x5")
axs[0,1].plot([-1,0,1],[x13a,x13c,x13b],'*',color='red',label="2x3")

axs[1,0].set_title("Threshold: 0.5<t<0.8")
axs[1,0].plot([-1,0,1],[x2120a,x2120c,x2120b],'s',color='blue',label="2x20")
axs[1,0].plot([-1,0,1],[x2110a,x2110c,x2110b],'s',color='orange',label="2x10")
axs[1,0].plot([-1,0,1],[x215a,x215c,x215b],'s',color='green',label="2x5")
axs[1,0].plot([-1,0,1],[x213a,x213c,x213b],'s',color='red',label="2x3")

axs[1,1].set_title("Threshold: 0.1<t<0.5")
axs[1,1].plot([-1,0,1],[x32120a,x32120c,x32120b],'v',color='blue',label="2x20")
axs[1,1].plot([-1,0,1],[x32110a,x32110c,x32110b],'v',color='orange',label="2x10")
axs[1,1].plot([-1,0,1],[x3215a,x3215c,x3215b],'v',color='green',label="2x5")
axs[1,1].plot([-1,0,1],[x3213a,x3213c,x3213b],'v',color='red',label="2x3")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle('Pairwise involvement')
#plt.xlabel("Involvement (-1:LW,0:Both,1:MC) ")
plt.tight_layout()
plt.show()

######################################################################
#Neural involvement over multiple experiments
######################################################################
reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20

#MC - only agents

mc3 = np.zeros((reps,nn3))
mc5 = np.zeros((reps,nn5))
mc10 = np.zeros((reps,nn10))
mc20 = np.zeros((reps,nn20))
for i in range(reps):
    f = np.load("Eduardo/Data/lesions_MC3_40_"+str(i)+".npy")
    mc3[i] = f
    f1 = np.load("Eduardo/Data/lesions_MC5_40_"+str(i)+".npy")
    mc5[i] = f1
    f2 = np.load("Eduardo/Data/lesions_MC10_40_"+str(i)+".npy")
    mc10[i] = f2
    f3 = np.load("Eduardo/Data/lesions_MC20_40_"+str(i)+".npy")
    mc20[i] = f3

T = 0.1

#MC individual involvement

lmc3 = np.count_nonzero(mc3 < T)
lmc5 = np.count_nonzero(mc5 < T)
lmc10 = np.count_nonzero(mc10 < T)
lmc20 = np.count_nonzero(mc20 < T)

###LW - only agents

lw3 = np.zeros((reps,nn3))
lw5 = np.zeros((reps,nn5))
lw10 = np.zeros((reps,nn10))
lw20 = np.zeros((reps,nn20))
for i in range(reps):
    f = np.load("Eduardo/Data/lesions_LW3_40_"+str(i)+".npy")
    lw3[i] = f
    f1 = np.load("Eduardo/Data/lesions_LW5_40_"+str(i)+".npy")
    lw5[i] = f1
    f2 = np.load("Eduardo/Data/lesions_LW10_40_"+str(i)+".npy")
    lw10[i] = f2
    f3 = np.load("Eduardo/Data/lesions_LW20_40_"+str(i)+".npy")
    lw20[i] = f3

T = 0.1

#LW individual involvement

llw3 = np.count_nonzero(lw3 < T)
llw5 = np.count_nonzero(lw5 < T)
llw10 = np.count_nonzero(lw10 < T)
llw20 = np.count_nonzero(lw20 < T)


#NO task involvement from LWMC
NT = 0.01
NT2 = 0.99

total3 = reps*6
total5 = reps*10
total10 = reps*20
total20 = reps*40

none3 = (np.count_nonzero(b3x < -NT) + np.count_nonzero(b3x > NT) + np.sum(none_dual3))
none5 = (np.count_nonzero(b5x < -NT) + np.count_nonzero(b5x > NT) + np.sum(none_dual5))
none10 = (np.count_nonzero(b10x < -NT) + np.count_nonzero(b10x > NT) + np.sum(none_dual10))
none20 = (np.count_nonzero(b20x < -NT) + np.count_nonzero(b20x > NT) + np.sum(none_dual20))



#####PLOTTING

size = ["2x3", "2x5", "2x10", "2x20"]
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,5))

ax[0,0].plot(size[0],lmc3, 'v',label='MC-ind',color='blue')
ax[0,0].plot(size[1],lmc5, 'v',color='blue')
ax[0,0].plot(size[2],lmc10, 'v',color='blue')
ax[0,0].plot(size[3],lmc20, 'v',color='blue')

#MC involvement from pairwise
ax[0,0].plot(size[0],x3a, '*',label='MC-MCLW',color='green')
ax[0,0].plot(size[1],x5a, '*',color='green')
ax[0,0].plot(size[2],x10a, '*',color='green')
ax[0,0].plot(size[3],x20a, '*',color='green')

#both involvement from pairwise
ax[0,0].plot(size[0],x3c, 'o',label='MC&LW-MCLW',color='orange')
ax[0,0].plot(size[1],x5c, 'o',color='orange')
ax[0,0].plot(size[2],x10c, 'o',color='orange')
ax[0,0].plot(size[3],x20c, 'o',color='orange')


#orange+green vs blue (MC)
ax[0,1].plot(size[0],(x3c+x3a), 'o',label='MC&LW(pairwise) + MC(pairwise)', color='brown')
ax[0,1].plot(size[1],(x5c+x5a), 'o',color='brown')
ax[0,1].plot(size[2],(x10c+x10a), 'o',color='brown')
ax[0,1].plot(size[3],(x20c+x20a), 'o',color='brown')

ax[0,1].plot(size[0],lmc3, 'v',label='MC-ind',color='blue')
ax[0,1].plot(size[1],lmc5, 'v',color='blue')
ax[0,1].plot(size[2],lmc10, 'v',color='blue')
ax[0,1].plot(size[3],lmc20, 'v',color='blue')

###ORANGE+GREEN VS BLUE (LW)

ax[1,1].plot(size[0],(x3c+x3b), 'o',label='MC&LW(pairwise) + LW(pairwise)', color='purple')
ax[1,1].plot(size[1],(x5c+x5b), 'o',color='purple')
ax[1,1].plot(size[2],(x10c+x10b), 'o',color='purple')
ax[1,1].plot(size[3],(x20c+x20b), 'o',color='purple')

ax[1,1].plot(size[0],llw3, 'v',label='LW-ind',color='teal')
ax[1,1].plot(size[1],llw5, 'v',color='teal')
ax[1,1].plot(size[2],llw10, 'v',color='teal')
ax[1,1].plot(size[3],llw20, 'v',color='teal')

#first plot but with NO involvement also

#none involvement from pairwise
ax[1,0].plot(size[0],lmc3, 'v',label='MC-ind',color='blue')
ax[1,0].plot(size[1],lmc5, 'v',color='blue')
ax[1,0].plot(size[2],lmc10, 'v',color='blue')
ax[1,0].plot(size[3],lmc20, 'v',color='blue')

#MC involvement from pairwise
ax[1,0].plot(size[0],x3a, '*',label='MC-MCLW',color='green')
ax[1,0].plot(size[1],x5a, '*',color='green')
ax[1,0].plot(size[2],x10a, '*',color='green')
ax[1,0].plot(size[3],x20a, '*',color='green')

#both involvement from pairwise
ax[1,0].plot(size[0],x3c, 'o',label='MC&LW-MCLW',color='orange')
ax[1,0].plot(size[1],x5c, 'o',color='orange')
ax[1,0].plot(size[2],x10c, 'o',color='orange')
ax[1,0].plot(size[3],x20c, 'o',color='orange')

#NO INVOLVEMENT FROM PAIRWISE

ax[1,0].plot(size[0],none3, 'd',label='NONE-MCLW',color='black')
ax[1,0].plot(size[1],none5, 'd',color='black')
ax[1,0].plot(size[2],none10, 'd',color='black')
ax[1,0].plot(size[3],none20, 'd',color='black')

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')

plt.suptitle("Heavily involved neurons")
ax[0,0].legend(bbox_to_anchor=(1.05, 1), loc='lower center', prop=fontP)
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc='lower center', prop=fontP)
ax[1,0].legend(bbox_to_anchor=(1.05, 1), loc='lower center', prop=fontP)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc='lower center', prop=fontP)

plt.tight_layout()
plt.show()

"""

######################################################################################################
######################################################################################################
#END OF LAUREN'S EDITS
######################################################################################################
######################################################################################################



#
# plt.hist(b3y.flatten(),10,density=False,alpha=0.5,label="2x3")
# plt.xlabel("Difference in Impact between the two tasks (LW-MC)")
# plt.ylabel("Number of neurons")
# plt.legend()
# plt.show()
#
# plt.hist(b5y.flatten(),10,density=False,alpha=0.5,label="2x5")
# plt.xlabel("Difference in Impact between the two tasks (LW-MC)")
# plt.ylabel("Number of neurons")
# plt.legend()
# plt.show()
#
# plt.hist(b10y.flatten(),10,density=False,alpha=0.5,label="2x10")
# plt.xlabel("Difference in Impact between the two tasks (LW-MC)")
# plt.ylabel("Number of neurons")
# plt.legend()
# plt.show()
#
# plt.hist(b20y.flatten(),10,density=False,alpha=0.5,label="2x20")
# plt.xlabel("Difference in Impact between the two tasks (LW-MC)")
# plt.ylabel("Number of neurons")
# plt.legend()
# plt.show()

#####################
# LESIONS
#####################
reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20

b3x = np.zeros((reps,nn3))
b5x = np.zeros((reps,nn5))
b10x = np.zeros((reps,nn10))
b20x = np.zeros((reps,nn20))
for i in range(reps):
    f = np.load("Eduardo/Data3/lesions_MCLW3_LW_40_"+str(i)+".npy")
    b3x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy")
    b5x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW10_LW_40_"+str(i)+".npy")
    b10x[i] = np.sort(f)
    f = np.load("Eduardo/Data3/lesions_MCLW20_LW_40_"+str(i)+".npy")
    b20x[i] = np.sort(f)

b3 = np.zeros((reps,nn3))
b5 = np.zeros((reps,nn5))
b10 = np.zeros((reps,nn10))
b20 = np.zeros((reps,nn20))
for i in range(reps):
    f = np.load("Eduardo/Data/lesions_LW3_40_"+str(i)+".npy")
    b3[i] = np.sort(f)
    f = np.load("Eduardo/Data/lesions_LW5_40_"+str(i)+".npy")
    b5[i] = np.sort(f)
    f = np.load("Eduardo/Data/lesions_LW10_40_"+str(i)+".npy")
    b10[i] = np.sort(f)
    f = np.load("Eduardo/Data/lesions_LW20_40_"+str(i)+".npy")
    b20[i] = np.sort(f)

err20 = []
err10 = []
err5 = []
err3 = []

for i in b20x.T:
    err20.append(np.std(i))
for i in b10x.T:
    err10.append(np.std(i))
for i in b5x.T:
    err5.append(np.std(i))
for i in b3x.T:
    err3.append(np.std(i))

nerr20 = []
nerr10 = []
nerr5 = []
nerr3 = []

for i in b20.T:
    nerr20.append(np.std(i))
for i in b10.T:
    nerr10.append(np.std(i))
for i in b5.T:
    nerr5.append(np.std(i))
for i in b3.T:
    nerr3.append(np.std(i))


####################################################################################################
#PLOTTING IMPACT FOR SINGLE VS DUAL
###################################################################################################3
import math
fig, axs = plt.subplots(2,2)
#print(len(b20x))
#print(np.mean(b20x,axis=0)-np.std(b20x,axis=0)/10)

axs[0,0].plot(np.mean(b20x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[0,0].fill_between(x20, np.mean(b20x,axis=0)-(np.divide(err20,math.sqrt(40))), np.mean(b20x,axis=0)+(np.divide(err20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0,0].plot(np.mean(b20,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[0,0].fill_between(x20, np.mean(b20,axis=0)-(np.divide(nerr20,math.sqrt(40))), np.mean(b20,axis=0)+(np.divide(nerr20,math.sqrt(40))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

axs[0,0].set_xlabel("Neuron (sorted)")
axs[0,0].set_ylabel("Impact")
axs[0,0].set_title("Avg (MCLW_LW 2x20)")
axs[0,0].legend()
#plt.show()

axs[0,1].plot(np.mean(b10x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[0,1].fill_between(x10, np.mean(b10x,axis=0)-(np.divide(err10,math.sqrt(20))), np.mean(b10x,axis=0)+(np.divide(err10,math.sqrt(20))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0,1].plot(np.mean(b10,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[0,1].fill_between(x10, np.mean(b10,axis=0)-(np.divide(nerr10,math.sqrt(20))), np.mean(b10,axis=0)+(np.divide(nerr10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[0,1].set_title("Avg (MCLW_LW 2x10)")
axs[0,1].legend()
#.show()

axs[1,0].plot(np.mean(b5x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[1,0].fill_between(x5, np.mean(b5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(b5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1,0].plot(np.mean(b5,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[1,0].fill_between(x5, np.mean(b5,axis=0)-(np.divide(nerr5,math.sqrt(10))), np.mean(b5,axis=0)+(np.divide(nerr5,math.sqrt(10))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[1,0].set_title("Avg (MCLW_LW 2x5)")
axs[1,0].legend()
#plt.show()

axs[1,1].plot(np.mean(b3x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[1,1].fill_between(x3, np.mean(b3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(b3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1,1].plot(np.mean(b3,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[1,1].fill_between(x3, np.mean(b3,axis=0)-(np.divide(nerr3,math.sqrt(6))), np.mean(b3,axis=0)+(np.divide(nerr3,math.sqrt(6))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[1,1].set_title("Avg (MCLW_LW 2x3)")
axs[1,1].legend()
plt.tight_layout()
plt.show()

# plt.boxplot([b3.T[0],b3.T[1],b3.T[2],b3.T[3]])
# plt.plot(np.arange(1,5),np.mean(b3,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x3)")
# plt.show()
#
# plt.boxplot([b5.T[0],b5.T[1],b5.T[2],b5.T[3]])
# plt.plot(np.arange(1,5),np.mean(b5,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x5)")
# plt.show()
#
# plt.boxplot([b10.T[0],b10.T[1],b10.T[2],b10.T[3]])
# plt.plot(np.arange(1,5),np.mean(b10,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x10)")
# plt.show()
#
# plt.boxplot([b20.T[0],b20.T[1],b20.T[2],b20.T[3]])
# plt.plot(np.arange(1,5),np.mean(b20,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x20)")
# plt.show()
#
#####################
# VARIABILITY AND RATE OF CHANGE
#####################

reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20


#MCLW_LW
v3x = np.zeros((reps,nn3))
v5x = np.zeros((reps,nn5))
v10x = np.zeros((reps,nn10))
v20x = np.zeros((reps,nn20))
for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW3_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v3x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v5x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW10_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v10x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW20_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v20x[i] = nn/max

#####LWMC-MC
mcv3x = np.zeros((reps,nn3))
mcv5x = np.zeros((reps,nn5))
mcv10x = np.zeros((reps,nn10))
mcv20x = np.zeros((reps,nn20))
for i in range(reps):
    nn = np.load("Eduardo/Data3/state_MCLW3_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv3x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW5_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv5x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW10_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv10x[i] = nn/max
    nn = np.load("Eduardo/Data3/state_MCLW20_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    mcv20x[i] = nn/max


x20 = list(range(0,40))
x10 = list(range(0,20))
x5 = list(range(0,10))
x3 = list(range(0,6))

err20 = []
err10 = []
err5 = []
err3 = []

for i in v20x.T:
    err20.append(np.std(i))
for i in v10x.T:
    err10.append(np.std(i))
for i in v5x.T:
    err5.append(np.std(i))
for i in v3x.T:
    err3.append(np.std(i))

mcerr20 = []
mcerr10 = []
mcerr5 = []
mcerr3 = []

for i in mcv20x.T:
    mcerr20.append(np.std(i))
for i in mcv10x.T:
    mcerr10.append(np.std(i))
for i in mcv5x.T:
    mcerr5.append(np.std(i))
for i in mcv3x.T:
    mcerr3.append(np.std(i))

import math
#############################
fig, axs = plt.subplots(2,sharex=True, constrained_layout=True)
axs[0].plot(np.mean(v20x,axis=0),'o-',markersize=2,label="2x20",color='#c88700')
axs[0].fill_between(x20, np.mean(v20x,axis=0)-(np.divide(err20,math.sqrt(40))), np.mean(v20x,axis=0)+(np.divide(err20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0].plot(np.mean(v10x,axis=0),'o-',markersize=2,label="2x10",color='#1B2ACC')
axs[0].fill_between(x10, np.mean(v10x,axis=0)-(np.divide(err10,math.sqrt(20))), np.mean(v10x,axis=0)+(np.divide(err10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

axs[0].plot(np.mean(v5x,axis=0),'o-',markersize=2,label="2x5",color='#3F7F4C')
axs[0].fill_between(x5, np.mean(v5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(v5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[0].plot(np.mean(v3x,axis=0),'o-',markersize=2,label="2x3",color='#ff0000')
axs[0].fill_between(x3, np.mean(v3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(v3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')

#plot difference for mclw-lw and mclw-mc 

#plt.plot(np.mean(v20,axis=0),'o-',label="2x20")
#plt.plot(np.mean(v10,axis=0),'o-',label="2x10")
#plt.plot(np.mean(v5,axis=0),'o-',label="2x5")
#plt.plot(np.mean(v3,axis=0),'o-',label="2x3")
axs[0].set_xlabel("Neuron (sorted)")
axs[0].set_ylabel("Diff")
axs[0].set_title("Average over Ensemble (MCLW_LW)")
axs[0].legend()

axs[1].plot(np.mean(mcv20x,axis=0),'o-',markersize=2,label="2x20",color='#c88700')
axs[1].fill_between(x20, np.mean(mcv20x,axis=0)-(np.divide(mcerr20,math.sqrt(40))), np.mean(mcv20x,axis=0)+(np.divide(mcerr20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1].plot(np.mean(mcv10x,axis=0),'o-',markersize=2,label="2x10",color='#1B2ACC')
axs[1].fill_between(x10, np.mean(mcv10x,axis=0)-(np.divide(mcerr10,math.sqrt(20))), np.mean(mcv10x,axis=0)+(np.divide(mcerr10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

axs[1].plot(np.mean(mcv5x,axis=0),'o-',markersize=2,label="2x5",color='#3F7F4C')
axs[1].fill_between(x5, np.mean(mcv5x,axis=0)-(np.divide(mcerr5,math.sqrt(10))), np.mean(mcv5x,axis=0)+(np.divide(mcerr5,math.sqrt(10))),alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')

axs[1].plot(np.mean(mcv3x,axis=0),'o-',markersize=2,label="2x3",color='#ff0000')
axs[1].fill_between(x3, np.mean(mcv3x,axis=0)-(np.divide(mcerr3,math.sqrt(6))), np.mean(mcv3x,axis=0)+(np.divide(mcerr3,math.sqrt(6))),alpha=0.2, edgecolor='#ff0000', facecolor='#ff0000')

axs[1].set_title("Average over Ensemble (MCLW_MC)")
axs[1].legend()
fig.suptitle("Neuron Participation: Pairwise MCLW")
plt.savefig("Eduardo/pairwise_participation_ensemble.png")
plt.show()


#############################################################IND#########################
#LW-IND
v3 = np.zeros((reps,nn3))
v5 = np.zeros((reps,nn5))
v10 = np.zeros((reps,nn10))
v20 = np.zeros((reps,nn20))
for i in range(reps):
    nn = np.load("Eduardo/Data/state_LW3_"+str(i)+".npy")
    nn = nn.T[4:-1]
    #find difference along given axis, then take the mean of this difference
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v3[i] = nn/max
    nn = np.load("Eduardo/Data/state_LW5_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v5[i] = nn/max
    nn = np.load("Eduardo/Data/state_LW10_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v10[i] = nn/max
    nn = np.load("Eduardo/Data/state_LW20_"+str(i)+".npy")
    nn = nn.T[4:-1]
    nn = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.sort(nn)[::-1]
    max = np.max(nn)
    v20[i] = nn/max

nerr20 = []
nerr10 = []
nerr5 = []
nerr3 = []

for i in v20.T:
    nerr20.append(np.std(i))
for i in v10.T:
    nerr10.append(np.std(i))
for i in v5.T:
    nerr5.append(np.std(i))
for i in v3.T:
    nerr3.append(np.std(i))

fig, axs = plt.subplots(2,2)
#print(len(b20x))
#print(np.mean(b20x,axis=0)-np.std(b20x,axis=0)/10)

axs[0,0].plot(np.mean(v20x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[0,0].fill_between(x20, np.mean(v20x,axis=0)-(np.divide(err20,math.sqrt(40))), np.mean(v20x,axis=0)+(np.divide(err20,math.sqrt(40))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0,0].plot(np.mean(v20,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[0,0].fill_between(x20, np.mean(v20,axis=0)-(np.divide(nerr20,math.sqrt(40))), np.mean(v20,axis=0)+(np.divide(nerr20,math.sqrt(40))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

axs[0,0].set_xlabel("Neuron (sorted)")
axs[0,0].set_ylabel("Participation")
axs[0,0].set_title("Avg 2x20")
axs[0,0].legend()
#plt.show()

axs[0,1].plot(np.mean(v10x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[0,1].fill_between(x10, np.mean(v10x,axis=0)-(np.divide(err10,math.sqrt(20))), np.mean(v10x,axis=0)+(np.divide(err10,math.sqrt(20))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[0,1].plot(np.mean(v10,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[0,1].fill_between(x10, np.mean(v10,axis=0)-(np.divide(nerr10,math.sqrt(20))), np.mean(v10,axis=0)+(np.divide(nerr10,math.sqrt(20))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[0,1].set_title("Avg 2x10")
axs[0,1].legend()
#.show()

axs[1,0].plot(np.mean(v5x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[1,0].fill_between(x5, np.mean(v5x,axis=0)-(np.divide(err5,math.sqrt(10))), np.mean(v5x,axis=0)+(np.divide(err5,math.sqrt(10))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1,0].plot(np.mean(v5,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[1,0].fill_between(x5, np.mean(v5,axis=0)-(np.divide(nerr5,math.sqrt(10))), np.mean(v5,axis=0)+(np.divide(nerr5,math.sqrt(10))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[1,0].set_title("Avg 2x5")
axs[1,0].legend()
#plt.show()

axs[1,1].plot(np.mean(v3x,axis=0),'o-',markersize=2,label="Dual",color='#c88700')
axs[1,1].fill_between(x3, np.mean(v3x,axis=0)-(np.divide(err3,math.sqrt(6))), np.mean(v3x,axis=0)+(np.divide(err3,math.sqrt(6))),alpha=0.2, edgecolor='#CC4F1B', facecolor='#c88700')

axs[1,1].plot(np.mean(v3,axis=0),'o-',markersize=2,label="Single",color='#1B2ACC')
axs[1,1].fill_between(x3, np.mean(v3,axis=0)-(np.divide(nerr3,math.sqrt(6))), np.mean(v3,axis=0)+(np.divide(nerr3,math.sqrt(6))),alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')

#plt.xlabel("Neuron (sorted)")
#plt.ylabel("Impact")
axs[1,1].set_title("Avg 2x3")
axs[1,1].legend()
fig.suptitle("Participation: MCLW_LW and LW")
plt.tight_layout()
plt.savefig("Eduardo/ParticipationLW")
plt.show()



plt.plot(np.mean(v20x,axis=0),'o-',label="Dual")
plt.plot(np.mean(v20,axis=0),'o-',label="Single")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Average over Ensemble (MCLW_LW 2x20)")
plt.legend()
plt.show()

plt.plot(np.mean(v10x,axis=0),'o-',label="Dual")
plt.plot(np.mean(v10,axis=0),'o-',label="Single")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Average over Ensemble (MCLW_LW 2x10)")
plt.legend()
plt.show()

plt.plot(np.mean(v5x,axis=0),'o-',label="Dual")
plt.plot(np.mean(v5,axis=0),'o-',label="Single")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Average over Ensemble (MCLW_LW 2x5)")
plt.legend()
plt.show()

plt.plot(np.mean(v3x,axis=0),'o-',label="Dual")
plt.plot(np.mean(v3,axis=0),'o-',label="Single")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Average over Ensemble (MCLW_LW 2x3)")
plt.legend()
plt.show()


plt.plot(v3.T,'o-')
plt.xlabel("Neuron (sorted)")
plt.ylabel("Variance")
plt.title("Individual Circuits (MCLW_MC 2x3)")
plt.show()

plt.plot(v5.T,'o-')
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Individual Circuits (MCLW_MC 2x5)")
plt.show()

plt.plot(v10.T,'o-')
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Individual Circuits (MCLW_MC 2x10)")
plt.show()

plt.plot(v20.T,'o-')
plt.xlabel("Neuron (sorted)")
plt.ylabel("Diff")
plt.title("Individual Circuits (MCLW_MC 2x20)")
plt.show()
#
# #plt.fill_between(np.arange(nn1),np.mean(b1,axis=0)-np.std(b1,axis=0),np.mean(b1,axis=0)+np.std(b1,axis=0),alpha=0.25)

# plt.boxplot([v3.T[0],v3.T[1],v3.T[2],v3.T[3]])
# plt.plot(np.arange(1,5),np.mean(v3,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x3)")
# plt.show()
#
# plt.boxplot([v5.T[0],v5.T[1],v5.T[2],v5.T[3]])
# plt.plot(np.arange(1,5),np.mean(v5,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x5)")
# plt.show()
#
# plt.boxplot([v10.T[0],v10.T[1],v10.T[2],v10.T[3]])
# plt.plot(np.arange(1,5),np.mean(v10,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x10)")
# plt.show()
#
# plt.boxplot([v20.T[0],v20.T[1],v20.T[2],v20.T[3]])
# plt.plot(np.arange(1,5),np.mean(v20,axis=0)[0:4],linewidth=2)
# plt.title("Stats over Ensemble (LW 2x20)")
# plt.show()
#
# #####################
# # RELATIONSHIP BETWEEN IMPACT AND CHANGE
# #####################
"""
plt.plot(np.mean(v20,axis=0),np.mean(b20,axis=0),'o-',label="2x20")
plt.plot(np.mean(v10,axis=0),np.mean(b10,axis=0),'o-',label="2x10")
plt.plot(np.mean(v5,axis=0),np.mean(b5,axis=0),'o-',label="2x5")
plt.plot(np.mean(v3,axis=0),np.mean(b3,axis=0),'o-',label="2x3")
plt.xlabel("Diff")
plt.ylabel("Impact")
plt.title("Average over Ensemble (MCLW_MC)")
plt.legend()
plt.show()
"""
# # #####################
# # # RELATIONSHIP BETWEEN IMPACT AND CHANGE
# # #####################
"""
reps = 100
nn3 = 2*3
nn5 = 2*5
nn10 = 2*10
nn20 = 2*20

b3x = np.zeros((reps,nn3))
b3y = np.zeros((reps,nn3))
v5x = np.zeros((reps,nn5))
v5y = np.zeros((reps,nn5))
v10x = np.zeros((reps,nn10))
v20x = np.zeros((reps,nn20))
for i in range(reps):
    nn = np.load("Data3/state_MCLW3_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    f1 = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.load("Data3/state_MCLW3_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    f2 = np.mean(np.abs(np.diff(nn)),axis=1)
    b3x[i] = (f1-f2)
    b3y[i] = (f1*f2)
    nn = np.load("Data3/state_MCLW5_LW_"+str(i)+".npy")
    nn = nn.T[4:-1]
    f1 = np.mean(np.abs(np.diff(nn)),axis=1)
    nn = np.load("Data3/state_MCLW5_MC_"+str(i)+".npy")
    nn = nn.T[4:-1]
    f2 = np.mean(np.abs(np.diff(nn)),axis=1)
    b5x[i] = (f1-f2)
    b5y[i] = (f1*f2)

plt.hist(b3x.flatten(),10,density=False,alpha=0.5,label="2x3")
plt.xlabel("Difference in Impact between the two tasks (LW-MC)")
plt.ylabel("Number of neurons")
plt.legend()
plt.show()
#


T = 0.90
T2 = 0.1
x3a = np.count_nonzero(b3x < -T)
x3b = np.count_nonzero(b3x > T)
x3c = np.count_nonzero(b3y < T2)
x5a = np.count_nonzero(b5x < -T)
x5b = np.count_nonzero(b5x > T)
x5c = np.count_nonzero(b5y < T2)
x10a = np.count_nonzero(b10x < -T)
x10b = np.count_nonzero(b10x > T)
x10c = np.count_nonzero(b10y < T2)
x20a = np.count_nonzero(b20x < -T)
x20b = np.count_nonzero(b20x > T)
x20c = np.count_nonzero(b20y < T2)

plt.plot([-1,0,1],[x20a,x20c,x20b],'o',label="2x20")
plt.plot([-1,0,1],[x10a,x10c,x10b],'o',label="2x10")
plt.plot([-1,0,1],[x5a,x5c,x5b],'o',label="2x5")
plt.plot([-1,0,1],[x3a,x3c,x3b],'o',label="2x3")
plt.legend()
plt.xlabel("Involvement (-1:LW,0:Both,1:MC) ")
plt.show()
"""
