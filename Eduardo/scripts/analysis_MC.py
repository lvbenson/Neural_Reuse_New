import numpy as np
import ffann                #Controller
import mountaincar          #Task 4
import matplotlib.pyplot as plt
import sys
import os

dir = "Data3"
id = str(sys.argv[1])
condition = str(sys.argv[2])
nh = int(sys.argv[3])
steps = int(sys.argv[4])

# ANN Params
nI = 4
nH1 = nh
nH2 = nh
nO = 1
WeightRange = 15.0
BiasRange = 15.0

#####################################
# MOUNTAIN CAR
duration_MC = 10.0 #220.0
stepsize_MC = 0.05
time_MC = np.arange(0.0,duration_MC,stepsize_MC)
trials_position_MC = 3 #6
trials_velocity_MC = 3 #6
total_trials_MC = trials_position_MC*trials_velocity_MC
position_range_MC = np.linspace(0.1, 0.1, num=trials_position_MC)
velocity_range_MC = np.linspace(0.01,0.01, num=trials_velocity_MC)

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(1)
    #Task 4
    body = mountaincar.MountainCar()
    nn_state_mc = np.zeros((total_trials_MC*len(time_MC),nI+nH1+nH2+nO))
    total_steps = len(position_range_MC) * len(velocity_range_MC) * len(time_MC)
    fit_MC = np.zeros((len(position_range_MC),len(velocity_range_MC)))
    i = 0
    k = 0
    for position in position_range_MC:
        j = 0
        for velocity in velocity_range_MC:
            body.position = position
            body.velocity = velocity
            fit = 0.0
            for t in time_MC:
                nn.step(body.state())
                nn_state_mc[k] = nn.states()
                k += 1
                f,done = body.step(stepsize_MC, nn.output())
                fit += f
                if done:
                    break
            fit_MC[i][j] = ((fit/duration_MC) + 1.0)/0.65
            j += 1
        i += 1
    fitness = np.mean(fit_MC)
    return fitness,fit_MC,nn_state_mc

def lesions(genotype,actvalues):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    #Task 4
    body = mountaincar.MountainCar()
    nn.setParameters(genotype,WeightRange,BiasRange)
    mc_fit = np.zeros(nH1+nH2)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            fit_act = np.zeros(len(actvalues[:,n]))
            if abs(actvalues[-1,n] - actvalues[0,n]) < 1.0:
                act = actvalues[0,n]
                fit_MC = np.zeros((len(position_range_MC),len(velocity_range_MC)))
                i = 0
                for position in position_range_MC:
                    j = 0
                    for velocity in velocity_range_MC:
                        body.position = position
                        body.velocity = velocity
                        fit = 0.0
                        for t in time_MC:
                            nn.step_lesioned(body.state(),neuron,layer,act)
                            f,done = body.step(stepsize_MC, nn.output())
                            fit += f
                            if done:
                                break
                        fit_MC[i][j] = ((fit/duration_MC) + 1.0)/0.65
                        j += 1
                    i += 1
                mc_fit[index] = np.mean(fit_MC)
                index += 1
            else:
                k =  0
                for act in actvalues[:,n]:
                    fit_MC = np.zeros((len(position_range_MC),len(velocity_range_MC)))
                    i = 0
                    for position in position_range_MC:
                        j = 0
                        for velocity in velocity_range_MC:
                            body.position = position
                            body.velocity = velocity
                            fit = 0.0
                            for t in time_MC:
                                nn.step_lesioned(body.state(),neuron,layer,act)
                                f,done = body.step(stepsize_MC, nn.output())
                                fit += f
                                if done:
                                    break
                            fit_MC[i][j] = ((fit/duration_MC) + 1.0)/0.65
                            j += 1
                        i += 1
                    fit_act[k] = np.mean(fit_MC)
                    k += 1
                mc_fit[index] = np.max(fit_act)
                index += 1
    print(mc_fit)
    return mc_fit

def find_all_lesions(dir,ind,steps):
    nn = np.load(dir+"/state_"+condition+"_MC_"+str(ind)+".npy")
    max = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
    min = np.min(nn[:,nI:nI+nH1+nH2],axis=0)
    actvalues = np.linspace(min, max, num=steps)
    bi = np.load(dir+"/best_individual"+condition+"_"+str(ind)+".npy")
    f = np.load(dir+"/perf_"+condition+"_MC_"+str(ind)+".npy")
    mcp = lesions(bi,actvalues)
    mcp = mcp/f
    np.save(dir+"/lesions_"+condition+"_MC_"+str(steps)+"_"+str(ind)+".npy",mcp)
    # Stats on neurons for Ablations
    Threshold = 0.95
    count = np.zeros(2)
    for mc_neuron in mcp:
        if mc_neuron > Threshold: # no task neurons
            count[0] += 1 #no tasks
        if mc_neuron <= Threshold: # ip task neurons
            count[1] += 1 #CP task
    np.save(dir+"/stats_"+condition+"_MC_"+str(ind)+".npy",count)

bf = np.load(dir+"/best_history"+condition+"_"+str(id)+".npy")
bi = np.load(dir+"/best_individual"+condition+"_"+str(id)+".npy")

if bf[-1] > 0.0:
    f,m1,ns1=analysis(bi)
    np.save(dir+"/perf_"+condition+"_MC_"+str(id)+".npy",f)
    np.save(dir+"/perfmap_"+condition+"_MC_"+str(id)+".npy",m1)
    np.save(dir+"/state_"+condition+"_MC_"+str(id)+".npy",ns1)
    find_all_lesions(dir,id,steps)

# l0=np.load(dir+"/lesions_"+condition+"_"+str(id)+".npy")
# # l1=np.load(dir+"/lesions_"+condition+"_10_"+str(id)+".npy")
# # l2=np.load(dir+"/lesions_"+condition+"_20_"+str(id)+".npy")
# # l3=np.load(dir+"/lesions_"+condition+"_30_"+str(id)+".npy")
# l4=np.load(dir+"/lesions_"+condition+"_40_"+str(id)+".npy")
# # plt.plot(l1,'o',label="10")
# # plt.plot(l2,'o',label="20")
# # plt.plot(l3,'o',label="30")
# plt.plot(l4,'o',label="40")
# plt.plot(l0,'ko',label="Original")
# plt.xlabel("Neuron")
# plt.ylabel("Lesion")
# plt.legend()
# plt.show()
