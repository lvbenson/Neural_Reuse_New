import numpy as np
#import infotheory
import ffann                #Controller
import invpend              #Task 1
#import cartpole             #Task 2
import leggedwalker         #Task 3
import mountaincar          #Task 4
import matplotlib.pyplot as plt
import sys
import os

dir = "C:/Users/benso/Desktop/Projects/Neural_Reuse/Neural_Reuse_New/Combined/MC_Reuse/Data_2x5"
#id = str(sys.argv[1])
reps = 100

# ANN Params
nI = 4
nH1 = 5
nH2 = 5
nO = 1
WeightRange = 15.0
BiasRange = 15.0

# Task Params
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 10 #50
stepsize_CP = 0.05
duration_LW = 220 #220.0
stepsize_LW = 0.05
duration_MC = 10.0 #220.0
stepsize_MC = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
time_MC = np.arange(0.0,duration_MC,stepsize_MC)


MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges

#Mountain Car
trials_position_MC = 6 #6
trials_velocity_MC = 6 #6
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
                f,d = body.step(stepsize_MC, nn.output())
                fit += f
            fit_MC[i][j] = ((fit/duration_MC) + 1.0)/0.65
            j += 1
        i += 1
    fitness = np.mean(fit_MC)
    
    
    return fitness,fit_MC,nn_state_mc

def lesions(genotype,actvalues):

    nn = ffann.ANN(nI,nH1,nH2,nO)
    
    
    #Task 4        
    mc_fit = np.zeros(nH1+nH2)
    body = mountaincar.MountainCar()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer ==1:
                n = neuron
            else:
                n = nH1 + neuron
            #print("MC:",n)
            maxfit = 0.0
            for act in actvalues[:,0,n]:
                fit = 0.0
                for position in position_range_MC:
                    for velocity in velocity_range_MC:
                        body.position = position
                        body.velocity = velocity
                        for t in time_MC:
                            #nn.step_lesioned(np.concatenate((body.state(),np.zeros(2))),neuron,layer,act)
                            nn.step_lesioned(body.state(),neuron,layer,act)
                            f,d = body.step(stepsize_MC, nn.output())
                            fit += f
                fit = ((fit/duration_MC) + 1.0)/0.65
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            mc_fit[index] = fit
            index += 1
                        
    

    return mc_fit

def find_all_lesions(dir,ind):
    max = np.zeros((1,nH1+nH2))
    #/Users/lvbenson/Research_Projects/Neural_Reuse_New/Combined/4T_2x5/Data/state_IP_0.npy
    nn = np.load(dir+"/state_MC_"+str(ind)+".npy")
    max[0] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)

    steps = 10
    actvalues = np.linspace(0.0, max, num=steps)

    bi = np.load(dir+"/best_individualMC_"+str(ind)+".npy")
    #bi = np.load("./{}/best_individualCP_{}.npy".format(dir,ind))
    f = np.load(dir+"/perf_"+str(ind)+".npy")

    mcp = lesions(bi,actvalues)

    mcp = mcp/f

    np.save(dir+"/lesions_MC_"+str(ind)+".npy",mcp)

    # Stats on neurons for Ablations
    Threshold = 0.95
    count = np.zeros(2)
    for mc_neuron in mcp:
        if mc_neuron > Threshold: # no task neurons
            count[0] += 1 #no tasks
        if mc_neuron <= Threshold: # ip task neurons
            count[1] += 1 #CP task
        
    np.save(dir+"/stats_"+str(ind)+".npy",count)

gens = len(np.load(dir+"/average_historyMC_0.npy"))
#print(gens)
gs=len(np.load(dir+"/best_individualMC_0.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))

index = 0
count = 0

for i in range(0,100):

    af[index] = np.load(dir+"/average_historyMC_"+str(i)+".npy")
    bf[index] = np.load(dir+"/best_historyMC_"+str(i)+".npy")
    bi[index] = np.load(dir+"/best_individualMC_"+str(i)+".npy")

    count += 1

    f,m1,ns1=analysis(bi[index])
        
    np.save(dir+"/perf_"+str(i)+".npy",f)
        #print(f,'analysis performance')
        
        

    np.save(dir+"/perfmap_MC_"+str(i)+".npy",m1)

    np.save(dir+"/state_MC_"+str(i)+".npy",ns1)


        
    find_all_lesions(dir,i)

index += 1