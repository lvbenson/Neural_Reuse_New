
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

dir = str(sys.argv[1])
start = int(sys.argv[2])
finish = int(sys.argv[3])
reps = finish-start

# ANN Params
nI = 4
nH1 = 10
nH2 = 10
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
#Inverted Pendulum
trials_theta_IP = 7
trials_thetadot_IP = 7
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(1)

    # Task 1
    body = invpend.InvPendulum()
    nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
    total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
    fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
    i=0
    k=0
    for theta in theta_range_IP:
        j=0
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            f = 0.0
            for t in time_IP:
                #nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3),np.zeros(2)))) #arrays for inputs for each task
                #nn.step(np.concatenate((body.state(),np.zeros(1))))
                nn.step(body.state())
                nn_state_ip[k] = nn.states()
                k += 1
                #f += body.step(stepsize_IP, np.array([nn.output()[0]]))
                f += body.step(stepsize_IP,nn.output())
            fit_IP[i][j] = ((f/duration_IP)+7.65)/7
            j += 1
        i += 1
    fitness = np.mean(fit_IP)
   
    #return fitness1

    
    
    return fitness,fit_IP,nn_state_ip

def lesions(genotype,actvalues):

    nn = ffann.ANN(nI,nH1,nH2,nO)

    # Task 1
    ip_fit = np.zeros(nH1+nH2)
    body = invpend.InvPendulum()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
           # print("IP:",n)
            maxfit = 0.0
            for act in actvalues[:,0,n]:
                fit = 0.0
                for theta in theta_range_IP:
                    for theta_dot in thetadot_range_IP:
                        body.theta = theta
                        body.theta_dot = theta_dot
                        for t in time_IP:
                            #nn.step_lesioned(np.concatenate((body.state(),np.zeros(4),np.zeros(3),np.zeros(2))),neuron,layer,act)
                            #nn.step(np.concatenate((body.state(),np.zeros(1))))
                            nn.step_lesioned(body.state(),neuron,layer,act)
                            #nn.step(body.state())
                            f = body.step(stepsize_IP, nn.output())
                            fit += f
                fit = fit/(duration_IP*total_trials_IP)
                fit = (fit+7.65)/7
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            ip_fit[index]=fit
            index += 1

    # Task 2
   
    

    return ip_fit

def find_all_lesions(dir,ind):
    max = np.zeros((1,nH1+nH2))
    #/Users/lvbenson/Research_Projects/Neural_Reuse_New/Combined/4T_2x5/Data/state_IP_0.npy
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    max[0] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)


    steps = 20
    actvalues = np.linspace(0.0, max, num=steps)

    bi = np.load("./{}/best_individualIP_{}.npy".format(dir,ind))
    f = np.load("./{}/perf_{}.npy".format(dir,ind))

    ipp = lesions(bi,actvalues)

    ipp = ipp/f


    np.save(dir+"/lesions_IP_"+str(ind)+".npy",ipp)


    # Stats on neurons for Ablations
    Threshold = 0.95
    count = np.zeros(2)
    for ip_neuron in ipp:
        if ip_neuron > Threshold: # no task neurons
            count[0] += 1 #no tasks
        if ip_neuron <= Threshold: # ip task neurons
            count[1] += 1 #IP task


    np.save(dir+"/stats_"+str(ind)+".npy",count)


def find_all_var(dir,ind):
    nI = 4
    nH = 10
    v = np.zeros((1,10))
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    v[0] = np.var(nn[:,nI:nI+nH],axis=0)
    max = np.max(v,axis=0)
    norm_var = np.zeros((10,4))
    for i in range(10):
        if max[i] > 0.0:
            norm_var[i] = v.T[i]/max[i]
        else:
            norm_var[i] = 0.0
    norm_var = norm_var.T
    np.save("./{}/NormVar_IP_{}.npy".format(dir,ind), norm_var[0])


gens = len(np.load(dir+"/average_historyIP_0.npy"))
#print(gens)
test = np.load(dir+"/average_historyIP_0.npy")
#print(test)
gs=len(np.load(dir+"/best_individualIP_0.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))

index = 0
count = 0
#plt.figure(figsize=[6, 4])
for i in range(start,finish):
    af[index] = np.load(dir+"/average_historyIP_"+str(i)+".npy")
    bf[index] = np.load(dir+"/best_historyIP_"+str(i)+".npy")
    bi[index] = np.load(dir+"/best_individualIP_"+str(i)+".npy")
    #evol_fit = bf[index][-1]**(1/4)
    if bf[index][-1] > 0.8:

        #plt.scatter(np.arange(1, 11), evol_fit, c='blue',s=5, alpha=0.7)
        count += 1
        f,m1,ns1=analysis(bi[index])
        np.save(dir+"/perf_"+str(i)+".npy",f)

        np.save(dir+"/perfmap_IP_"+str(i)+".npy",m1)

        np.save(dir+"/state_IP_"+str(i)+".npy",ns1)

        
        find_all_lesions(dir,i)
        
    index += 1
   