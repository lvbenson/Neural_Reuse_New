import numpy as np
import ffann                #Controller
import leggedwalker          #Task 3
import matplotlib.pyplot as plt
import sys
import os

dir = "Data"
id = str(sys.argv[1])
condition = str(sys.argv[2])
nh = int(sys.argv[3])
#steps = int(sys.argv[4])

# ANN Params
nI = 4
nH1 = nh
nH2 = nh
nO = 1
WeightRange = 15.0
BiasRange = 15.0

#####################################
# LEGGED WALKER
duration_LW = 2*220.0 #220.0
stepsize_LW = 0.1
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
MaxFit = 0.627 #Leggedwalker
trials_theta_LW = 1
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta_LW)
trials_omega_LW = 1
omega_range_LW = np.linspace(-1, 1, num=trials_omega_LW)
total_trials_LW = trials_theta_LW * trials_omega_LW

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(1)
    #Task 3
    nn_state_lw = np.zeros((total_trials_LW*len(time_LW),nI+nH1+nH2+nO))
    fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
    body = leggedwalker.LeggedAgent(0.0,0.0)
    i = 0
    k = 0
    for theta in theta_range_LW:
        j = 0
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(body.state())
                nn_state_lw[k] = nn.states()
                k += 1
                body.step(stepsize_LW, np.array([nn.output()[0]]))
            fitness =  body.cx/duration_LW
            if fitness < 0.0:
                fitness = 0.0
            fit_LW[i][j] = fitness/MaxFit
            j += 1
        i += 1
    fitness = np.mean(fit_LW)
    return fitness,fit_LW,nn_state_lw

def lesions(genotype,actvalues):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    lw_fit = np.zeros(nH1+nH2)
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
                fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
                i = 0
                for theta in theta_range_LW:
                    j = 0
                    for omega in omega_range_LW:
                        body.reset()
                        body.angle = theta
                        body.omega = omega
                        fit = 0.0
                        for t in time_LW:
                            nn.step_lesioned(body.state(),neuron,layer,act)
                            body.step(stepsize_LW, np.array([nn.output()[0]]))
                        fitness =  body.cx/duration_LW
                        if fitness < 0.0:
                            fitness = 0.0
                        fit_LW[i][j] = fitness/MaxFit
                        j += 1
                    i += 1
                lw_fit[index] = np.mean(fit_LW)
                index += 1
            else:
                k =  0
                for act in actvalues[:,n]:
                    fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
                    i = 0
                    for theta in theta_range_LW:
                        j = 0
                        for omega in omega_range_LW:
                            body.reset()
                            body.angle = theta
                            body.omega = omega
                            fit = 0.0
                            for t in time_LW:
                                nn.step_lesioned(body.state(),neuron,layer,act)
                                body.step(stepsize_LW, np.array([nn.output()[0]]))
                            fitness =  body.cx/duration_LW
                            if fitness < 0.0:
                                fitness = 0.0
                            fit_LW[i][j] = fitness/MaxFit
                            j += 1
                        i += 1
                    fit_act[k] = np.mean(fit_LW)
                    k += 1
                lw_fit[index] = np.max(fit_act)
                index += 1
    return lw_fit

def find_all_lesions(dir,ind,steps):
    nn = np.load(dir+"/state_"+condition+"_LW_"+str(ind)+".npy")
    max = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
    min = np.min(nn[:,nI:nI+nH1+nH2],axis=0)
    actvalues = np.linspace(min, max, num=steps)
    bi = np.load(dir+"/best_individual"+condition+"_"+str(ind)+".npy")
    f = np.load(dir+"/perf_"+condition+"_LW_"+str(ind)+".npy")
    mcp = lesions(bi,actvalues)
    mcp = mcp/f
    np.save(dir+"/lesions_"+condition+"_LW_"+str(steps)+"_"+str(ind)+".npy",mcp)
    # Stats on neurons for Ablations
    Threshold = 0.95
    count = np.zeros(2)
    for mc_neuron in mcp:
        if mc_neuron > Threshold: # no task neurons
            count[0] += 1 #no tasks
        if mc_neuron <= Threshold: # ip task neurons
            count[1] += 1 #CP task
    np.save(dir+"/stats_"+condition+"_LW_"+str(ind)+".npy",count)


def find_all_var(dir,ind):
    nI = 4
    nH = 10
    v = np.zeros((1,10))
    nn = np.load(dir+"/state_LW5_"+str(ind)+".npy")
    v = np.var(nn[:,nI:nI+nH],axis=0)
    """
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    v[0] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_CP_{}.npy".format(dir,ind))
    v[1] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_LW_{}.npy".format(dir,ind))
    v[2] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_MC_{}.npy".format(dir,ind))
    v[3] = np.var(nn[:,nI:nI+nH],axis=0)
    """
    max = np.max(v,axis=0)
    norm_var = np.zeros((10,1))
    for i in range(10):
        if max[i] > 0.0:
            norm_var[i] = v.T[i]/max[i]
        else:
            norm_var[i] = 0.0
    norm_var = norm_var.T
    print(norm_var)

    np.save(dir+"/NormVar_"+"LW5_40_"+str(ind)+".npy",norm_var)


bf = np.load(dir+"/best_history"+condition+"_"+str(id)+".npy")
bi = np.load(dir+"/best_individual"+condition+"_"+str(id)+".npy")

if bf[-1] > 0.0:
    f,m1,ns1=analysis(bi)
    np.save(dir+"/perf_"+condition+"_LW_"+str(id)+".npy",f)
    np.save(dir+"/perfmap_"+condition+"_LW_"+str(id)+".npy",m1)
    np.save(dir+"/state_"+condition+"_LW_"+str(id)+".npy",ns1)
    #find_all_lesions(dir,id,steps)
    find_all_var(dir,id)
