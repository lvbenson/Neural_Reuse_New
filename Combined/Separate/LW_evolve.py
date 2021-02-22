import mga                  #Optimimizer
import ffann                #Controller
#import invpend              #Task 1
#import cartpole             #Task 2
import leggedwalker         #Task 3
#import mountaincar          #Task 4

import numpy as np
import sys

id = str(sys.argv[1])
#id = 'multi'

# ANN Params
#nI = 3+4+3 #+2
nI = 4
#nI = 5
nH1 = 10 #20
nH2 = 10 #10
nO = 1
WeightRange = 15.0
BiasRange = 15.0

# EA Params
popsize = 50
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO # 115 parameters
recombProb = 0.5
mutatProb = 0.05 # 1/genesize # 1/g = 0.0086 we can make it 0.01 because with 1/g, avg seemed to trail close to best.
demeSize = 49
generations = 500 #1000 # With 150 (17hours), lots of progress, but more possible easily (with 300, 34hours);
boundaries = 0

# Task Params
duration_IP = 10.0
stepsize_IP = 0.05
duration_CP = 10.0 #50
stepsize_CP = 0.05
duration_LW = 220.0 #220.0
stepsize_LW = 0.1
duration_MC = 10.0 #220.0
stepsize_MC = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
time_MC = np.arange(0.0,duration_MC,stepsize_MC)

MaxFit = 0.627 #Leggedwalker

#Legged walker ## 9 * 2200 = 19,800 updates ==> 4 * 1100 = 4400
trials_theta = 1
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 1
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)

    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    for theta in theta_range_LW:
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:

                nn.step(body.state())
                body.step(stepsize_LW, nn.output())
                #nn.step(np.concatenate((body.state(),np.zeros(1))))))
                #body.step(stepsize_LW, np.array([nn.output()[0]]))
            fit += body.cx/duration_LW # Maximize the final forward distance covered
    fitness3 = (fit/total_trials_LW)/MaxFit
    if fitness3 < 0.0:
        fitness3 = 0.0
    return fitness3

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()
#ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()

np.save('average_historyLW_'+id+'.npy',ga.avgHistory)
np.save('best_historyLW_'+id+'.npy',ga.bestHistory)
np.save('best_individualLW_'+id+'.npy',bi)

