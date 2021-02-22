import mga                  #Optimimizer
import ffann                #Controller
import mountaincar          #Task 4

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

#Mountain Car
trials_position_MC = 3 #6
trials_velocity_MC = 3 #6
total_trials_MC = trials_position_MC*trials_velocity_MC
position_range_MC = np.linspace(0.1, 0.1, num=trials_position_MC)
velocity_range_MC = np.linspace(0.01,0.01, num=trials_velocity_MC)

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)

    # # Task 4
    body = mountaincar.MountainCar()
    fit = 0.0
    for position in position_range_MC:
        for velocity in velocity_range_MC:
            body.position = position
            body.velocity = velocity
            for t in time_MC:
                #nn.step(np.concatenate((body.state(),np.zeros(2))))
                nn.step(body.state())
                f,done = body.step(stepsize_MC, nn.output())
                #f,done = body.step(stepsize_MC, np.array([nn.output()[0]]))
                fit += f
                if done:
                    break
    fitness4 = ((fit/(duration_MC*total_trials_MC)) + 1.0)/0.65
    return fitness4

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()
#ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()

np.save('average_historyMC_'+id+'.npy',ga.avgHistory)
np.save('best_historyMC_'+id+'.npy',ga.bestHistory)
np.save('best_individualMC_'+id+'.npy',bi)

