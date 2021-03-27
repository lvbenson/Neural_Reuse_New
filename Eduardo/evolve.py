import mga                  #Optimimizer
import ffann                #Controller
import leggedwalker         #Task 3
import mountaincar          #Task 4
import numpy as np
import sys

task = str(sys.argv[1])
nh = int(sys.argv[2])
id = str(sys.argv[3])

# ANN Params
nI = 4
nH1 = nh
nH2 = nh
nO = 1
WeightRange = 15.0
BiasRange = 15.0

# EA Params
popsize = 50
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05
demeSize = 2
generations = 1000 #500
boundaries = 0

# Task Params
# duration_IP = 10.0
# stepsize_IP = 0.05
# duration_CP = 10.0
# stepsize_CP = 0.05
# time_IP = np.arange(0.0,duration_IP,stepsize_IP)
# time_CP = np.arange(0.0,duration_CP,stepsize_CP)

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

def fitnessFunctionMC(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = mountaincar.MountainCar()
    fit = 0.0
    for position in position_range_MC:
        for velocity in velocity_range_MC:
            body.position = position
            body.velocity = velocity
            for t in time_MC:
                nn.step(body.state())
                f,done = body.step(stepsize_MC, np.array([nn.output()[0]]))
                fit += f
                if done:
                    break
    return ((fit/(duration_MC*total_trials_MC)) + 1.0)/0.65
#####################################

#####################################
# LEGGED WALKER
duration_LW = 220.0 #220.0
stepsize_LW = 0.1
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
MaxFit = 0.627 #Leggedwalker
trials_theta = 1
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 1
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

def fitnessFunctionLW(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    for theta in theta_range_LW:
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(body.state())
                body.step(stepsize_LW, np.array([nn.output()[0]]))
            fit += body.cx/duration_LW
    fitness = (fit/total_trials_LW)/MaxFit
    if fitness < 0.0:
        fitness = 0.0
    return fitness
#####################################

def fitnessFunctionMCLW(genotype):
    return (fitnessFunctionMC(genotype) * fitnessFunctionLW(genotype))

#####################################
# MAIN
if task == "MC":
    fitnessFunction =fitnessFunctionMC
elif task == "LW":
    fitnessFunction = fitnessFunctionLW
elif task == "MCLW":
    fitnessFunction = fitnessFunctionMCLW
else:
    print("No task defined.")

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()
np.save('best_history'+task+str(nh)+'_'+id+'.npy',ga.bestHistory)
np.save('best_individual'+task+str(nh)+'_'+id+'.npy',bi)
