import mga_seq                  #Optimimizer
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import mountaincar          #Task 4

import numpy as np
import sys

#id = str(sys.argv[1])



# ANN Params
nI = 4
nH1 = 5
nH2 = 5
nO = 1
WeightRange = 15.0
BiasRange = 15.0

# EA Params
popsize = 30 ##
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO #length of the genotype
recombProb = 0.5
mutatProb = 0.05 #1/genesize
demeSize = 49 ##
generations = 300 ##
boundaries = 0 ##

# Task Params
duration_IP = 10.0
stepsize_IP = 0.05
duration_CP = 10.0
stepsize_CP = 0.05
duration_LW = 220.0
stepsize_LW = 0.05
duration_MC = 10.0
stepsize_MC = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
time_MC = np.arange(0.0,duration_MC,stepsize_MC)

MaxFit = 0.627 #Leggedwalker

trials_theta_IP = 3
trials_thetadot_IP = 3
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

trials_theta_CP = 3
trials_thetadot_CP = 3
trials_x_CP = 1
trials_xdot_CP = 1
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(0.0, 0.0, num=trials_x_CP)
xdot_range_CP = np.linspace(0.0, 0.0, num=trials_xdot_CP)

trials_theta_LW = 1
trials_omega_LW = 1
theta_range_LW = np.linspace(0.0, 0.0, num=trials_theta_LW)
omega_range_LW = np.linspace(0.0, 0.0, num=trials_omega_LW)
total_trials_LW = trials_theta_LW * trials_omega_LW

trials_position_MC = 3
trials_velocity_MC = 3
total_trials_MC = trials_position_MC*trials_velocity_MC
position_range_MC = np.linspace(0.1, 0.1, num=trials_position_MC)
velocity_range_MC = np.linspace(0.01,0.01, num=trials_velocity_MC)

#

# Fitness function
def fitnessFunction(genotype):
    #print(genotype,'genotype')

    # Create neural network
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)

    # Task 1

    body = mountaincar.MountainCar()
    fit = 0.0
    for position in position_range_MC:
        for velocity in velocity_range_MC:
            body.position = position
            body.velocity = velocity
            for t in time_MC:
                nn.step(body.state())
                f,done = body.step(stepsize_MC, nn.output())
                fit += f
                if done:
                    break
    fitness1 = ((fit/(duration_MC*total_trials_MC)) + 1.0)/0.65

    # Task 2

    body = cartpole.Cartpole()
    fit = 0.0
    for theta in theta_range_CP:
        for theta_dot in thetadot_range_CP:
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    for t in time_CP:
                        nn.step(body.state())
                        f,done = body.step(stepsize_CP, nn.output())
                        fit += f
                        if done:
                            break
    fitness2 = fit/(duration_CP*total_trials_CP)
    if fitness2 < 0.0:
        fitness2 = 0.0
#

    fitnesses = [fitness1,fitness1*fitness2]
    
    return fitnesses

# Evolve and visualize fitness over generations
# 
#
#

#f = fitnessFunction()

for id in range(0,50):

    ga = mga_seq.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
    ga.run(0)
    af,bf,bi = ga.fitStats()
    if bf < 0.05:
        #print('Not good enough')
        pass
    else:
        #print('good enough')
        ga.run(1)
        af,bf,bi = ga.fitStats()

        #ga.showFitness()
        np.save('./Combined/Cog_Models_Project/Data/Model2/average_history_'+str(id)+'.npy',ga.avgHistory)
        np.save('./Combined/Cog_Models_Project/Data/Model2/best_history_'+str(id)+'.npy',ga.bestHistory)
        np.save('./Combined/Cog_Models_Project/Data/Model2/best_individual_'+str(id)+'.npy',bi)



        #/Users/lvbenson/Research_Projects/Neural_Reuse_New/Combined/Cog_Models_Project/Data