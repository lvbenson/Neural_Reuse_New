import mga                  #Optimimizer
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
#import mountaincar          #Task 4

import numpy as np
import sys

#id = str(sys.argv[1])
id = 2

# ANN Params
#nI = 3+4+3 #+2
nI = 7
#nI = 5
nH1 = 5 #20
nH2 = 5 #10
nO = 1+1+3 #+1 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

# EA Params
popsize = 2
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO # 115 parameters
recombProb = 0.5
mutatProb = 0.05 # 1/genesize # 1/g = 0.0086 we can make it 0.01 because with 1/g, avg seemed to trail close to best.
demeSize = 49
generations = 5 #1000 # With 150 (17hours), lots of progress, but more possible easily (with 300, 34hours);
boundaries = 0

# Task Params
duration_IP = 10.0
stepsize_IP = 0.05
duration_CP = 10.0 #50
stepsize_CP = 0.05
duration_LW = 220.0 #220.0
stepsize_LW = 0.1
# duration_MC = 10.0 #220.0
# stepsize_MC = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)
# time_MC = np.arange(0.0,duration_MC,stepsize_MC)

MaxFit = 0.627 #Leggedwalker

## 43000 updates 9200
# Fitness initialization ranges
#Inverted Pendulum ## 36*200 = 7,200 updates ==> 4 * 200 = 800
trials_theta_IP = 3 #6
trials_thetadot_IP = 3 #6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

#Cartpole ## 16 * 1000 = 16,000 updates ==> 4 * 1000 = 4000
trials_theta_CP = 3 #2
trials_thetadot_CP = 3 #2
trials_x_CP = 1 #2
trials_xdot_CP = 1 #2
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(0.0, 0.0, num=trials_x_CP) #x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(0.0, 0.0, num=trials_xdot_CP) #xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

#Legged walker ## 9 * 2200 = 19,800 updates ==> 4 * 1100 = 4400
trials_theta = 3
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 3
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

# Mountain Car
# trials_position_MC = 3 #6
# trials_velocity_MC = 3 #6
# total_trials_MC = trials_position_MC*trials_velocity_MC
# position_range_MC = np.linspace(0.1, 0.1, num=trials_position_MC)
# velocity_range_MC = np.linspace(0.01,0.01, num=trials_velocity_MC)

# Fitness function (task1, sequential)
def fitnessFunctionInv(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    #nn = ffann.ANN(nI,nH1,nH2,nO)
    fitnessFunctionInv.circuit = nn
    nn.setParameters(genotype,WeightRange,BiasRange)
    # Task 1
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time_IP:
                #create single array for shared inputs.
                st = body.state()[0]
                ct = body.state()[1]
                td = body.state()[2]
                i = np.concatenate((st,ct,td,np.zeros(4)), axis=None) #sintheta, costheta, thetadot, ....
                nn.step(i)

                #combine sensory information
                #input task 1: sintheta, costheta, thetadot
                #input task 2: theta, thetadot, x, xdot
                #input task 3: theta, thetadot, footstate
                #inputs necessary: 1 for sintheta, 1 for costheta, 1 for thetadot, 1 for theta, 1 for x, 1 for xdot, 1 for footstate (7 total)

                #output task 1:
                #output task 2:
                #output task 3:

                f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                fit += f    # Minimize the cost of moving the pole up
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    if fitness1 < 0.0:
        fitness1 = 0.0

    return fitness1

gi = mga.Microbial(fitnessFunctionInv, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
gi.run()

#fitness function for combined tasks 1 and 2, seq training 

def fitnessFunctionIPCP(genotype):
    #nn = ffann.ANN(nI,nH1,nH2,nO)
    nn = fitnessFunctionInv.circuit
    nn.setParameters(genotype,WeightRange,BiasRange)
    # Task 1
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time_IP:
                #create single array for shared inputs.
                st = body.state()[0]
                ct = body.state()[1]
                td = body.state()[2]
                i = np.concatenate((st,ct,td,np.zeros(4)), axis=None) #sintheta, costheta, thetadot, ....
                nn.step(i)

                #combine sensory information
                #input task 1: sintheta, costheta, thetadot
                #input task 2: theta, thetadot, x, xdot
                #input task 3: theta, thetadot, footstate
                #inputs necessary: 1 for sintheta, 1 for costheta, 1 for thetadot, 1 for theta, 1 for x, 1 for xdot, 1 for footstate (7 total)

                #output task 1:
                #output task 2:
                #output task 3:

                f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                fit += f    # Minimize the cost of moving the pole up
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    if fitness1 < 0.0:
        fitness1 = 0.0


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

                        t_c = body.state()[0]
                        td_c = body.state()[1]
                        x = body.state()[2]
                        xd = body.state()[3]

                        c = np.concatenate((np.zeros(2),td_c,t_c,x,xd,np.zeros(1)), axis = None) #_, _ , thetadot, theta, x, xdot
                        nn.step(c)

                        f,done = body.step(stepsize_CP, np.array([nn.output()[1]]))
                        fit += f  # (Maximize) Amount of time pole is balanced
                        ###
                        if done:
                            break
    fitness2 = fit/(duration_CP*total_trials_CP)
    if fitness2 < 0.0:
        fitness2 = 0.0
    
    return fitness1*fitness2

ga = mga.Microbial(fitnessFunctionIPCP, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()



def fitnessFunction(genotype):
    #nn = ffann.ANN(nI,nH1,nH2,nO)
    nn = fitnessFunctionInv.circuit
    nn.setParameters(genotype,WeightRange,BiasRange)
    # Task 1
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time_IP:
                #create single array for shared inputs.
                st = body.state()[0]
                ct = body.state()[1]
                td = body.state()[2]
                i = np.concatenate((st,ct,td,np.zeros(4)), axis=None) #sintheta, costheta, thetadot, ....
                nn.step(i)

                #combine sensory information
                #input task 1: sintheta, costheta, thetadot
                #input task 2: theta, thetadot, x, xdot
                #input task 3: theta, thetadot, footstate
                #inputs necessary: 1 for sintheta, 1 for costheta, 1 for thetadot, 1 for theta, 1 for x, 1 for xdot, 1 for footstate (7 total)

                #output task 1:
                #output task 2:
                #output task 3:

                f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                fit += f    # Minimize the cost of moving the pole up
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    if fitness1 < 0.0:
        fitness1 = 0.0

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

                        t_c = body.state()[0]
                        td_c = body.state()[1]
                        x = body.state()[2]
                        xd = body.state()[3]

                        c = np.concatenate((np.zeros(2),td_c,t_c,x,xd,np.zeros(1)), axis = None) #_, _ , thetadot, theta, x, xdot
                        nn.step(c)

                        f,done = body.step(stepsize_CP, np.array([nn.output()[1]]))
                        fit += f  # (Maximize) Amount of time pole is balanced
                        ###
                        if done:
                            break
    fitness2 = fit/(duration_CP*total_trials_CP)
    if fitness2 < 0.0:
        fitness2 = 0.0
  

    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    for theta in theta_range_LW:
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:

                #input leggedwalker: theta, thetadot, footstate

                t_l = body.state()[0]
                td_l = body.state()[1]
                f_l = body.state()[2]

                l = np.concatenate((np.zeros(2),td_l,t_l,np.zeros(2),f_l), axis = None)
                nn.step(l)
                #nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                #print(nn.output()[2:5])
                #lw_o1 = nn.output()[2]
                #lw_o2 = nn.output()[3]
                #lw_o3 = nn.output()[4]
                #body.step(stepsize_LW, np.array(lw_o1,np.zeros(1),lw_o2,lw_o3))
                #body.step(stepsize_LW, np.array(nn.output()[0],nn.output()[2:4]))
                body.step(stepsize_LW, np.array(nn.output()[2:5]))

            fit += body.cx/duration_LW # Maximize the final forward distance covered
    fitness3 = (fit/total_trials_LW)/MaxFit
    if fitness3 < 0.0:
        fitness3 = 0.0

    # # Task 4
    # body = mountaincar.MountainCar()
    # fit = 0.0
    # for position in position_range_MC:
    #     for velocity in velocity_range_MC:
    #         body.position = position
    #         body.velocity = velocity
    #         for t in time_MC:
    #             nn.step(np.concatenate((np.zeros(3),np.zeros(4),np.zeros(3),body.state())))
    #             f,done = body.step(stepsize_MC, np.array([nn.output()[5]]))
    #             fit += f
    #             if done:
    #                 break
    # fitness4 = ((fit/(duration_MC*total_trials_MC)) + 1.0)/0.65
    # return fitness1*fitness2*fitness3*fitness4
    return fitness1*fitness2*fitness3
    #return fitness1,fitness2,fitness3

# Evolve and visualize fitness over generations
#this is where sequential training occurs

ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()
#ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()

np.save('average_history_'+id+'.npy',ga.avgHistory)
np.save('best_history_'+id+'.npy',ga.bestHistory)
np.save('best_individual_'+id+'.npy',bi)

#incremental evolution
#for EACH evlutionary run, evolve incrementally so that I'm only evolving ONCE,
#first, evolve for A, if A reaches a certain threshold, then add 2, if B reaches
#certain threshold, then add C
#within an ensemble, successful e. runs will have passed all three thresholds 
#idea: set up parameterized fitness function
#ga = mga.Microbial(fitnessFunction(1,0,0,0), popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
