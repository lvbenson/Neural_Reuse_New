import os
import numpy as np
import infotheory
import ffann  # Controller
import invpend  # Task 1
import cartpole  # Task 2
import leggedwalker  # Task 3
import mountaincar #task 4
import matplotlib.pyplot as plt
import sys

# ANN Params
nI = 4
nH1 = 10
nH2 = 10
nO = 1  # output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

# Task Params
duration_IP = 15.0
stepsize_IP = 0.05
duration_CP = 15.0  # 50
stepsize_CP = 0.05
duration_LW = 220  # 220.0
stepsize_LW = 0.1
duration_MC = 15.0 #220.0
stepsize_MC = 0.05
time_IP = np.arange(0.0, duration_IP, stepsize_IP)
time_CP = np.arange(0.0, duration_CP, stepsize_CP)
time_LW = np.arange(0.0, duration_LW, stepsize_LW)
time_MC = np.arange(0.0, duration_MC, stepsize_MC)


MaxFit = 0.627  # Leggedwalker

# Fitness initialization ranges
# Inverted Pendulum
trials_theta_IP = 3
trials_thetadot_IP = 3
total_trials_IP = trials_theta_IP * trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0, 1.0, num=trials_thetadot_IP)

# Cartpole
trials_theta_CP = 3
trials_thetadot_CP = 3
trials_x_CP = 1
trials_xdot_CP = 1
total_trials_CP = trials_theta_CP * trials_thetadot_CP * trials_x_CP * trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
#theta_range_CP = np.linspace(-np.deg2rad(6), np.deg2rad(6), num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

# Legged walker
trials_theta_LW = 1
theta_range_LW = np.linspace(-np.pi / 6, np.pi / 6, num=trials_theta_LW)
trials_omega_LW = 1
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta_LW * trials_omega_LW


#Mountain Car
trials_position_MC = 3
trials_velocity_MC = 3
total_trials_MC = trials_position_MC * trials_velocity_MC
position_range_MC = np.linspace(-0.1, 0.1, num=trials_position_MC)
velocity_range_MC = np.linspace(-0.01,0.01, num=trials_velocity_MC)



# Fitness function
def simulate_individual(save_dir, run_num):
    # Common setup
    genotype = np.load("./Combined/4T_2x10/Data/best_individualS1_" + "1" + ".npy")
    nn = ffann.ANN(nI, nH1, nH2, nO)
    nn.setParameters(genotype, WeightRange, BiasRange)
    fitness = np.zeros(4)

    # Task 1
    print("Simulating IP")
    body = invpend.InvPendulum()
    total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
    i = 0
    k = 0
    theta_traces_IP = []
    for theta in theta_range_IP:
        j = 0
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            f = 0.0
            theta_trace = []
            for t in time_IP:
                #nn.step(
                    #np.concatenate((body.state(), np.zeros(4), np.zeros(3), np.zeros(2)))
                #)  # arrays for inputs for each task
                #nn.step(np.concatenate((body.state(),np.zeros(1))))
                nn.step(body.state())
                k += 1
                #f += body.step(stepsize_IP, np.array([nn.output()[0]]))
                f = body.step(stepsize_IP,nn.output())
                theta_trace.append(np.rad2deg(body.theta))
            j += 1
            theta_traces_IP.append(theta_trace)
        i += 1
    np.save(
        os.path.join(save_dir, "theta_traces_IP_{}.npy".format(run_num)),
        theta_traces_IP,
    )
    del theta_traces_IP

    # Task 2
    print("Simulating CP")
    body = cartpole.Cartpole()
    total_steps = (
        len(theta_range_CP)
        * len(thetadot_range_CP)
        * len(x_range_CP)
        * len(xdot_range_CP)
        * len(time_CP)
    )
    fit_CP = np.zeros((len(theta_range_CP), len(thetadot_range_CP)))
    i = 0
    k = 0
    theta_traces_CP = []
    for theta in theta_range_CP:
        j = 0
        for theta_dot in thetadot_range_CP:
            f_cumulative = 0
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    theta_trace = []
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    f = 0.0
                    for t in time_CP:
                        #nn.step(
                        #    np.concatenate((np.zeros(3), body.state(), np.zeros(3), np.zeros(2)))
                        #)
                        nn.step(body.state())
                        k += 1
                        f_temp, d = body.step(stepsize_CP, nn.output())
                        theta_trace.append(np.rad2deg(body.theta))
                    theta_traces_CP.append(theta_trace)
            j += 1
        i += 1
    np.save(
        os.path.join(save_dir, "theta_traces_CP_{}.npy".format(run_num)),
        theta_traces_CP,
    )
    del theta_traces_CP

    # Task 3
    print("Simulating LW")
    body = leggedwalker.LeggedAgent(0.0, 0.0)
    total_steps = len(theta_range_LW) * len(omega_range_LW) * len(time_LW)
    fit_LW = np.zeros((len(theta_range_LW), len(omega_range_LW)))
    i = 0
    k = 0
    theta_traces_LW = []
    for theta in theta_range_LW:
        j = 0
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            theta_trace = []
            for t in time_LW:
                #nn.step(np.concatenate((np.zeros(3), np.zeros(4), body.state(), np.zeros(2))))
                nn.step(body.state())
                k += 1
                #body.step(stepsize_LW, np.array(nn.output()[2:5]))
                body.step(stepsize_LW, nn.output())
                theta_trace.append(np.rad2deg(body.angle))
            theta_traces_LW.append(theta_trace)
            j += 1
        i += 1
    np.save(
        os.path.join(save_dir, "theta_traces_LW_{}.npy".format(run_num)),
        theta_traces_LW,
    )
    del theta_traces_LW
    
    
    #Task 4
    print("Simulating MC")
    body = mountaincar.MountainCar()
    total_steps = len(position_range_MC) * len(velocity_range_MC) * len(time_MC)
    fit_MC = np.zeros((len(position_range_MC),len(velocity_range_MC)))
    i = 0
    k = 0
    position_traces_MC = []
    for position in position_range_MC:
        j = 0
        for velocity in velocity_range_MC:
            body.position = position
            body.velocity = velocity
            position_trace = []
            for t in time_MC:
                #nn.step(np.concatenate((np.zeros(3),np.zeros(4),np.zeros(3),body.state())))
                #nn.step(np.concatenate((body.state(),np.zeros(2))))
                nn.step(body.state())
                k += 1
                f,d = body.step(stepsize_MC, nn.output())
                position_trace.append(body.position)
            position_traces_MC.append(position_trace)
            j += 1
        i += 1
    np.save(
        os.path.join(save_dir, "position_traces_MC_{}.npy".format(run_num)),
        position_traces_MC,
    )
    del position_traces_MC
    
    
    
    


individual_id = 1
simulate_individual("./Combined/4T_2x10/Data", individual_id)
