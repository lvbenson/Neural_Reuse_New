#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:20:21 2020

@author: lvbenson
"""

import os
import numpy as np
import infotheory
import ffann  # Controller
import invpend  # Task 1
import cartpole  # Task 2
import leggedwalker  # Task 3
import matplotlib.pyplot as plt
import sys

# ANN Params
nI = 3 + 4 + 3
nH1 = 10
nH2 = 10
nO = 1 + 1 + 3  # output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

# Task Params
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 10  # 50
stepsize_CP = 0.05
duration_LW = 220  # 220.0
stepsize_LW = 0.1
time_IP = np.arange(0.0, duration_IP, stepsize_IP)
time_CP = np.arange(0.0, duration_CP, stepsize_CP)
time_LW = np.arange(0.0, duration_LW, stepsize_LW)

MaxFit = 0.627  # Leggedwalker

# Fitness initialization ranges
# Inverted Pendulum
trials_theta_IP = 5
trials_thetadot_IP = 2
total_trials_IP = trials_theta_IP * trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0, 1.0, num=trials_thetadot_IP)

# Cartpole
trials_theta_CP = 7
trials_thetadot_CP = 2
trials_x_CP = 2
trials_xdot_CP = 2
total_trials_CP = trials_theta_CP * trials_thetadot_CP * trials_x_CP * trials_xdot_CP
theta_range_CP = np.linspace(-np.deg2rad(6), np.deg2rad(6), num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

# Legged walker
trials_theta = 5
theta_range_LW = np.linspace(-np.pi / 6, np.pi / 6, num=trials_theta)
trials_omega_LW = 2
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

# Fitness function
def simulate_individual(save_dir, run_num):
    # Common setup
    genotype = np.load("./2x10/best_individual_" + "53" + ".npy")
    nn = ffann.ANN(nI, nH1, nH2, nO)
    nn.setParameters(genotype, WeightRange, BiasRange)
    fitness = np.zeros(3)

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
                nn.step(
                    np.concatenate((body.state(), np.zeros(4), np.zeros(3)))
                )  # arrays for inputs for each task
                k += 1
                f += body.step(stepsize_IP, np.array([nn.output()[0]]))
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
                        nn.step(
                            np.concatenate((np.zeros(3), body.state(), np.zeros(3)))
                        )
                        k += 1
                        f_temp, d = body.step(stepsize_CP, np.array([nn.output()[1]]))
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
                nn.step(np.concatenate((np.zeros(3), np.zeros(4), body.state())))
                k += 1
                body.step(stepsize_LW, np.array(nn.output()[2:5]))
                theta_trace.append(np.rad2deg(body.angle))
            theta_traces_LW.append(theta_trace)
            j += 1
        i += 1
    np.save(
        os.path.join(save_dir, "theta_traces_LW_{}.npy".format(run_num)),
        theta_traces_LW,
    )
    del theta_traces_LW


individual_id = 7
simulate_individual("./2x10", individual_id)