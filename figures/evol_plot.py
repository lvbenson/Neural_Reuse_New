#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:34:36 2020

@author: lvbenson
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])

reps = 90
gens = len(np.load(dir+"/average_history_0.npy"))
gs=len(np.load(dir+"/best_individual_0.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))


index = 0
count = 0
for i in range(0,86):
    af[index] = np.load(dir+"/average_history_"+str(i)+".npy")
    bf[index] = np.load(dir+"/best_history_"+str(i)+".npy")
    bi[index] = np.load(dir+"/best_individual_"+str(i)+".npy")


    plt.plot(af.T,'y')
    plt.plot(bf.T,'b')
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Evolution")
plt.show()