# Title: OU_process.py
#
# Description: This module impliments Ornstien-Uhlenbeck process using parameters calculated in 
#              analysis.py. The primary function is E[x_t | x_0], ie the expected spread at time t 
#              given the current spread. We also implement trajectory simulations
# 
#       
# Author: Monem Moheq
#
# Date: Aug 2026

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import OU_params
import nlopt
import sys
import random


class OU():

    def __init__(self, mu, theta, sigma_2, omega_2, dt):
        
        self.mu = mu
        self.theta = theta
        self.sigma_2 = sigma_2
        self.omega_2 = omega_2
        self.dt = dt

    def generate_trajectories(self, max_iter, num_trajectories):
        '''
            Simulates many trajectories given OU parameters

            returns list of trajectories
        '''
        trajectories = []
        for i in range (num_trajectories):

            X = np.zeros(max_iter)

            for i in range(max_iter - 1):
                e = random.gauss(0, np.sqrt(self.omega_2))
                X[i+1] = X[i] + (self.dt * self.theta * (self.mu - X[i])) + (np.sqrt(self.sigma_2 * self.dt)* e )

            trajectories.append(X)

        return trajectories
    
    def expectation(self, x,t):
        '''
            x: current price, t future time in question
            E[x_t | x_0]
            https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
        '''
        e = np.exp(-self.theta * t)
        return (x * e) + (self.mu(1 - e))


def main(spread_file):

    spread = pd.read_csv(spread_file)

    mle = OU_params.MLE(spread)
    
    # params via regression
    # theta, mu, omega, sigma = OU_params.calibrate_regression(spread)
    # mu = 0
    # omega_2 = omega * omega
    # sigma_2 = sigma * sigma

    # params via MLE
    params = mle.calibrate_MLE()
    mu = 0
    theta = params[0]
    sigma_2 = params[1]
    omega_2 = params[2]

    dt = 0.1
    process = OU(mu, theta, sigma_2, omega_2,dt)

    max_iter = 1000
    num_trajectories = 30
    simulation = process.generate_trajectories(max_iter, num_trajectories)

    total_time =  max_iter * dt
    x_axis = np.linspace(0, total_time, max_iter)
    for i in range(num_trajectories):
        plt.plot(x_axis, simulation[i], color = 'black', lw=0.5 )
    
    spread['diff'] = spread['timestamp'] - spread.iloc[0]['timestamp']
    spread = spread[spread['diff'] <= total_time]
    x_axis_irreg = spread['diff'].to_numpy()

    #plt.plot(x_axis_irreg, spread['spread'].to_numpy()[0:max_iter], color = 'red')
    plt.title("Ornstein-Uhlenbeck Process Simulation")
    plt.xlabel("Time (100ms)")
    plt.ylabel("X(t)")
    plt.show()

    return

if __name__ == '__main__':
    main(sys.argv[1])