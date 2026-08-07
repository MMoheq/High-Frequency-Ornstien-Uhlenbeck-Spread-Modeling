# Title: OU_params.py
#
# Description: This script takes the spread data from clean_1.py, and accomplishes two tasks with it
#                   1. Use regression to find coefficients for the Ornstien-Uhlenback process
#                   2. Use Ornstien-Uhlenback process to create a function that calculates expected price at time T
#                      Given the price at the current time. 
#                   
#              This file expects the spread csv as a command line input. 
#
#              Regression math: https://fxpaul.wordpress.com/2011/05/26/calibration-of-ornstein-uhlenbeck-process/
#               Regression: y = a + bx + e 
#               S_t - S_{t-1} = theta * mu * delta t - theta * S_{t-1} delta t + sigma * sqrt(delta t) * W_{t-1}
#               ie we want to regress S_{t-1} against S_t - S_{t-1}
#               
#              Note that the MLE approach assumes a mean of 0 for the spread
#       
# Author: Monem Moheq
#
# Date: Aug 2026

import numpy as np 
import pandas as pd
import sys
import nlopt
from scipy import stats

class MLE():
    '''
        Class for using MLE uses nlopt library
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#objective-function
    '''
    
    def __init__(self, spread):
        self.spread = spread

        # getting t_i - t_{i-1}
        self.spread['time-1'] = self.spread['timestamp'].shift(1)
        self.spread = self.spread.iloc[1:]
        self.spread['change_time'] = self.spread['timestamp'] - self.spread['time-1']

        # getting S_{t-1}
        self.spread['spread-1'] = self.spread['spread'].shift(1)
        self.spread = self.spread.iloc[1:]

        # getting S_t - S_{t-1}
        self.spread['change'] = self.spread['spread'] - self.spread['spread-1']

        self.X = self.spread['spread'].to_numpy()
        self.X_1 = self.spread['spread-1'].to_numpy()
        self.dt = self.spread['change_time'].to_numpy()


    def Expectation(self, X_1, dt, p):
        ''' X should be spread values at time t-1, dt should be change in time, is t_i - t_{i-1}'''
        mu = 0
        theta = p[0]
        sigma_2 = p[1]
        omega_2 = p[2]
        return (((X_1 * sigma_2) + (2 * theta * mu * omega_2)) * (np.exp(- theta * dt)) / (sigma_2 + (2 * theta * omega_2))) + (mu * (1 - np.exp(- theta * dt)))

    def Variance(self, dt, p):
        ''' X should be spread values at time t-1, dt should be change in time, is t_i - t_{i-1}'''
        mu = 0
        theta = p[0]
        sigma_2 = p[1]
        omega_2 = p[2]

        return ((sigma_2 * omega_2 ) * (np.exp(-2 * theta * dt)) / (sigma_2 + (2 * theta * omega_2) )) + ((sigma_2/(2* theta)) * (1 - (np.exp(-2 * theta * dt)))) + omega_2

    def obj_function(self, p, grad):
        '''p is the vector containing the paremeters of the optimization problem'''
        Var = self.Variance(self.dt, p)
        Exp = self.Expectation(self.X_1, self.dt, p)
        f = -(1/2)*(np.sum(np.log(2 * np.pi * Var ))  +  np.sum(np.square(self.X - Exp)/Var))
        return float(f)
    
    def calibrate_MLE(self):
        '''
            This function calibrates the paremeters using Maximum-Likelyhood Estimates. 
            In particular it uses nlopt to optimize the parameters for the log-likelyhood formula
            of a irregular Ornstien-Uhlenbeck process. 
            https://arxiv.org/pdf/1811.09312
        '''

        # initialize optimization settings
        lower_bounds = [0.00000001, 0.00000001, 0.00000001]
        tolerance = 0.0001
        max_iter = 10000

        # create optimization object
        opt = nlopt.opt(nlopt.LN_SBPLX, 3)
        opt.set_max_objective(self.obj_function)
        opt.set_lower_bounds(lower_bounds)
        opt.set_ftol_abs(tolerance)
        opt.set_maxeval(max_iter)

        init_guess = np.array([0.01, 2 * 0.01 * np.var(self.X),0.001])
        x_opt = opt.optimize(init_guess)

        #print(x_opt)
        
        return x_opt

def calibrate_regression(spread):
    '''
        Calculates parameters for OU process using regression
        Returns parameters
        Note that this function changes the spread dataframe, mostly adding new columns,
    '''
    # time interval in seconds 
    delta_t = 0.1

    # getting S_{t-1}
    spread['spread-1'] = spread['spread'].shift(1)
    spread = spread.iloc[1:]

    # getting S_t - S_{t-1}
    spread['change'] = spread['spread'] - spread['spread-1']

    x = spread['spread-1'].to_numpy()
    y = spread['change'].to_numpy()
    reg = stats.linregress(x, y)

    # get regression coefficients
    a = reg.intercept
    b = reg.slope

    # calculate residuals
    resids = y - (a + b * x)

    # calculate Ornstien-Uhlenbeck parameters
    theta = -b/delta_t
    mu = -a/b
    omega = resids.std()
    sigma = omega/np.sqrt(delta_t)

    #print(a,b, resids)

    return theta, mu, omega, sigma

def main(spread_file):

    spread = pd.read_csv(spread_file)
    calibrate_regression(spread)

    print(spread.iloc[0]['timestamp'])
    mle = MLE(spread)
    mle.calibrate_MLE()

    return 0

if __name__ == '__main__':
    main(sys.argv[1])
