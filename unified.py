# Title: unified.py
#
# Description: A script that unifies the different componenets of this project, 
#              creating a quick way to collect, clean, and analyze data
#
# Author: Monem Moheq
#
# Date: Sept 2026

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ccxt
import data_3 as d3
import data_4 as d4
import data_5 as d5
import clean_1 as clean 
import OU_params as par
import OU_process as proc

def standard(exch_1, exch_2, symbol, i, use_MLE = True):
    '''
        i should be which data collection method to use, either 0, or 1, corresponding to data_3, or data_4
    '''

    # collect
    print('Collecting Data')
    datas = [d3, d4]
    trades_1 = datas[i].gather_trades(exch_1, symbol, show_loop = False)
    trades_2 = datas[i].gather_trades(exch_2, symbol, show_loop = False)

    # print(trades_1)
    # print(trades_2)

    print('Cleaning Data')
    # clean
    prices_1 = clean.clean_trades(trades_1)
    prices_2 = clean.clean_trades(trades_2)
    spread = clean.make_spread_frame(prices_1, prices_2, irregular = False)

    # print(prices_1)
    # print(prices_2)
    # print(spread)
    # exit()

    print('Determining Parameters')
    if not use_MLE:
        # params via regression
        theta, mu, omega, sigma = par.calibrate_regression(spread)
        mu = 0
        omega_2 = omega * omega
        sigma_2 = sigma * sigma
    else:
        # params via MLE
        mle = par.MLE(spread)
        params = mle.calibrate_MLE()
        mu = 0
        theta = params[0]
        sigma_2 = params[1]
        omega_2 = params[2]

    print('Simulating Trajectories')
    # simulate
    dt = 0.1
    process = proc.OU(mu, theta, sigma_2, omega_2,dt)

    max_iter = 1000
    num_trajectories = 30
    simulation = process.generate_trajectories(max_iter, num_trajectories)

    print('Plotting Results')
    # plot
    total_time =  max_iter * dt
    x_axis = np.linspace(0, total_time, max_iter)
    for i in range(num_trajectories):
        plt.plot(x_axis, simulation[i], color = 'black', lw=0.5 )

    # print(x_axis)
    # print(spread['spread'].to_numpy()[4,0:max_iter])
    # exit()
    plt.plot(x_axis, spread['spread'].to_numpy()[0:max_iter], color = 'red')
    plt.title("Ornstein-Uhlenbeck Process Simulation")
    plt.xlabel("Time (100ms)")
    plt.ylabel("X(t)")
    plt.savefig('simulation.png')

    return 0

def main():
    exch_1 = ccxt.binance({'enableRateLimit': True})
    exch_2 = ccxt.kraken({'enableRateLimit': True})
    symbol = 'BTC/USDT'

    standard(exch_1, exch_2, symbol, 0, use_MLE = False)

    return 0 

if __name__  == '__main__':
    main()