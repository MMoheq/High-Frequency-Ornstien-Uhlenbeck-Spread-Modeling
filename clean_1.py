# Title: clean_1.py
# 
# Description: This script cleans the data price data generated. We want price data for each 100ms interval for the last hour.
#              The data gathered comes at non-uniform datetimes, and we may or may not have a data point for each 100ms interval.
#              We must create 100ms bins for the data
#                  If there are multiple data points for a particular 100ms interval: average the price.
#                  If there is only one data point for an interval, assign that as the price at that time
#                  If there is no data on an interval, assume it has maintained the last known price. 
#              
#              First aggregate each dataframe into 100ms bins for the 100ms intervals that they have. 
#              Then join this onto another dataframe that has 100ms bins for every moment in the past hour.
#              Then fill in the gaps with the most recent known price
#                   Alternatively, sample from a normal distirbution with a windowed mean and variance of the respective dataframe.
#              
#              This file takes dataframes as command line inputs as csv. 
# 
# Author: Monem Moheq
# 
# Date: August 2026

import numpy as np 
import pandas as pd
import sys
import random
from datetime import datetime

def clean_trades(trades):
    '''
        Expects trades as a pandas dataframe
        puts data into 100ms bins and averages their price

        returns the cleaned dataframe

        Have chosen to remove fields besides price, datetime, and timestamp
        May choose to change this in the future
    '''

    # Turn datetime strings into datetime objects and timestamps
    trades['datetime'] = trades['datetime'].apply(datetime.fromisoformat)
    trades['timestamp'] = trades['datetime'].apply(datetime.timestamp)
    #trades = trades.drop(columns = ['side', 'id', 'type', 'cost', 'amount'])
    trades = trades[['timestamp', 'datetime', 'price']]

    # round to the nearest 100ms, ie 10th of a second
    trades['timestamp'] = trades['timestamp'].round(decimals = 1)

    # after rounding, there may be multiple rows with the same timestamp
    # group and average
    trades = trades.groupby('timestamp').mean().reset_index()

    return trades

def make_spread_frame(prices_1, prices_2, irregular = False, noisy_interpolation = False):
    '''
        This functions creates the spread dataframe that has 
        all timestamps and datetimes strating from the earliest recorded instance in both exchanges data frame
        It will have the price of the coin on both exchanges and the difference (the spread)

        For timestamps that we have no data for, we will use time based linear interpolation by default
        alternatively we can calculate a windowed sample variance, and sample values to add on top of 
        linear interpolation to create more realistic interpolation.

        This function can create the spread frame without filling in for missing timestamps
        This is meant for the MLE estimator which works with irregularly spaced time series data
        
        note that noise_interpolation = True will do nothing if irregular = True
        
        returns the spread_frame, it should be set up for analysis
    '''

    start_time = np.max([prices_1.iloc[0]['timestamp'], prices_2.iloc[0]['timestamp']])
    x = np.max([prices_1['timestamp'].min(), prices_2['timestamp'].min()])

    # I understand this is strange programming
    # this is to be prepared for future adjustments
    if not start_time == x:
        print("Potential Conflict, min time is not the same as first time in dataframe")
        print("Function continues with min time")
        start_time = x

    # print(prices_1)
    # print(prices_2)

    # filter out times before the common start time
    prices_1 = prices_1[prices_1['timestamp'] >= start_time].reset_index(drop = True)
    prices_2 = prices_2[prices_2['timestamp'] >= start_time].reset_index(drop = True)

    # print(prices_1)
    # print(prices_2)

    end_time = np.min([prices_1.iloc[-1]['timestamp'], prices_2.iloc[-1]['timestamp']])
    y = np.min([prices_1['timestamp'].max(), prices_2['timestamp'].max()])

    if not end_time == y:
        print("Potential Conflict, max time is not the same as first time in dataframe")
        print("Function continues with min time")
        start_time = x

    # filter out times past the common end time
    prices_1 = prices_1[prices_1['timestamp'] <= end_time].reset_index(drop = True)
    prices_2 = prices_2[prices_2['timestamp'] <= end_time].reset_index(drop = True)

    # print(prices_1)
    # print(prices_2)

    # convert start and end into datetimes
    start_date = datetime.fromtimestamp(start_time)
    end_date = datetime.fromtimestamp(end_time)

    # create the spread frame by making a data frame indexed by times starting from the 
    # start and end dates calculated above, with intervals of 100ms
    times_index = pd.date_range(start_date, end_date, freq = '100ms')
    spread = pd.DataFrame(index = times_index).reset_index(names = 'datetime')
    spread['timestamp'] = spread['datetime'].apply(datetime.timestamp)

    # rename price columns so join doesn't raise issues
    prices_1 = prices_1.set_index('timestamp').rename(columns = {'price' : 'price_1'})
    prices_2 = prices_2.set_index('timestamp').rename(columns = {'price' : 'price_2'})

    # print(prices_1)
    # print(prices_2)

    # set index to timestamp to join on this axis. 
    # this is used instead of datetime since datatimes haven't been rounded to 100ms bins
    spread = spread.set_index('timestamp')

    # join prices to spread dataframe
    spread = spread.join([prices_1['price_1'], prices_2['price_2']])
    
    # swap the index to datetime so that interpolation works
    spread = spread.reset_index().set_index('datetime')

    if irregular:
        spread = spread.dropna()
    else:

        spread_irreg = spread.dropna()

        # using method = 'time' is somewhat trivial since our indices are spaced apart linearly
        spread['price_1'] = spread['price_1'].interpolate(method = 'time')
        spread['price_2'] = spread['price_2'].interpolate(method = 'time')

        if noisy_interpolation:

            # arbitrary window size, future work to deterimine optimal window size
            window = 5
            
            # windowed means
            spread_irreg['mean_1'] = spread_irreg['price_1'].rolling(window = window).mean()
            spread_irreg['mean_2'] = spread_irreg['price_2'].rolling(window = window).mean()

            # remove mean as the 'signal'
            spread_irreg['noise_1'] = spread_irreg['price_1'] - spread_irreg['mean_1']
            spread_irreg['noise_2'] = spread_irreg['price_2'] - spread_irreg['mean_2']

            # calculate variance of the noise
            spread_irreg['variance_1'] = spread_irreg['noise_1'].rolling(window = window).var()
            spread_irreg['variance_2'] = spread_irreg['noise_2'].rolling(window = window).var()

            # join to main spread frame
            spread = spread.join([spread_irreg['variance_1'], spread_irreg['variance_2']])

            # interpolate variances
            spread['sigma_1'] = np.sqrt(spread['variance_1'].interpolate(method = 'time'))
            spread['sigma_2'] = np.sqrt(spread['variance_2'].interpolate(method = 'time'))

            # generate random noise based on the windowed noise variance
            spread['noise_1'] = spread.apply(lambda x: random.gauss(0, x['sigma_1']), axis = 1)
            spread['noise_2'] = spread.apply(lambda x: random.gauss(0, x['sigma_2']), axis = 1)

            spread['is_Nan_1'] = pd.isnull(spread['price_1']).astype(int)
            spread['is_Nan_2'] = pd.isnull(spread['price_2']).astype(int)

            # add noise to the linear interpolation
            # we multiply by the is_Nan_* column so that we're only adding nosie to the interpolated values
            spread['price_1'] = spread['price_1'] + (spread['is_Nan_1'] * spread['noise_1'])
            spread['price_1'] = spread['price_1'] + (spread['is_Nan_2'] * spread['noise_2'])

        # interpolation fails for the starting indices, so remaining NaN's with the price recorded on the other exchange
        spread['price_1'] = spread['price_1'].fillna(spread['price_2'])
        spread['price_2'] = spread['price_2'].fillna(spread['price_1'])

    # get rid of auxilary columns
    spread = spread[['timestamp', 'price_1', 'price_2']]
    # this setup assumes we consider prices on exchange 1 being greater than those on exchange 2 as positive
    # that is to say we have coins on exchange 1 that we can sell and get cheaper coins on exchange 2
    spread['spread'] = spread['price_1'] - spread['price_2']
    # print(spread)

    return spread


def main(exch_1_file, exch_2_file):
    
    trades_1 = pd.read_csv(exch_1_file)
    trades_2 = pd.read_csv(exch_2_file)

    prices_1 = clean_trades(trades_1)
    prices_2 = clean_trades(trades_2)

    spread = make_spread_frame(prices_1, prices_2, irregular = True)

    print(spread)
    
    save_trades_1 = input('Save exchange 1 trade history to csv? [Y/n]: ')
    if save_trades_1 == 'Y':
        spread.to_csv('output/spread_live_data_irreg.csv')
   
    return 0

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
