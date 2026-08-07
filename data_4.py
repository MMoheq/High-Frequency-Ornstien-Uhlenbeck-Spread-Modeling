# Title: data_4
# 
# Description: This attempts to gather historical data from the past hour using fetch_trades or fetch_tickers
#              we want data that comes in intervals of 100ms 
#              This version uses trade id's rather than timestamps to paginate
#              Certain exchanges such as Binance or bybit prefer this.
#
# Author: Monem Moheq
#
# Date: August 2026

import ccxt
import numpy as np 
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import ccxt.pro as ccxtp
import asyncio

def trades(exch_1, exch_2, symbol):
    # for reference of columns
    # ['id', 'order', 'info', 'timestamp', 'datetime', 'symbol', 'type',
    #    'side', 'takerOrMaker', 'price', 'amount', 'cost', 'fee', 'fees']
    now = datetime.now(timezone.utc)
    hour_ago = int((now - timedelta(hours = 1)).timestamp() * 1000)
    print(hour_ago)
    # note that datetime are string objects
    exch_1_trades = pd.DataFrame(exch_1.fetch_trades(symbol, since = hour_ago, limit = 1000))[['id', 'datetime','side','type','cost', 'price', 'amount']]
    exch_2_trades = pd.DataFrame(exch_2.fetch_trades(symbol, since = hour_ago, limit = 1000))[['id', 'datetime','side','type','cost', 'price', 'amount']]

    
    print(exch_1_trades)
    print(exch_2_trades)
    print(exch_1_trades.at[len(exch_1_trades) - 1, 'datetime'])
    print(datetime.fromisoformat(exch_1_trades.at[len(exch_1_trades) - 1, 'datetime']))
    print(int((datetime.fromisoformat(exch_1_trades.at[len(exch_1_trades) - 1, 'datetime'])).timestamp() * 1000))
    # t = int((datetime.fromisoformat(exch_1_trades.at[len(exch_1_trades) - 1, 'datetime'])).timestamp() * 1000)
    # exch_1_trades = pd.DataFrame(exch_1.fetch_trades(symbol, since = t, limit = 1000))[['id', 'datetime','side','type','cost', 'price', 'amount']]
    # print(exch_1_trades)
    return

def gather_trades_1(exch, symbol, show_loop = False):
    '''
        This Gathers trade history for the past hour
        This function assumes the 'since' kwarg is applicable to the exchange
        Some exchanges such as okx can run into issues using since

        uses id's rather than time stamps

        show_loop is primarily for debugging
    '''

    # get timestamp of one hour ago.
    # use .isoformat() to get string if needed
    now = datetime.now(timezone.utc)
    hour_ago = int((now - timedelta(hours = 1)).timestamp() * 1000) # expected format for since

    # note that trade id's are generally unique, so they are a good way to tell if two trades are the same one
    
    exch_trades = pd.DataFrame(exch.fetch_trades(symbol,  since = hour_ago, limit = 20))[['id', 'datetime','side','type','cost', 'price', 'amount']]
    exch_trade_new = pd.DataFrame(exch.fetch_trades(symbol, limit = 20))[['id', 'datetime','side','type','cost', 'price', 'amount']]

    id_tracker = int(exch_trades.iloc[-1]['id'])
    id_new = int(exch_trade_new.iloc[-1]['id'])

    i = 0
    # looping through first exchange
    while  id_tracker < id_new:
        
        if show_loop:
            print(exch.id, ' ', symbol, ' loop number:', i , 'id: ', id_tracker)
            trades_1 = pd.DataFrame(exch.fetch_trades(symbol, params = {'fromId' : id_tracker}, limit = 1000))
            
        if not trades_1.empty:
            trades_1 = trades_1[['id', 'datetime','side','type','cost', 'price', 'amount']]

            # add it to trade history, ingore index so it doesn't repeat indices
            exch_trades = pd.concat([exch_trades, trades_1], ignore_index = True)
        # else:
        #     id_tracker = id_tracker + 1
        #     continue

        # get id of last trade
        id_tracker_next = int(exch_trades.iloc[-1]['id'])
        id_tracker = id_tracker_next

        # if time_tracker_next == time_tracker:
        #     time_tracker = time_tracker_next + 1000
        # else:
        #     if time_tracker > time_tracker_next:
        #         time_tracker = time_tracker + 1000
        #     else:
        #         time_tracker = time_tracker_next

        time.sleep (exch.rateLimit / 1000)
        i = i + 1

    return exch_trades.drop_duplicates()
    
def main():
    exch_1 = ccxt.binance({'enableRateLimit': True})
    exch_2 = ccxt.kraken({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    exch_1_trades = gather_trades_1(exch_1, symbol, show_loop = True)
    exch_2_trades = gather_trades_1(exch_2, symbol, show_loop = True)

    print(exch_1_trades)
    print(exch_2_trades)

    save_trades_1 = input('Save exchange 1 trade history to csv? [Y/n]: ')
    if save_trades_1 == 'Y':
        exch_1_trades.to_csv('output/exch_1_trades.csv', index = False)
    save_trades_2 = input('Save exchange 2 trade history to csv? [Y/n]: ')
    if save_trades_2 == 'Y':
        exch_2_trades.to_csv('output/exch_2_trades.csv', index = False)

    # exch_1 = ccxt.binance({'enableRateLimit': True})
    # symbols = ['BTC/USDT', 'BTC/USD']
    # symbol_1_trades = gather_trades_1(exch_1, symbols[0], show_loop = True)
    # symbol_2_trades = gather_trades_1(exch_1, symbols[1], show_loop = True)

    # print(symbol_1_trades)
    # print(symbol_2_trades)
    return 0

if __name__ == '__main__':
    main()

# binance and okx seem to be the best for gathering high frequency data. 