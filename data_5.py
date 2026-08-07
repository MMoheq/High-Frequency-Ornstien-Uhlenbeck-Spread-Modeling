# Title: data_6
# 
# Description: This script gathers data as it comes in using live ticker streams, it can save the data gathered in to a csv
#              Note that if you want an hours worth of data, then you must run this for an hour.
#              This is ideal for exchanges that don't have 'since' or 'fromid' parameters in fetch_trades
# 
# Author: Monem Moheq
#
# Date: Aug 2026

import ccxt
import numpy as np 
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import ccxt.pro as ccxtp
import asyncio

async def watch_exch(exch, symbol, stop_event):
    '''
        saves data to a pandas data frame as it comes in from the websocket
    '''
    prices = pd.DataFrame(columns = ['timestamp','datetime', 'price'])
    markets = await exch.load_markets()

    record = []

    while not stop_event.is_set():
        try:
            ticker = await exch.watch_ticker(symbol)
            print(exch.id, 'timestamp: ', ticker['timestamp'], 'price: ', ticker['last'], 'date: ', ticker['datetime'])
            record.append({'timestamp': ticker['timestamp'], 
                                    'datetime': ticker['datetime'],
                                    'price': ticker['last']})
            
        except asyncio.TimeoutError:
            continue

    return pd.DataFrame(record)

async def main():

    exch1 = ccxtp.binance()
    exch2 = ccxtp.okx()
    symbol = 'BTC/USDT'

    stop_event = asyncio.Event()

    try:
        task_1 = asyncio.create_task(watch_exch(exch1, symbol, stop_event))
        task_2 = asyncio.create_task(watch_exch(exch2, symbol, stop_event))

        await asyncio.to_thread(input)

        print('Saving to dataframe')

        stop_event.set()
        prices_1, prices_2 = await asyncio.gather(task_1, task_2)

        print(prices_1), 
        print(prices_2)
    except asyncio.CancelledError:
        print()
        print("Stream task cancelled.")
    finally:
        await exch1.close()
        await exch2.close()
        print('Done')

    save_trades_1 = input('Save exchange 1 trade history to csv? [Y/n]: ')
    if save_trades_1 == 'Y':
        prices_1.to_csv('output/streamed_prices_1.csv', index = False)
    save_trades_2 = input('Save exchange 2 trade history to csv? [Y/n]: ')
    if save_trades_2 == 'Y':
        prices_2.to_csv('output/streamed_prices_2.csv', index = False)
    return 0 

if __name__ == '__main__':
    asyncio.run(main())
