# todo: remove to_datetime, already in imports

import numpy as np
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
import matplotlib.pyplot as plt
import datetime
#from indicators import moving_average, exp_moving_average, vwap
#from Signals import *
from AlgoCrac import moving_average, exp_moving_average, vwap
from AlgoCrac.Signals.Signals import *
from AlgoCrac.Signals.utils import *
from tqdm import tqdm
import time



class Strategy:
    """
    Strategy class is an abstract class that defines the basic structure of a strategy.
    
    """
    def __init__(self,
                entry_signal,
                exit_signal,
                side = 'long', # long or short
                timeframe = '1D', # 'D' for day,  1H for 1 hour, 15m for 5 minute,  1m for 1 minute etc
                allocation = 10000, # total amount of money available for this strategy
                trade_allocation = 1000, # amount of money allocated to each trade
                intraday = None, # if timeframe 1m-1H, then intraday = True, else intraday = False, unless defined otherwise 
                manual_exit_time = None, # relevant for intra-day strategies, if None then exit at end of day, 5 minutes before the market closes (15:55)
                negative_allocation = False, # if True, then allow negative allocation, else, if allocation < trade_allocation, then no trades are opened
                min_commision = 1.5, # minimum commision per trade
                comminision_rate = 0.00, # commision rate per share
                 ):

        self.entry_signal = entry_signal
        self.exit_signal = exit_signal
        self.side = side
        self.open_trades = pd.DataFrame(columns = ['reward'])
        self.closed_trades = pd.DataFrame(columns = ['reward'])
        self.status = pd.DataFrame()
        self.trade_id = 0
        self.allocation = allocation
        self.trade_allocation = trade_allocation
        self.initial_allocation = allocation
        self.negative_allocation = negative_allocation
        self.min_commision = min_commision
        self.comminision_rate = comminision_rate
        self.manual_exit_time = manual_exit_time


        self.timeframe = timeframe
        # if timeframe is 1m-1H, then intraday = True, else intraday = False, unless defined otherwise
        if intraday is None:
            if timeframe in ['1m','5m','15m']:
                self.intraday = True
            else:
                self.intraday = False
        self.delta_dict = {'1m':1,'5m':5,'15m':15,'30m':30,'1H':60,'1D':1440}

        last_intraday_time = datetime.time(15,60 - self.delta_dict[self.timeframe])
        if self.intraday==True and ((manual_exit_time is None) or (manual_exit_time >= last_intraday_time)):
            self.manual_exit_time = last_intraday_time

        # verify that the period of the signals is not too large for intraday strategy
        self.verify_day()

    def set_signal(self, entry_signal, entry = True):
        if entry == True:
            self.entry_signal = entry_signal
        else:
            self.exit_signal = entry_signal
        self.verify_day()



    def insert_signals(self, signals, entry = True):
        signals = copy.deepcopy(signals)
        assert type(self.entry_signal) == Confluence
        assert type(self.exit_signal) == Confluence

        if type(signals) != list:
            signals = [signals]

        if entry == True:
            new_signal_list = self.entry_signal.signals_list
            # check if signals with similar profile is already in the list, if so do not insert
            for s in signals:
                exists = False
                for s1 in self.entry_signal.signals_list:
                    if s.profile == s1.profile:
                        exists = True
                if exists == False:
                    new_signal_list.append(s)
            self.entry_signal = Confluence(new_signal_list,
                                           max_signals = self.entry_signal.max_signals,
                                            invert = self.entry_signal.invert)
        else:
            new_signal_list = self.exit_signal.signals_list
            for s in signals:
                exists = False
                for s1 in self.exit_signal.signals_list:
                    if s.profile == s1.profile:
                        exists = True
                if exists == False:
                    new_signal_list.append(s)
            self.exit_signal = Confluence(new_signal_list,
                                           max_signals = self.exit_signal.max_signals,
                                           invert = self.exit_signal.invert)
        self.verify_day()


    def remove_signals(self, signals, entry = True):
        assert type(self.entry_signal) == Confluence
        assert type(self.exit_signal) == Confluence

        if type(signals) != list:
            signals = [signals]

        if entry == True:
            new_signal_list = self.entry_signal.signals_list
            for s in signals:
                new_signal_list = [s1 for s1 in new_signal_list if s1.profile != s.profile]
            self.entry_signal = Confluence(new_signal_list,
                                           max_signals = self.entry_signal.max_signals,
                                           invert = self.entry_signal.invert)
        else:
            new_signal_list = self.exit_signal.signals_list
            for s in signals:
                new_signal_list = [s1 for s1 in new_signal_list if s1.profile != s.profile]
            self.exit_signal = Confluence(new_signal_list,
                                           max_signals = self.exit_signal.max_signals,
                                           invert = self.exit_signal.invert)
        self.verify_day()

        
    def verify_day(self):   
        # verify that the period of the signals is not too large for intraday strategy
        if self.intraday == True:
            all_signals_list = []
            for s in [self.entry_signal, self.exit_signal]:
                # check if the signal is of type Confluence
                if type(s)  == Confluence:
                    all_signals_list += s.signals_list
                else:
                    all_signals_list.append(s)
            for s in all_signals_list:
                if hasattr(s,'period'):
                    # number of minutes between 9:30 and 16:00
                    intraday_minutes = 390
                    if s.period is not None and s.period >= intraday_minutes/self.delta_dict[self.timeframe]:
                        # raise warning
                        print("Error: period of the signal is too large for intraday strategy, strategy is not initialized")
                        return None              

    def reset(self):
        self.open_trades = pd.DataFrame(columns = ['reward'])
        self.closed_trades = pd.DataFrame(columns = ['reward'])
        self.status = pd.DataFrame()
        self.trade_id = 0
        self.allocation = self.initial_allocation


    def update_open_trades(self, data, last_datetime_stamp ,cur_exit_signals = None):

        datetime_stamp = data.index[-1]
        date_stamp = datetime_stamp.date()
        time_stamp = datetime_stamp.time()

       # if self.last_datetime_stamp is not None:
        end_activity = False
        if  (datetime_stamp + datetime.timedelta(minutes=self.delta_dict[self.timeframe])) > last_datetime_stamp:
            end_activity = True
        if self.intraday == True:
            if time_stamp  >= self.manual_exit_time:
                end_activity = True
            data = data.loc[data.index.date == datetime_stamp.date()]

        if len(self.open_trades) > 0:
            symbols_open = self.open_trades['symbol'].unique()  
            open_trade_data = data.loc[:, data.columns.get_level_values(1).map(lambda x: x in symbols_open)]

            # close all trades
            if end_activity:
                is_exit_symbol = symbols_open
            # continue activity
            else:    
                if cur_exit_signals is None:
                    # get last signal
                    signals_exit = self.get_exit_signals(open_trade_data).iloc[0,:]
                else:
                    signals_exit = cur_exit_signals.loc[:,symbols_open].iloc[0,:]
                is_exit_symbol = signals_exit[signals_exit==1].index
            closed_trades_lists = [self.closed_trades]
            for trade_id in self.open_trades['id'].values:
                symbol = self.open_trades[self.open_trades['id']==trade_id]['symbol'].values[0]

                # close trades
                if symbol in is_exit_symbol:
                    trade_info = self.open_trades[self.open_trades['id']==trade_id].copy()
                    trade_info['exit time']= time_stamp
                    trade_info['exit date'] = date_stamp
                    # use open price because the last close price gave the exit signal
                    trade_info['exit price'] = data.iloc[:, data.columns.get_level_values(1) == symbol]['Open'].values[-1]
                    trade_info['id'] = trade_id
                    if self.side == 'long':
                        trade_info['reward'] = (trade_info['exit price'] - trade_info['entry price'])*trade_info['position']
                    else:
                        trade_info['reward'] = (trade_info['entry price'] - trade_info['exit price'])*trade_info['position']

                    net_reward = trade_info['reward'] - trade_info['commission']
                    self.allocation += net_reward.values[0] + (trade_info['entry price']*trade_info['position']).values[0]
                    trade_info.drop(labels = ['market value','last'],axis = 1, inplace = True)
                    self.open_trades.drop(self.open_trades[self.open_trades['id']==trade_id].index, inplace = True)
                    
                    # add trade_info to closed_trades
                    closed_trades_lists.append(trade_info)

                # update trades
                else:
                    last = data.iloc[:, data.columns.get_level_values(1)==symbol]['Close'].values[-1]
                    position = self.open_trades[self.open_trades['id']==trade_id]['position']
                    entry_price = self.open_trades[self.open_trades['id']==trade_id]['entry price']

                    self.open_trades.loc[self.open_trades['id']==trade_id,'last'] = last
                    self.open_trades.loc[self.open_trades['id']==trade_id,'market value'] = last*position

                    max = self.open_trades[self.open_trades['id']==trade_id]['max'].values[0]
                    min = self.open_trades[self.open_trades['id']==trade_id]['min'].values[0]

                    self.open_trades.loc[self.open_trades['id']==trade_id,'max'] = np.maximum(max, last)
                    self.open_trades.loc[self.open_trades['id']==trade_id,'min'] = np.minimum(min, last)
                    if self.side =='long':
                        reward = (last - entry_price)*position 
                    else:
                        reward = (entry_price - last)*position
                    self.open_trades.loc[self.open_trades['id']==trade_id,'reward'] = reward
                    
            # add trade_info to closed_trades
            self.closed_trades = pd.concat(closed_trades_lists,ignore_index = True)


    def open_new_trades(self, data, last_datetime_stamp, cur_entry_signals = None):

        datetime_stamp = data.index[-1]
        date_stamp = datetime_stamp.date()
        time_stamp = datetime_stamp.time()

        end_backtest = False
        if last_datetime_stamp is not None:
            if  (datetime_stamp + datetime.timedelta(minutes=self.delta_dict[self.timeframe])) > last_datetime_stamp:
                end_backtest = True
            if self.intraday == True:
                #closest_exit = (datetime.datetime.combine(datetime.date(1,1,1), time_stamp) + datetime.timedelta(minutes=self.delta_dict[self.timeframe])).time()
                if time_stamp  >= self.manual_exit_time:
                    end_backtest = True
                #if last_datetime_stamp.date() != datetime_stamp.date():
                    # get all data rows from the last day
                data = data.loc[data.index.date == datetime_stamp.date()]
            

        # check for new trades
        if not end_backtest:
            if cur_entry_signals is None:
                signals_entry = self.get_entry_signals(data)
            else:
                signals_entry = cur_entry_signals

            # filter out symbols that have no entry signals
            symbols_entry = signals_entry[signals_entry==1].dropna(axis=1).columns

            open_trade_lists = [self.open_trades]

            for symbol in symbols_entry:
                self.trade_id += 1
                trade_id = self.trade_id
                new_trade = pd.DataFrame(columns = ['id','symbol','entry time',
                                                    'entry date','entry price',
                                                    'position','market value',
                                                    'last','max','min',
                                                    'stop loss','reward'])
                # use open price because the last close price gave the exit signal
                new_trade['entry price'] = data.iloc[:, data.columns.get_level_values(1)==symbol]['Open'].values[-1]
                new_trade['position'] = int(self.trade_allocation/new_trade['entry price'])
                if new_trade['position'].values[0]==0:
                #  print("Trade allocation not sufficient for this trade, entry price: ", new_trade['entry price'].values[0], " trade allocation: ", self.trade_allocation)
                    continue
                if self.negative_allocation == False and self.allocation < self.trade_allocation:
                #  print("Not enough allocation for this trade, trade allocation: ", self.trade_allocation, " allocation: ", self.allocation)
                    continue
                new_trade['id'] = [trade_id]
                new_trade['symbol'] = [symbol]
                new_trade['entry time'] = [time_stamp]
                new_trade['entry date'] = [date_stamp]
                new_trade['market value'] = new_trade['entry price']*new_trade['position']
                new_trade['last'] = new_trade['entry price']
                new_trade['max'] = new_trade['entry price']
                new_trade['min'] = new_trade['entry price']
                new_trade['reward'] = 0
                new_trade['commission'] = np.max([self.min_commision, self.comminision_rate*new_trade['position'].values[0]])
                self.allocation -= new_trade['market value'].values[0]
                open_trade_lists.append(new_trade)
                #self.open_trades = pd.concat([self.open_trades,new_trade], ignore_index = True)
            self.open_trades = pd.concat(open_trade_lists, ignore_index = True)



    def fast_update(self,
                data,
                cur_exit_signals = None, # if None, then use self.exit_signal
                cur_entry_signals = None, # if None, then use self.entry_signal
                ):
        

        time5 = time.time()
        # check for new trades
        if cur_entry_signals is None:
            signals_entry = self.get_entry_signals(data, last_signal = False)
        if cur_exit_signals is None:
            signals_exit = self.get_exit_signals(data, last_signal = False)

        # filter out symbols that have no entry signals
        symbol_list = signals_entry.loc[:,(signals_entry != 0).any(axis=0)].columns

       # symbol_list = data.columns.get_level_values(1).unique()
        #time6 = time.time()
       # print("get signals time: ", time6-time5)


        # create time and date lists
        time_list = []
        date_list = []
        for d in data.index:
            time_list = np.append(time_list,datetime.time(d.hour, d.minute))
            date_list = np.append(date_list,datetime.date(d.year, d.month, d.day))
        
        manual_exit  = []
        if self.manual_exit_time is not None:
            if self.intraday:
                # get rows from data where time is equal to manual_exit_time
                manual_exit_time_inds = data[data.index.time == self.manual_exit_time].index
                # get the i location of manual_exit_time_inds in data
                manual_exit = data.index.get_indexer(manual_exit_time_inds)

        if len(manual_exit) == 0:
            # the last exit is at the begining of the last day, therefore -2
            manual_exit = [len(data)-2]



        symbol_closed_trades_list = [] 

        for symbol_i in range(len(symbol_list)):
            symbol = symbol_list[symbol_i]
            symbol_data = data.loc[:, data.columns.get_level_values(1)==symbol]

          #  symbol_inds_entry = np.tile(np.arange(len(signals_entry[0])), (len(signals_entry),1))
            symbol_inds_entry = np.arange(len(signals_entry[symbol]))
            symbol_inds_entry = symbol_inds_entry[signals_entry[symbol].values != 0]
            inds_entry = symbol_inds_entry+1 # because we want to enter at the next timestep after the signal

            symbol_inds_exit = np.arange(len(signals_exit[symbol]))
            symbol_inds_exit= symbol_inds_exit[signals_exit[symbol].values != 0]
            symbol_inds_exit = symbol_inds_exit+1 # because we want to exit at the next timestep after the signal

            inds_exit = np.concatenate([symbol_inds_exit, manual_exit])
            
           # exit_timesteps = symbol_data.iloc[inds_exit]
            entry_timesteps = symbol_data.iloc[inds_entry]

            # remove all buy_timesteps that are after the last sell timestep
            last_sell_i = np.max(inds_exit)
            last_sell_timestep = symbol_data.iloc[last_sell_i].name
            
            entry_timesteps = entry_timesteps[entry_timesteps.index < last_sell_timestep]
            inds_entry = inds_entry[inds_entry < last_sell_i]

            outs = np.tile(inds_exit,(len(inds_entry),1))
            ins = np.tile(inds_entry,(len(inds_exit),1)).T

            c = (outs - ins).astype(float)
            c[c < 0] = np.inf
            exit_inds_per_entry = np.argmin(c, axis = 1)
            exit_inds_per_entry = np.take_along_axis(outs, exit_inds_per_entry[:,None], axis=1).T

            # this is 2d matrix, designed for row per symbol, here we iterate per symbol therefore take the first row
            exit_inds_per_entry = exit_inds_per_entry[0]

            symbol_closed_trades = pd.DataFrame()
            entry_symbol_data = symbol_data.iloc[inds_entry]
            exit_symbol_data = symbol_data.iloc[exit_inds_per_entry]

            symbol_closed_trades['entry time'] = time_list[inds_entry]
            symbol_closed_trades['entry date'] = date_list[inds_entry]
            symbol_closed_trades['exit time'] = time_list[exit_inds_per_entry]
            symbol_closed_trades['exit date'] = date_list[exit_inds_per_entry]

            symbol_closed_trades['entry price'] = entry_symbol_data['Open'].values
            symbol_closed_trades['exit price'] = exit_symbol_data['Open'].values

            symbol_closed_trades['symbol'] = symbol

            symbol_closed_trades_list.append(symbol_closed_trades) 

      #  time7 = time.time()
        symbol_closed_trades_list.append(self.closed_trades)
        self.closed_trades = pd.concat(symbol_closed_trades_list,ignore_index = True)

       # print("total update per symbol time: ", time7-time6)
        if self.closed_trades.empty:
            return
        self.closed_trades['position'] = (self.trade_allocation/self.closed_trades['entry price']).astype(int)
        self.closed_trades = self.closed_trades[self.closed_trades['position'] != 0] # remove trades with position = 0
        self.closed_trades['commission'] = np.maximum(self.min_commision, self.comminision_rate*self.closed_trades['position'].values)
        self.closed_trades['market value'] = self.closed_trades['exit price']*self.closed_trades['position']

        if self.side == 'long':
            self.closed_trades['reward'] = (self.closed_trades['exit price'] - self.closed_trades['entry price'])*self.closed_trades['position']
        else:
            self.closed_trades['reward'] = (self.closed_trades['entry price'] - self.closed_trades['exit price'])*self.closed_trades['position']
        self.allocation +=  self.closed_trades['reward'].sum() - self.closed_trades['commission'].sum()
        
        self.closed_trades.sort_values(by = ['entry date','entry time'], inplace = True)
        self.closed_trades.reset_index(drop = True, inplace = True)
        self.closed_trades['id'] = np.arange(len(self.closed_trades))
        time8 = time.time()
       # print("all symbols time: ", time8-time7)
 
    def get_entry_signals(self, data, last_signal = True):

        datetime_stamp = data.index[-1]
        symbols_open = data.columns.get_level_values(1).unique()

        if  len(data) == 1:
            signals = np.reshape([0]*len(symbols_open),(1,-1))
        else:
            if last_signal:
                if (self.intraday == True and datetime_stamp.time() >= self.manual_exit_time) or len(data) == 1:
                    signals = [0]*len(symbols_open)      
                else:
                    # must always send data without the last row, because the last row is the current row
                    signals = self.entry_signal.get_last_signal(data.iloc[:-1], signal_type = 'dis')
                signals = np.reshape(signals,(1,-1))
            else:
                # must always send data without the last row, because the last row is the last row
                signals = self.entry_signal.get_signals(data.iloc[:-1], signal_type = 'dis').T
        es = pd.DataFrame(signals,columns = symbols_open.values)
        return es

    def get_exit_signals(self, data, last_signal = True):
        datetime_stamp = data.index[-1]

        symbols_close = data.columns.get_level_values(1).unique()
        if  len(data) == 1:
            signals = np.reshape([0]*len(symbols_close),(1,-1))
        else:
            if last_signal:
                if self.intraday == True and datetime_stamp.time() >= self.manual_exit_time:
                    signals = [1]*len(symbols_close)
                else:
                    # must always send data without the last row, because the last row is the current row
                    signals = self.exit_signal.get_last_signal(data.iloc[:-1], signal_type = 'dis')
                signals = np.reshape(signals,(1,-1))
            else:
                # must always send data without the last row, because the last row is the last row
                signals = self.exit_signal.get_signals(data.iloc[:-1], signal_type = 'dis').T          
        es = pd.DataFrame(signals,columns = symbols_close)     
        return es



class Backtest:
    """
    Backtest class is used to run a backtest on a given strategy and data

    Args:
        strategy (Strategy): Strategy object
        data (pd.DataFrame): Dataframe of the data to run the backtest on
        commission (float, optional): Commission rate per share. Defaults to 0.01.
        min_commission (float, optional): Minimum commission per trade. Defaults to 1.5.
        slipage (float, optional): Slippage per share. Defaults to 0.0.
        start_offset (int, optional): Number of rows to skip before starting the backtest. Defaults to 5.
        include_commission (bool, optional): Whether to include commission in the reward calculation. Defaults to True.
        fast (bool, optional): Whether to run the backtest in fast mode. Defaults to False. Note: fast mode defaults negative_allocation to True.
    """
    def __init__(self, strategy, data,  commission = 0.01, min_commission=2.0, slipage = 0.0, start_offset = 5, include_commission = True, fast = False):
        self.strategy = strategy
        self.data = data
        self.commission = commission
        self.min_commission = min_commission
        self.slipage = slipage
        self.include_commission = include_commission
        self.fast = fast
        self.intraday = self.strategy.intraday
        self.status = pd.DataFrame()
        self.status_closed = self.status.copy()
        self.start_offset = start_offset 

    def update_status(self,datetime):
            """
            Update the status of the backtest

            Args:
                datetime (datetime): datetime of the current row
            """
            setting = {"closed": (self.strategy.closed_trades).copy(),
                        "all": (pd.concat([self.strategy.open_trades, self.strategy.closed_trades], ignore_index=True)).copy()}

            for key, table in setting.items():
                if len(table) == 0 or (self.fast and key == "all"):
                    continue
                
                cur_status = pd.DataFrame()
                cur_status['datetime'] = [datetime]
                cur_status['allocation'] = [self.strategy.allocation]

                sum_commission = table['commission'].sum()     
                if self.include_commission:        
                    rc = (table['reward'] - table['commission'])
                    sum_reward= rc.sum()
                    std_reward =  rc.std() if len(table)>1 else 0
                    avg_reward = rc.mean() 
                else:
                    r = table['reward']
                    sum_reward = r.sum()
                    std_reward =  r.std() if len(table)>1 else 0
                    avg_reward = r.mean() 

                cur_status['total reward'] = [sum_reward]
                cur_status['total commission'] = [sum_commission]
                cur_status['avg reward'] = [avg_reward]
                cur_status['std reward'] = [std_reward]

                cur_status['sharpe ratio'] = [0] if std_reward == 0 else [sum_reward/std_reward]
                cur_status['% reward'] = [0] if sum_reward == 0 else [100*sum_reward/self.strategy.initial_allocation]

                #compute avg risk reward ratio
                entry_mrk_value = table['entry price']*table['position']
                risk_reward_ratio = (entry_mrk_value + table['reward'])/entry_mrk_value
                cur_status['avg risk reward ratio'] = [risk_reward_ratio.mean()]

                #compute max drawdown and avg trade duration
                if key=="closed":
                    entry_datetime = [datetime.combine(table['entry date'].values[i], table['entry time'].values[i]) for i in range(len(table))]
                    exit_datetime = [datetime.combine(table['exit date'].values[i], table['exit time'].values[i]) for i in range(len(table))]
                    trade_duration = [(exit_datetime[i] - entry_datetime[i]).total_seconds()/60 for i in range(len(table))]
                    
                    if self.intraday:
                        cur_status['avg duration (min)'] =  np.mean(trade_duration) #minutes
                    else:
                        cur_status['avg duration (hour)'] =  np.mean(trade_duration)/60 #hours
                    
                    reward_arr = table['reward'].values  # for max drawdown
                else:
                    reward_arr = self.status['total reward'].values if len(self.status)>0 else np.array([0]) # for max drawdown
                total_value = reward_arr + self.strategy.initial_allocation
                
                if len(total_value)==0:
                    cur_status['max drawdown'] = [0]
                else:
                    i = np.argmax(np.maximum.accumulate(total_value) - total_value) # end of the period
                    j = 0 if i==0 else np.argmax(total_value[:i]) # start of period
                    if total_value[j] == 0:
                        cur_status['max drawdown'] = [0]
                    else:
                        cur_status['max drawdown'] = [100*(total_value[i] - total_value[j]) /total_value[j]]
                
                #compute win rate
                cur_status['num trades'] = [len(table)]
                cur_status['win rate'] = [0] if len(table) == 0 else [len(table[table['reward']>0])/len(table)]

                if key == "closed":
                    self.status_closed = pd.concat([self.status_closed,cur_status],ignore_index = True)
                else:
                    self.status = pd.concat([self.status,cur_status],ignore_index = True)


    def summary(self):
        """
        Get the summary of the backtest

        Returns:
            pd.DataFrame: Summary of the backtest
        """
        summary = pd.DataFrame(columns = ['datetime','allocation','total reward','total commission','avg reward','std reward','sharpe ratio','% reward','avg risk reward ratio','avg duration (min)','max drawdown','num trades','win rate'])
       # summary = pd.DataFrame(columns = self.status_closed.columns)

        if len(self.status_closed[-1:])>0:
            #summary.loc['closed',list(self.status_closed[-1].columns)] = self.status_closed[-1:].values
            summary.loc['closed',self.status_closed[-1:].columns] = self.status_closed[-1:].values
        else:
            # set zeros
            summary.loc['closed',summary.columns] = [0]*len(summary.columns)

        if len(self.status[-1:])>0:
            summary.loc['all',self.status[-1:].columns] = self.status[-1:].values
        else:
            # set zeros
            summary.loc['all',summary.columns] = [0]*len(summary.columns)

        summary.drop(labels = ['datetime'],axis = 1, inplace = True)
        
        """
        summary = pd.concat([self.status_closed[-1:],self.status[-1:]])
        indx = []   
        if len(self.status_closed)>0:
            indx += ['closed']
        if len(self.status)>0:
            indx += ['all']
        if len(indx)>0:
            summary.index = indx
            summary.drop(labels = ['datetime'],axis = 1, inplace = True)
        """
        return summary

    def run(self, plot = True):
        """
        Run the backtest

        Args:
            plot (bool, optional): Whether to plot the results. Defaults to True. fast mode does not plot.
        """

        data = self.data
        
        total_time12 = 0
        total_time34 = 0
        if self.fast:
            if self.intraday:
                # get unique date in data
                dates = np.unique(data.index.date)
               # print(dates)
                for d in dates:
                    daily_data = data.loc[data.index.date == d]
                    time1 = time.time()
                    self.strategy.fast_update(daily_data)
                    time2 = time.time()
                    total_time12 += time2-time1
                    datetime = daily_data.index[-1]
                    time3 = time.time()
                    self.update_status(datetime)
                    time4 = time.time()
                    total_time34 += time4-time3
            else:
                time1 = time.time()
                self.strategy.fast_update(data)
                time2 = time.time()
                total_time12 += time2-time1

                datetime = data.index[-1]
                time3 = time.time()
                self.update_status(datetime)
                time4 = time.time()
                total_time34 += time4-time3
        
        else:
            last_timestamp = data.index[-1]
            for i in tqdm(range(self.start_offset,len(data)+1)):

                time1 = time.time()
               # self.strategy.update(data.iloc[:i]) 
                self.strategy.update_open_trades(data.iloc[:i], last_timestamp)
                self.strategy.open_new_trades(data.iloc[:i], last_timestamp)
                
                time2 = time.time()
                total_time12 += time2-time1         
        
                datetime = data.iloc[:i].index[-1]

                time3 = time.time()
                self.update_status(datetime)
                time4 = time.time()
                total_time34 += time4-time3

       # print("Total time strategy update: ", total_time12)
       # print("Total time status update: ", total_time34)

        if plot and not self.fast and len(self.status_closed)>0: 
            b1 = self.status.datetime
            b2 = self.status_closed.datetime

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Total Reward','% Reward','Allocation'), 
                        row_width=[1,1,1])

            fig.add_trace(go.Scatter(x=b2, y=self.status_closed['total reward'],
                          mode='lines',
                          name='closed trades'),row=1, col=1)


            fig.add_trace(go.Scatter(x=b1, y=self.status['total reward'],
                        mode='lines',
                        name="closed + open trades"),row=1, col=1)

            fig.add_trace(go.Scatter(x=b2, y=self.status_closed['% reward'],
                          mode='lines',
                          name='closed trades'),row=2, col=1)
            

            fig.add_trace(go.Scatter(x=b1, y=self.status['allocation'],
                            mode='lines',
                            name="allocation"),row=3, col=1)

        

            fig.add_trace(go.Scatter(x=b1, y=self.status['% reward'],
                        mode='lines',
                        name="closed + open trades"),row=2, col=1)
            rb =[dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(values=["2020-12-25", "2021-01-01"]) ] # hide holidays (Christmas and New Year's, etc)
            if self.strategy.intraday:    
                rb+=[dict(bounds=[16, 9.5], pattern="hour")]    
            fig.update_xaxes( 
                rangebreaks=rb
            )

            # add title
            fig.update_layout(title_text="Reward")
            # add x axis labels with title font size
            fig.update_xaxes(title_text="Date", title_font_size=14)
            fig.update_layout(height=800, width=1300)
            fig.show()

        # reset strategy
       # if reset_strategy:
       #     self.strategy.reset()

        return self.strategy.open_trades, self.strategy.closed_trades