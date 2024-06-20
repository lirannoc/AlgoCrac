"""
Signals module.
todo:
    - forecasting signals, with an easy model forecasting integration. (with corssover with as indicator)
    - time based signals, allowing different times, for example: signals only between 9:30 - 10:30
    - pattern detection with dynamic time warping, such as double top, double bottom, head and shoulders, etc.
"""

import numpy as np
import pandas as pd
import copy
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
import matplotlib.pyplot as plt
from .indicators import *
from .utils import *


ind_func = {"SMA": moving_average, "EMA": exp_moving_average, "VWAP": vwap}


class Signals(ABC):
    """
    Abstract class for signals.

    Args:
        min_timesteps (int): The minimum number of timesteps to retrieve the last signal.
        min_lookback (int): The minimum number of timesteps to look back, if min_timesteps is None. if lower than this argument, returns signal 0.
        signal_type (str): Whether to return the last discrete or continuous last signal {'dis', 'con'}.

    """

    def __init__(
        self,
        #  data,
        min_timesteps=None,  # the minimum number of timesteps to retrieve the last signal, this is useful due to computation efficiency.
        # if None, use the whole data, may result is a difference, since only of a fraction of the data is used to calculate the last signal
        min_lookback=3,  # the minimum number of timesteps to look back, if min_timesteps is None. if lower than this argument, returns signal 0.
        cancel_first_k=1,  # cancel the first k signals, avoids considereing the first k timesteps as signals, this important for quick judgement of the signals.
    ):

        self.min_timesteps = min_timesteps
        if min_timesteps is not None:
            if min_timesteps < min_lookback:
                print("min_timesteps should be larger than or equal to min_lookback")
        self.min_lookback = min_lookback
        self.cancel_first_k = cancel_first_k

        # the order of the symbols corresponding to the signals
        self.symbol_order = []

    # continuous signals for buy or sell, positive for buy and negative for sell   else 0
    @abstractmethod
    def con_signals(self, data):
        """
        Continuous signals.
        """
        pass

    # translate the continuous signals to point-wise discrete signals
    @abstractmethod
    def dis_signals(self, data):
        """
        Translate the continuous signals to point-wise discrete signals.
        """
        con_signals = self.con_signals(data)
        # if self.invert:
        #     con_signals  = con_signals*(-1)
        convolve = partial(np.convolve, mode="same")
        convolve_v = np.vectorize(convolve, signature="(n),(m)->(k)")
        all_signals = convolve_v(con_signals, [1, -1])
        output = np.zeros(all_signals.shape)
        output[all_signals > 0] = 1
        return output

    def get_signals(self, data, signal_type="dis"):
        """
        Returns the signals.

        Args:
            data (pd.DataFrame): The data to be used for the signals.
            signal_type (str): Whether to return the last discrete or continuous last signal {'dis', 'con'}.

        Returns:
            The signals.

        """
        # if data is None:
        #     data = self.data
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        data = data.tail(min_timesteps)

        if len(data) < min_timesteps or len(data) < self.min_lookback:
            signals = np.zeros(
                (len(data.columns.get_level_values(1).unique()), len(data))
            )
        #  if signal_type is None:
        #      signal_type = self.signal_type
        if signal_type == "dis":
            signals = self.dis_signals(data)
        elif signal_type == "con":
            signals = self.con_signals(data)

        # set the symbol order
        self.symbol_order = data.columns.get_level_values(1).unique().tolist()
        return signals

    def get_last_signal(self, data, signal_type="dis"):
        """
        Returns the last signal.

        Args:
            data (pd.DataFrame): The data to be used for the signals.

        Returns:
            The last signal.

        """
        signals = self.get_signals(data, signal_type=signal_type)
        return signals[:, -1]

    def process_plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals, default is according to signal_type
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        intraday=None,  # whether to plot intra day data, if data spans over 1 day, returns different plots for each day.
    ):
        # if data is None:
        #     data = self.data.copy()
        data = data.copy()
        col1 = data.columns.get_level_values(0)[0]
        if len(data[col1].columns) > 1:
            if symbol is None:
                print(
                    "WARNING: Data contains multiple symbols, plotting the first symbol, if you wish to plot a different symbol set symbol = 'symbol'"
                )
                symbol = data[col1].columns[0]
            data = data.iloc[:, data.columns.get_level_values(1) == symbol]

        # check the delta between 2 consecutive timesteps
        # delta = data.index[1:] - data.index[:-1]
        # check if delta is less than 1 day

        # intraday = check_intraday(data = data) or intraday
        # if  (intraday in [True,None]) and np.any(delta < pd.Timedelta(minutes=30)):
        #    if not intraday:
        #        print("WARNING: Data is not daily, setting intraday = True")
        #        intraday = True
        # intraday = check_intraday(data = data) or intraday
        # delta = data.index[1:] - data.index[:-1]
        # if intraday and not np.any(delta < pd.Timedelta(days=1)):
        #     raise Exception("Data is not intraday, cannot plot intraday data")

        if not set(["Close", "High", "Low", "Open", "Volume"]).issubset(
            data.columns.get_level_values(0)
        ):
            raise Exception("Data does not contain all the required columns")
        #  target_data = data[self.on].to_numpy()

        if signal_type is None:
            signal_type = self.signal_type
        if signals is None:
            if intraday:
                print("plot intraday data")
                signals_list = []
                unique_dates = np.unique(np.array([d.date() for d in data.index]))
                for d in unique_dates:
                    date_data = data.loc[data.index.date == d]
                    if signal_type == "con":
                        signals_list.append(self.con_signals(data=date_data)[0])
                    elif signal_type == "dis":
                        signals_list.append(self.dis_signals(data=date_data)[0])
                signals = np.concatenate(signals_list)
            else:
                if signal_type == "con":
                    signals = self.con_signals(data=data)[0]
                elif signal_type == "dis":
                    signals = self.dis_signals(data=data)[0]

        # b = [to_datetime(str(t)) for t in data.index]
        b = data.index
        data.columns = data.columns.droplevel(1)
        # Create subplots and mention plot grid size
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(symbol, "Volume"),
            row_width=[0.17, 0.17, 0.3, 1],
        )

        # Plot OHLC on 1st row
        fig.add_trace(
            go.Candlestick(
                x=b,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # plot the signals
        fig.add_trace(
            go.Scatter(
                x=b,
                y=signals,
                mode="markers",
                name="signals",
                line=dict(color="royalblue"),
            ),
            row=3,
            col=1,
        )

        fig.add_trace(go.Bar(x=b, y=data["Volume"], showlegend=False), row=2, col=1)
        # Do not show OHLC's rangeslider plot
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(height=height, width=width)

        rb = [
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(values=["2020-12-25", "2021-01-01"]),
        ]  # hide holidays (Christmas and New Year's, etc)

        # show intraday trading hours only
        if intraday is True:
            rb += [dict(bounds=[16, 9.5], pattern="hour")]

        fig.update_xaxes(rangebreaks=rb)
        return fig, data

    # fig.show()

    def plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals, default is according to signal_type
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        intraday=None,  # whether to plot intra day data, if data spans over 1 day, returns different plots for each day.
    ):

        intraday = check_intraday(data=data) or intraday
        delta = data.index[1:] - data.index[:-1]
        if intraday and not np.any(delta < pd.Timedelta(days=1)):
            raise Exception("Data is not intraday, cannot plot intraday data")

        self.process_plot_signals(
            symbol=symbol,
            signal_type=signal_type,
            signals=signals,
            height=height,
            width=width,
            data=data,
            intraday=intraday,
        )[0].show()


# Crossover signals w.r.t. the given data
class Crossover(Signals):
    """
    Crossover signals w.r.t. the given data.

    Args:
        period (int): The period for the crossover indicator.
        indicator (str): The indicator function to be used.
        on (str): The column to be used for the crossover.
        invert (bool): Whether to invert the signal.
        eps (float): The minimum fraction of the target to be surpassed by the indicator.

    """

    # indicator: a function for applying the indicator to retrieve the signals
    # invert: simply changes the default signal from 1 to -1 and vice versa.
    def __init__(
        self,
        period=None,
        indicator=None,
        on="Close",  # the column to be used for the crossover e.g. {Close, Open, High, Low, Volume}
        invert=False,  # invert the signal, can return when to exit a position.
        eps=0.0,  # allow only signals that the indicator surpasses a certain fraction of the target, [0,inf]
        min_timesteps=None,
        min_lookback=3,
        cancel_first_k=1,
    ):
        # if period is None:
        # period = len(data)
        #     min_timesteps = period
        #  else:
        # the minimum number of timesteps to retrieve the last signal, is the period + the last signal
        #      min_timesteps = period+1
        super().__init__(
            min_timesteps=min_timesteps,
            min_lookback=min_lookback,
            cancel_first_k=cancel_first_k,
        )

        self.period = period
        self.invert = invert
        self.indicator = indicator
        # check if eps is in [0,1]
        if eps < 0:
            raise Exception("eps should be in [0,inf]")
        self.eps = eps
        self.on = on

        self.profile = {
            "class": "Crossover",
            "params": {
                "period": period,
                "indicator": indicator,
                "on": on,
                "invert": invert,
                "eps": eps,
                "min_timesteps": min_timesteps,
                "min_lookback": min_lookback,
                "cancel_first_k": cancel_first_k,
            },
        }

        """
        self.intraday_space = {"period": np.arange(1,25),
                                "indicator": ["SMA", "EMA", "VWAP"],
                                "on": ["Close", "Open", "High", "Low", "Volume"],
                                "invert": [True, False],
                                "eps": np.concatenate([np.arange(0,1,0.1),np.arange(1,4.5,0.5)]),
                                } # the indicator to be used for intraday data
        
        self.space = {"period": np.arange(1,200),
                                "indicator": ["SMA", "EMA"],
                                "on": ["Close", "Open", "High", "Low", "Volume"],
                                "invert": [True, False],
                                "eps": np.concatenate([np.arange(0,1,0.1),np.arange(1,4.5,0.5)]),
                                }

        self.intraday_space = copy.deepcopy(self.space)# the indicator to be used for intraday data
        self.intraday_space["indicator"] = ["SMA", "EMA", "VWAP"]
        self.intraday_space["period"] = np.arange(1,31)

        self.intraday_space_mini = copy.deepcopy(self.intraday_space)
        self.intraday_space_mini["period"] = [5,10,15,20,30]
        self.intraday_space_mini["eps"] = [0,0.1,0.25]

        """

    def apply_indicator(self, data):
        # if target_data is None:
        #     target_data = self.target_data
        target_data = data[self.on].to_numpy().T
        #  if data is None:
        #      data = self.data
        if self.period is None or len(data) < self.period:
            period = len(data)
        else:
            period = self.period
        if self.indicator is None:
            raise Exception("crossover indicator not given")
        else:
            if self.indicator == "VWAP":
                cols = data.columns.get_level_values(0).unique()
                if not set(["Close", "High", "Low", "Open", "Volume"]).issubset(cols):
                    raise Exception("Data does not contain all the required columns")
                return ind_func[self.indicator](
                    close=data["Close"].to_numpy().T,
                    high=data["High"].to_numpy().T,
                    low=data["Low"].to_numpy().T,
                    volume=data["Volume"].to_numpy().T,
                    k=period,
                )
            else:
                return ind_func[self.indicator](data=target_data, k=period)

    def con_signals(self, data):
        if self.indicator is None:
            raise Exception("neither crossover data nor crossover function were given")
        # if data is None:
        #  data = self.data
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        if len(data) < min_timesteps or len(data) < self.min_lookback:
            return np.zeros((len(data.columns.get_level_values(1).unique()), len(data)))
        target_data = data[self.on].to_numpy().T
        co_data = self.apply_indicator(data=data)
        c_signals = target_data - co_data
        c_signals = np.divide(c_signals, co_data)
        signals = np.zeros(c_signals.shape)
        signals[c_signals > self.eps] = 1
        self.symbol_order = data.columns.get_level_values(1).unique().tolist()
        if self.invert:
            signals = 1 - signals
        if self.cancel_first_k > 0:
            signals[:, : self.cancel_first_k] = 0
        return signals

    def dis_signals(self, data):
        # call from super class
        output = super().dis_signals(data=data)
        return output

    def plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals, default is according to signal_type
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        #  data = None, # data to plot
        intraday=None,  # whether to plot intra day data, if data spans over 1 day, returns different plots for each day.
    ):

        intraday = check_intraday(data=data) or intraday
        delta = data.index[1:] - data.index[:-1]
        if intraday and not np.any(delta < pd.Timedelta(days=1)):
            raise Exception("Data is not intraday, cannot plot intraday data")

        # call plot_signal from the base class
        fig, cur_data = self.process_plot_signals(
            symbol=symbol,
            signal_type=signal_type,
            signals=signals,
            height=height,
            width=width,
            data=data,
            intraday=intraday,
        )

        # timesteps = np.arange(len(data.index))
        # timesteps = data.index

        if intraday:
            indi_list = []
            unique_dates = np.unique(np.array([d.date() for d in data.index]))
            for d in unique_dates:
                intraday_data = cur_data.loc[cur_data.index.date == d]
                indi_list.append(self.apply_indicator(data=intraday_data)[0])
            indi = np.concatenate(indi_list)
        else:
            indi = self.apply_indicator(data=cur_data)[0]
        if self.on == "Volume":
            ind_row = 2
        else:
            ind_row = 1
        # plot the indicator
        name = f"{self.indicator} {str(self.period)}, on {self.on}"
        fig.add_trace(
            go.Scatter(x=cur_data.index, y=indi, mode="lines", name=name),
            row=ind_row,
            col=1,
        )
        fig.show()


# a signal class that combines multiple signals
class Confluence(Signals):
    def __init__(
        self,
        signals_list=None,
        # weak_signals_list = None,
        # weak_signals_list_on = None,
        max_signals=None,  # the maximum signals to be considered
        invert=False,
    ):
        super().__init__()
        self.signals_list = signals_list
        self.invert = invert
        if self.signals_list is None or len(self.signals_list) == 0:
            raise Exception("no signals were given")

        if max_signals is None or max_signals > len(signals_list):
            self.max_signals = len(signals_list)
        else:
            self.max_signals = max_signals

        if self.max_signals < 0:
            raise Exception("max_signals must be positive")

        self.profile = {
            "class": "Confluence",
            "signals_list": [s.profile for s in self.signals_list],
        }

        """
        self.space = {"max_signals": np.arange(1,len(signals_list)+1),
                      "invert": [True, False]}
        self.space_list = [s.space for s in self.signals_list]

        self.intraday_space = self.space
        self.intraday_space_list = [s.intraday_space for s in self.signals_list]

        self.intraday_space_mini = self.space
        self.intraday_space_mini_list = [s.intraday_space_mini for s in self.signals_list]
        """

    def con_signals(self, data):
        #  if data is None:
        #      data = self.data
        c_signals = 0.0 # must be float
        for s in self.signals_list:
            #print(type(s.con_signals(data=data)))
            c_signals += s.con_signals(data=data)

        signals = (c_signals / self.max_signals).astype(int)
        signals[signals >= 1] = 1
        self.symbol_order = data.columns.get_level_values(1).unique().tolist()

        if self.invert:
            signals = 1 - signals
        if self.cancel_first_k > 0:
            signals[:, : self.cancel_first_k] = 0
        return signals

    # this function is repeated, similar to the one in crossover.
    def dis_signals(self, data):
        # call from super class
        output = super().dis_signals(data=data)
        return output

    def plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        intraday=None,  # whether to plot intraday data
    ):

        intraday = check_intraday(data=data) or intraday
        delta = data.index[1:] - data.index[:-1]
        if intraday and not np.any(delta < pd.Timedelta(days=1)):
            raise Exception("Data is not intraday, cannot plot intraday data")

        # call plot_signal from the base class
        fig, cur_data = self.process_plot_signals(
            symbol=symbol,
            signal_type=signal_type,
            signals=signals,
            height=height,
            width=width,
            data=data,
            intraday=intraday,
        )

        # timesteps = np.arange(len(data.index))
        # timesteps = data.index

        # plot the indicator
        for s in self.signals_list:
            # check if s has apply_indicator
            if hasattr(s, "apply_indicator"):
                if intraday:
                    indi_list = []
                    unique_dates = np.unique(np.array([d.date() for d in data.index]))
                    for d in unique_dates:
                        intraday_data = data.loc[data.index.date == d]
                        indi_list.append(s.apply_indicator(data=intraday_data)[0])
                    indi = np.concatenate(indi_list)
                else:
                    indi = s.apply_indicator(data=cur_data)[0]
                # indi = s.apply_indicator(data = cur_data)[0]
                if s.on == "Volume":
                    ind_row = 2
                else:
                    ind_row = 1
                name = f"{s.indicator} {str(s.period)}, on {s.on}"
                fig.add_trace(
                    go.Scatter(x=cur_data.index, y=indi, mode="lines", name=name),
                    row=ind_row,
                    col=1,
                )
        fig.show()


class RelativeSize(Signals):
    def __init__(
        self,
        period=10,  # period for determining the relative size
        invert=False,  # if true, then the signal is inverted
        size_frac=1,  # fraction of the average size to be used as a threshold
        cr_max="High",  # criteria for the maximum value to be used for the size calculation
        cr_min="Low",  # criteria for the minimum value to be used for the size calculation,
        min_timesteps=None,  # minimum number of timesteps to be used for the calculation,
        min_lookback=3,  # minimum lookback to be used for the calculation
        cancel_first_k=1,  # cancel the first k signals, avoids considereing the first k timesteps as signals, this important for quick judgement of the signals.
    ):

        super().__init__(
            min_timesteps=min_timesteps,
            min_lookback=min_lookback,
            cancel_first_k=cancel_first_k,
        )
        self.period = period
        self.cr_max = cr_max
        self.cr_min = cr_min
        self.invert = invert
        self.size_frac = size_frac

        self.profile = {
            "class": "RelativeSize",
            "params": {
                "period": period,
                "invert": invert,
                "size_frac": size_frac,
                "cr_max": cr_max,
                "cr_min": cr_min,
                "min_timesteps": min_timesteps,
                "min_lookback": min_lookback,
                "cancel_first_k": cancel_first_k,
            },
        }
        """
        self.space = {"period": np.arange(1,200),
                                    "invert": [True, False],
                                    "size_frac": np.concatenate([np.arange(0.5,3.5,0.25)]),
                                    "cr_max": ["High", "Close", "Open", "Low"],
                                    "cr_min": ["Low", "Close", "Open", "High"],
                                    }
        self.intraday_space = copy.deepcopy(self.space)
        self.intraday_space["period"] = np.arange(1,31)

        self.intraday_space_mini = copy.deepcopy(self.space)
        self.intraday_space_mini["period"] = [5,10,15,20,30]
        self.intraday_space_mini["size_frac"] = [0.5,1,1.5,2,2.5,3]
        """

    def con_signals(self, data):
        # if data is None:
        #     data = self.data
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        if len(data) < min_timesteps or len(data) < self.min_lookback:
            return np.zeros((len(data.columns.get_level_values(1).unique()), len(data)))
        period = self.period

        max = data[self.cr_max].values.T
        min = data[self.cr_min].values.T
        if len(max.shape) == 1:
            max = max.reshape(1, -1)
        if len(min.shape) == 1:
            min = min.reshape(1, -1)
        size = np.abs(max - min)
        kernel = np.ones([period + 1])
        kernel[1:] = 1 / (period)
        kernel[0] = -kernel[0] * self.size_frac
        convolve = partial(np.convolve, mode="full")
        convolve_v = np.vectorize(convolve, signature="(n),(m)->(k)")
        c_signals = convolve_v(size, kernel)[:, :-(period)]
        signals = np.zeros(c_signals.shape)
        signals[c_signals < 0] = 1
        self.symbol_order = data.columns.get_level_values(1).unique().tolist()

        if self.cancel_first_k > 0:
            signals[:, : self.cancel_first_k] = 0

        return signals

    def dis_signals(self, data):
        return self.con_signals(data=data)


class BasicMetric(Signals):
    def __init__(
        self,
        condition_value,  # value to compare
        metric="mean",  # max, min, mean, median, min, max, std, last, first
        condition="greater",  # greater, less, equal
        min_timesteps=None,
        min_lookback=3,
        cancel_first_k=1,
        on="Close",  # Open, High, Low, Close, Volume
        invert=False,
        period=None,
    ):
        super().__init__(
            min_timesteps=min_timesteps,
            min_lookback=min_lookback,
            cancel_first_k=cancel_first_k,
        )
        self.condition_value = condition_value
        self.metric = metric
        self.condition = condition

        self.on = on
        self.invert = invert
        self.period = period

        self.profile = {
            "class": "BasicMetric",
            "params": {
                "condition_value": condition_value,
                "metric": metric,
                "condition": condition,
                "on": on,
                "invert": invert,
                "period": period,
                "min_timesteps": min_timesteps,
                "min_lookback": min_lookback,
                "cancel_first_k": cancel_first_k,
            },
        }
        """
        self.space = {"metric": ["mean", "median", "max", "min", "std", "last", "first"],
                      "condition": ["greater", "less", "equal"],
                      "on": ["Close", "Open", "High", "Low", "Volume"],
                      "invert": [True, False],
                      "period": np.arange(1,200),
                      }
        self.intraday_space = copy.deepcopy(self.space)
        self.intraday_space["period"] = np.arange(1,31)

        self.intraday_space_mini = copy.deepcopy(self.intraday_space)
        self.intraday_space_mini["period"] = [5,10,15,20,30]
        """

    def get_metrics_data(self, data, metric=None, period=None):
        if metric is None:
            metric = self.metric
        if period is None:
            period = self.period
        if period is None or len(data) < period:
            period = len(data)
        if metric == "mean":
            # co_data = target_data.cumsum(axis = 0)
            # co_data = co_data/np.arange(1,len(target_data[0])+1).reshape(-1,1)
            m_data = data[self.on].rolling(window=period, min_periods=1).mean().values
        elif metric == "sum":
            m_data = data[self.on].rolling(window=period, min_periods=1).sum().values
        elif metric == "median":
            m_data = data[self.on].rolling(window=period, min_periods=1).median().values
        elif metric == "max":
            m_data = data[self.on].rolling(window=period, min_periods=1).max().values
        elif metric == "min":
            m_data = data[self.on].rolling(window=period, min_periods=1).min().values
        elif metric == "std":
            m_data = data[self.on].rolling(window=period, min_periods=1).std().values
            # fill nan with value 0
            m_data = np.nan_to_num(m_data)
        elif metric == "last":
            m_data = data[self.on].values
        elif metric == "first":
            m_data = np.tile(data[self.on].values[0, :], (len(data), 1))
        else:
            raise ValueError("metric not recognized")

        if len(m_data.shape) == 1:
            m_data = m_data.reshape(1, -1)
        else:
            m_data = m_data.T
        return m_data

    def condition_signals(self, m_data, condition_value=None, condition=None):

        if condition_value is None:
            condition_value = self.condition_value
        if condition is None:
            condition = self.condition

        c_signals = m_data - condition_value

        if condition in ["greater", "above", "over", "higher"]:
            c_signals = c_signals > 0
        elif condition in ["less", "below", "under", "lower"]:
            c_signals = c_signals < 0
        elif condition in ["equal", "equals", "same"]:
            c_signals = c_signals == 0
        else:
            raise ValueError("condition not recognized")

        return c_signals.astype(int)

    def con_signals(self, data):
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        if len(data) < min_timesteps or len(data) < self.min_lookback:
            return np.zeros((len(data.columns.get_level_values(1).unique()), len(data)))
        m_data = self.get_metrics_data(data)
        c_signals = self.condition_signals(m_data)

        if self.invert:
            c_signals = 1 - c_signals
        return c_signals

    def dis_signals(self, data):
        # call from super class
        output = super().dis_signals(data=data)
        return output

    def plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals, default is according to signal_type
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        #  data = None, # data to plot
        intraday=None,  # whether to plot intra day data, if data spans over 1 day, returns different plots for each day.
    ):

        intraday = check_intraday(data=data) or intraday
        delta = data.index[1:] - data.index[:-1]
        if intraday and not np.any(delta < pd.Timedelta(days=1)):
            raise Exception("Data is not intraday, cannot plot intraday data")

        # call plot_signal from the base class
        fig, cur_data = self.process_plot_signals(
            symbol=symbol,
            signal_type=signal_type,
            signals=signals,
            height=height,
            width=width,
            data=data,
            intraday=intraday,
        )

        if intraday:
            indi_list = []
            unique_dates = np.unique(np.array([d.date() for d in cur_data.index]))
            for d in unique_dates:
                intraday_data = cur_data.loc[cur_data.index.date == d]
                indi_list.append(self.get_metrics_data(data=intraday_data)[0])
            indi = np.concatenate(indi_list)
        else:
            indi = self.get_metrics_data(data=cur_data)[0]
        if self.on == "Volume":
            ind_row = 2
        else:
            ind_row = 1
        # plot the indicator

        name = f'{self.metric} {str(self.period) if not None else " "}, on {self.on}'
        if self.metric == "std":
            fig.add_trace(
                go.Scatter(
                    x=cur_data.index,
                    y=cur_data[self.on].values + indi,
                    mode="lines",
                    marker_color="rgba(0,0,0,0.2)",
                    name=name,
                ),
                row=ind_row,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=cur_data.index,
                    y=cur_data[self.on].values - indi,
                    mode="lines",
                    marker_color="rgba(0,0,0,0.2)",
                    name=name,
                ),
                row=ind_row,
                col=1,
            )

            name = f"condition value: {self.condition_value}, on {self.on}"
            fig.add_trace(
                go.Scatter(
                    x=cur_data.index,
                    y=cur_data[self.on].values + self.condition_value,
                    mode="lines",
                    marker_color="rgba(0,0,0,0.5)",
                    name=name,
                ),
                row=ind_row,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=cur_data.index,
                    y=cur_data[self.on].values - self.condition_value,
                    mode="lines",
                    marker_color="rgba(0,0,0,0.5)",
                    name=name,
                ),
                row=ind_row,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=cur_data.index, y=indi, mode="lines+markers", name=name),
                row=ind_row,
                col=1,
            )
            name = f"condition value: {self.condition_value}, on {self.on}"
            fig.add_trace(
                go.Scatter(
                    x=cur_data.index,
                    y=[self.condition_value] * len(cur_data.index),
                    mode="lines+markers",
                    name=name,
                ),
                row=ind_row,
                col=1,
            )

        fig.show()


class Range(BasicMetric):
    def __init__(
        self,
        max_val,  # value to compare
        min_val=0,
        metric="mean",  # max, min, mean, median, min, max, last, first
        in_range=True,  # whether to check if the metric is in the range or out
        min_timesteps=None,
        min_lookback=3,
        cancel_first_k=1,
        on="Close",  # Open, High, Low, Close, Volume
        invert=False,
    ):
        if metric == "std":
            raise Exception("std is not supported for Range")

        period = None
        super().__init__(
            condition_value=max_val,
            metric=metric,
            condition="above",  # not used
            min_timesteps=min_timesteps,
            min_lookback=min_lookback,
            cancel_first_k=cancel_first_k,
            on=on,
            invert=invert,
            period=period,
        )
        self.min_val = min_val
        self.max_val = max_val

        if self.min_val > self.max_val:
            # raise warning
        #    print(
        #        "min_val is greater than max_val, setting min_val to max_val and max_val to min_val"
        #    )
            self.min_val, self.max_val = self.max_val, self.min_val

        self.in_range = in_range
        self.profile = {
            "class": "Range",
            "params": {
                "max_val": max_val,
                "min_val": min_val,
                "metric": metric,
                "in_range": in_range,
                "on": on,
                "invert": invert,
                "period": period,
                "min_timesteps": min_timesteps,
                "min_lookback": min_lookback,
                "cancel_first_k": cancel_first_k,
            },
        }

        """
        self.space["metric"] = ["mean", "median", "max", "min", "last", "first"]
        self.space["min_val"] = [0,1,5,10,20,50,100,200,500,1000]
        self.space["max_val"] = [1,5,10,20,50,100,200,500,1000]
        self.space["in_range"] = [True, False]

        self.intraday_space = copy.deepcopy(self.space)
        self.intraday_space["period"] = np.arange(1,31)

        self.intraday_space_mini = copy.deepcopy(self.intraday_space)
        self.intraday_space_mini["period"] = [5,10,15,20,30]
        """

    def con_signals(self, data):
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        if len(data) < min_timesteps or len(data) < self.min_lookback:
            return np.zeros((len(data.columns.get_level_values(1).unique()), len(data)))

        m_data = self.get_metrics_data(data, metric=self.metric, period=self.period)
        condition_max = "less" if self.in_range else "greater"
        c_signals_less = self.condition_signals(
            m_data, condition_value=self.max_val, condition=condition_max
        )
        condition_min = "greater" if self.in_range else "less"
        c_signals_greater = self.condition_signals(
            m_data, condition_value=self.min_val, condition=condition_min
        )

        if self.in_range:
            c_signals = c_signals_greater * c_signals_less
        else:
            c_signals = c_signals_greater + c_signals_less
        if self.invert:
            c_signals = 1 - c_signals
        return c_signals

    def plot_signals(
        self,
        data,
        symbol=None,  # symbol to plot
        signal_type=None,  # whether to plot continous signals, default is according to signal_type
        signals=None,  # signals to plot
        height=800,  # height of the plot
        width=1000,  # width of the plot
        #  data = None, # data to plot
        intraday=None,  # whether to plot intra day data, if data spans over 1 day, returns different plots for each day.
    ):

        intraday = check_intraday(data=data) or intraday
        delta = data.index[1:] - data.index[:-1]
        if intraday and not np.any(delta < pd.Timedelta(days=1)):
            raise Exception("Data is not intraday, cannot plot intraday data")

        # call plot_signal from the base class
        fig, cur_data = self.process_plot_signals(
            symbol=symbol,
            signal_type=signal_type,
            signals=signals,
            height=height,
            width=width,
            data=data,
            intraday=intraday,
        )

        if intraday:
            indi_list = []
            unique_dates = np.unique(np.array([d.date() for d in cur_data.index]))
            for d in unique_dates:
                intraday_data = cur_data.loc[cur_data.index.date == d]
                indi_list.append(self.get_metrics_data(data=intraday_data)[0])
            indi = np.concatenate(indi_list)
        else:
            indi = self.get_metrics_data(data=cur_data)[0]
        if self.on == "Volume":
            ind_row = 2
        else:
            ind_row = 1
        # plot the indicator

        name = f'{self.metric} {str(self.period) if not None else " "}, on {self.on}'
        fig.add_trace(
            go.Scatter(x=cur_data.index, y=indi, mode="lines+markers", name=name),
            row=ind_row,
            col=1,
        )

        name = f"min value: {self.min_val}, on {self.on}"
        fig.add_trace(
            go.Scatter(
                x=cur_data.index,
                y=[self.min_val] * len(cur_data.index),
                mode="lines",
                name=name,
            ),
            row=ind_row,
            col=1,
        )
        name = f"max value: {self.max_val}, on {self.on}"
        fig.add_trace(
            go.Scatter(
                x=cur_data.index,
                y=[self.max_val] * len(cur_data.index),
                mode="lines",
                name=name,
            ),
            row=ind_row,
            col=1,
        )

        fig.show()


class DatetimeRange(Signals):
    def __init__(
        self,
        start_time=None,  # value to compare
        end_time=None,  # value to compare
        start_date=None,
        end_date=None,
        tod=None,  # time of day for intraday data
        min_timesteps=None,
        min_lookback=3,
        cancel_first_k=1,
        invert=False,
    ):
        super().__init__(
            min_timesteps=min_timesteps,
            min_lookback=min_lookback,
            cancel_first_k=cancel_first_k,
        )
        self.start_time = start_time
        self.end_time = end_time
        self.start_date = start_date
        self.end_date = end_date

        self.invert = invert

        self.tod = tod
        if self.tod is not None:
            if self.tod not in ["open", "close", "mid"]:
                raise Exception("tod should be either open, close or mid")

        # check if start_time is after end_time
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise Exception("start_time should be before end_time")

        # check if start_date is after end_date
        if self.start_date is not None and self.end_date is not None:
            if self.start_date >= self.end_date:
                raise Exception("start_date should be before end_date")

        self.profile = {
            "class": "DatetimeRange",
            "params": {
                "start_time": start_time,
                "end_time": end_time,
                "start_date": start_date,
                "end_date": end_date,
                "tod": tod,
                "invert": invert,
                "min_timesteps": min_timesteps,
                "min_lookback": min_lookback,
                "cancel_first_k": cancel_first_k,
            },
        }
        """    
        self.space = {}
        self.intraday_space = {"tod": ["open", "close", "mid"],
                               "invert": [True, False]}
        self.intraday_space_mini = copy.deepcopy(self.intraday_space)
        """

    def con_signals(self, data):
        if self.min_timesteps is None:
            min_timesteps = len(data)
        else:
            min_timesteps = self.min_timesteps
        if len(data) < min_timesteps or len(data) < self.min_lookback:
            return np.zeros((len(data.columns.get_level_values(1).unique()), len(data)))

        datetime_inds = pd.Series(index=data.index, data=np.arange(len(data.index)))
        if self.start_date is not None:
            datetime_inds = datetime_inds.loc[
                datetime_inds.index.date >= self.start_date
            ]
        if self.end_date is not None:
            datetime_inds = datetime_inds.loc[datetime_inds.index.date <= self.end_date]

        if self.start_time is not None:
            datetime_inds = datetime_inds.loc[
                datetime_inds.index.time >= self.start_time
            ]
        if self.end_time is not None:
            datetime_inds = datetime_inds.loc[datetime_inds.index.time <= self.end_time]
        if self.tod is not None:
            if self.tod == "open":
                datetime_inds = datetime_inds.between_time("09:30", "10:30")
            elif self.tod == "mid":
                datetime_inds = datetime_inds.between_time("10:30", "14:30")
            elif self.tod == "close":
                datetime_inds = datetime_inds.between_time("14:30", "16:00")
        c_signals = np.zeros(len(data))
        c_signals[datetime_inds.values] = 1

        # get number of symbols
        n_symbols = len(data.columns.get_level_values(1).unique())
        signals = np.tile(c_signals, (n_symbols, 1))

        if self.invert:
            signals = 1 - signals
        return signals

    def dis_signals(self, data):
        # call from super class
        output = super().dis_signals(data=data)
        return output
