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
from .Signals import *


space = {
    "Crossover": {
        "class": Crossover,
        "params": {
            "period": np.arange(2, 200),
            "indicator": ["SMA", "EMA"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "eps": np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 4.5, 0.5)]),
        },
    },
    # "Confluence": {"class": Confluence,
    #                "params": {"max_signals": np.arange(1,5),
    #                "invert": [True, False],
    #               }},
    "RelativeSize": {
        "class": RelativeSize,
        "params": {
            "period": np.arange(1, 200),
            "invert": [True, False],
            "size_frac": np.concatenate([np.arange(0.5, 3.5, 0.25)]),
            "cr_max": ["High", "Close", "Open", "Low"],
            "cr_min": ["Low", "Close", "Open", "High"],
        },
    },
    "BasicMetric": {
        "class": BasicMetric,
        "params": {
            "metric": ["mean", "median", "max", "min", "std", "last", "first"],
            "condition_value": np.arange(0, 1000, 1),
            "condition": ["greater", "less", "equal"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "period": np.arange(1, 200),
        },
    },
    "Range": {
        "class": Range,
        "params": {
            "max_val": np.arange(0, 500, 5),
            "min_val": np.arange(0, 500, 5),
            "metric": ["mean", "median", "max", "min",  "last", "first"],
            "in_range": [True, False],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
        },
    },
}


intraday_space = {
    "Crossover": {
        "class": Crossover,
        "params": {
            "period": np.arange(2, 31),
            "indicator": ["SMA", "EMA", "VWAP"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "eps": np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 4.5, 0.5)]),
        },
    },
    # "Confluence": {"class": Confluence,
    #                 "params": {"max_signals": np.arange(1,5),
    #                 "invert": [True, False],
    #                 }},
    "RelativeSize": {
        "class": RelativeSize,
        "params": {
            "period": np.arange(1, 31),
            "invert": [True, False],
            "size_frac": np.concatenate([np.arange(0.5, 3.5, 0.25)]),
            "cr_max": ["High", "Close", "Open", "Low"],
            "cr_min": ["Low", "Close", "Open", "High"],
        },
    },
    "BasicMetric": {
        "class": BasicMetric,
        "params": {
            "metric": ["mean", "median", "max", "min", "std", "last", "first"],
            "condition_value": np.arange(0, 500, 5),
            "condition": ["greater", "less", "equal"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "period": np.arange(1, 31),
        },
    },
    "Range": {
        "class": Range,
        "params": {
            "max_val": np.arange(0, 500, 5),
            "min_val": np.arange(0, 500, 5),
            "metric": ["mean", "median", "max", "min", "last", "first"],
            "in_range": [True, False],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
        },
    },
    "DatetimeRange": {
        "class": DatetimeRange,
        "params": {
            "tod": ["open", "close", "mid"],
            "invert": [True, False],
        },
    },
}


intraday_space_mini = {
    "Crossover": {
        "class": Crossover,
        "params": {
            "period": [5, 10, 15, 20, 30],
            "indicator": ["SMA", "EMA", "VWAP"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "eps": [0, 0.1, 0.25],
        },
    },
    #  "Confluence": {"class": Confluence,
    #                  "params": {"max_signals": np.arange(1,5),
    #                  "invert": [True, False],
    #                  }},
    "RelativeSize": {
        "class": RelativeSize,
        "params": {
            "period": [5, 10, 15, 20, 30],
            "invert": [True, False],
            "size_frac": [0.5, 1, 1.5, 2, 2.5, 3],
            "cr_max": ["High", "Close", "Open", "Low"],
            "cr_min": ["Low", "Close", "Open", "High"],
        },
    },
    "BasicMetric": {
        "class": BasicMetric,
        "params": {
            "metric": ["mean", "median", "max", "min", "std", "last", "first"],
            "condition_value": np.arange(0, 100, 5),
            "condition": ["greater", "less", "equal"],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
            "period": np.arange(1, 31),
        },
    },
    "Range": {
        "class": Range,
        "params": {
            "max_val": np.arange(0, 500, 5),
            "min_val": np.arange(0, 500, 5),
            "metric": ["mean", "median", "max", "min", "last", "first"],
            "in_range": [True, False],
            "on": ["Close", "Open", "High", "Low", "Volume"],
            "invert": [True, False],
        },
    },
    "DatetimeRange": {
        "class": DatetimeRange,
        "params": {
            "tod": ["open", "close", "mid"],
            "invert": [True, False],
        },
    },
}
