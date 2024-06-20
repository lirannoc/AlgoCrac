
import numpy as np
import pandas as pd
import datetime

def to_datetime(d):
    if len(d.split("-"))==4:
        dtime = pd.Timestamp(d)
        dtime = dtime.to_pydatetime().replace(tzinfo=None )
    else:
        dtime = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') 
    return dtime


def check_intraday(data, intraday_delta = 30):
        # check the delta between 2 consecutive timesteps
        delta = data.index[1:] - data.index[:-1]
        # check if delta is less than 1 day
        if  np.any(delta < pd.Timedelta(minutes=intraday_delta)):
            return True
        else:
            return False