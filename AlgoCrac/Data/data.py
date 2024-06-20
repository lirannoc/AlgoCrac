
import numpy as np
import pandas as pd
import yfinance as yf
import datetime


def to_datetime(d):
    if len(d.split("-"))==4:
        dtime = pd.Timestamp(d)
        dtime = dtime.to_pydatetime().replace(tzinfo=None )
    else:
        dtime = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') 
    return dtime

def get_data(symbols, start, end, interval, source = 'Yahoo'):
    """
    import OHLC data.
    Parameters
    ----------
    symbols : list
        symbols to import.
    start : str
        start date.
    end : str
        end date.
    interval : str
        interval of data.
    source : str, optional
        source of data. The default is 'Yahoo'. 
    """
    if source == 'Yahoo':
        symbols = " ".join(symbols)
        data = yf.download(symbols, start = start, end = end, interval = interval)
        if len(symbols.split(" "))==1:
            cols = [(col,symbols ) for col in data.columns]
            data.columns = pd.MultiIndex.from_tuples(cols)

        # bug fix
        # change index dtype to datetime64[ns]
        data.index = pd.to_datetime(data.index)
        start_timestap = datetime.datetime.strptime(start, "%Y-%m-%d").date()
        end_timestap = datetime.datetime.strptime(end, "%Y-%m-%d").date()
        
        data = data.loc[(data.index.date >= start_timestap) & (data.index.date < end_timestap)]
            
        return pre_process(data)
    else:
        raise ValueError("Source not supported")

def pre_process(data):
    """
    pre process data.
    Parameters
    ----------
    data : pd.DataFrame
        data to pre process.
    Returns
    -------
    data : pd.DataFrame
        pre processed data.
    """
    # drop symbols with only NaN values
    nan_cols = data.columns[data.isna().all()].tolist()
    if len(nan_cols)>0:
        nan_symbols = np.unique([col[1] for col in nan_cols])
        # drop columns with only NaN values
        data = data.drop(nan_symbols, axis = 1, level = 1)

    #fill NaN values with the previous value
    data = data.fillna(method='ffill')
    #fill NaN values with the next value
    data = data.fillna(method='bfill')

    # change data['Volume'] values 0 to 0.0001
    data['Volume'] = data['Volume'].replace(0,0.0001)


    dt_index = [to_datetime(str(t)) for t in data.index]
    data.index = dt_index
    return data        

def get_SandP500_data(start, end, interval, source = 'Yahoo'):
    """
    import S&P500 OHLC data.
    Parameters
    ----------
    start : str
        start date.
    end : str
        end date.
    interval : str

    Returns
    -------
    data : pd.DataFrame
    """
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = sp500[0]['Symbol'].values
    return get_data(sp500, start, end, interval, source )

def get_nasdaq100_data(start, end, interval, source = 'Yahoo'):
    """
    import nasdaq100 OHLC data.
    Parameters
    ----------
    start : str
        start date.
    end : str
        end date.
    interval : str

    Returns
    -------
    data : pd.DataFrame
    """
    nasdaq = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
    nasdaq = nasdaq[4]['Ticker'].values
    return get_data(nasdaq, start, end, interval, source )

def get_nasdaq_data(start, end, interval, source = 'Yahoo'):
    """
    import all nasdaq OHLC data.
    Parameters
    ----------
    start : str
        start date.
    end : str
        end date.
    interval : str

    Returns
    -------
    data : pd.DataFrame

    Source
    ------
    https://www.nasdaq.com/market-activity/stocks/screener
    """
    print("last update for symbol names: 11/02/2023")
    nasdaq = pd.read_csv('nasdaq_11022023.csv')
    nasdaq = nasdaq['Symbol'].values
    nasdaq = nasdaq[[type(i)==str for i in nasdaq]]
    return get_data(nasdaq, start, end, interval, source )

def get_nyse_data(start, end, interval, source = 'Yahoo'):
    """
    import all NYSE OHLC data.
    Parameters
    ----------
    start : str
        start date.
    end : str
        end date.
    interval : str

    Returns
    -------
    data : pd.DataFrame

    Source
    ------
    https://www.nasdaq.com/market-activity/stocks/screener
    """
    print("last update for symbol names: 11/02/2023")
    nyse = pd.read_csv('nyse_11022023.csv')
    nyse = nyse['Symbol'].values
    nyse = nyse[[type(i)==str for i in nyse]]
    return get_data(nyse, start, end, interval, source )