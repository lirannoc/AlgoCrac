
import numpy as np
import pandas as pd
from functools import partial


def moving_average(data, k = 30):
    if len(data.shape)==1:
        data = np.array(data).reshape(1,-1)
    # if k is greater than or equal to length of data, return data
    if len(data[0])<=k:
        k = len(data[0])
        return data
    kernel = np.ones((k))/k
    convolve = partial(np.convolve,mode = 'valid')
    convolve_v = np.vectorize(convolve, signature='(n),(m)->(k)')
    ma = convolve_v(data, kernel) 
    ma = np.concatenate([data[:,:k-1],ma],axis = 1)
    return ma


def exp_moving_average(data, k = 30, smoother = 2):
    if len(data.shape)==1:
        data = np.array(data).reshape(1,-1)
    a = smoother/(k+1)
    if len(data[0])<k:
        k = len(data[0])
    ema =  data.copy()
    for i in np.arange(1,len(data[0])):
        ema[:,i] = data[:,i]*a + ema[:,i-1]*(1-a)
    return ema


def vwap(close, high, low, volume, k = None):
    # can only be allowed with len of 2 and above
    # k is default to full length
    # change arguments to 2D
    if len(close.shape)==1:
        close = np.array(close).reshape(1,-1)
    if len(high.shape)==1:
        high = np.array(high).reshape(1,-1)
    if len(low.shape)==1:
        low = np.array(low).reshape(1,-1)
    if len(volume.shape)==1:
        volume = np.array(volume).reshape(1,-1)
    if close.shape!= high.shape or close.shape!= low.shape:
        raise Exception("Data shapes do not match")
    t_price = (close + high + low)/3

    if k is None or k>len(close[0]):
        k = len(close[0])
    kernel = np.ones([k])

    convolve = partial(np.convolve,mode = 'full')
    convolve_v = np.vectorize(convolve, signature='(n),(m)->(k)')
    a = np.multiply(t_price,volume)
    a = convolve_v(a, kernel)[:,:-(k-1)]
    b = convolve_v(volume, kernel)[:,:-(k-1)]

    vwap = np.divide(a,b)
    return vwap
