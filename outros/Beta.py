import pandas as pd
import pandas_datareader as web
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = web.DataReader(['EVEN3.sa','^BVSP'],'yahoo',start = '03/11/2016',end='03/11/2021')
df = df['Adj Close']

df = df.resample('M').mean()

df['even3_returns'] = np.log(df['EVEN3.sa']/df['EVEN3.sa'].shift())
df['bvsp_returns'] = np.log(df['^BVSP']/df['^BVSP'].shift())






