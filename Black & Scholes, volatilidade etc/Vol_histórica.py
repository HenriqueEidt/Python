from operator import le
import numpy as np
import pandas as pd
import pandas_datareader as web
import math

#Determinação da volatilidade histórica toma como base o adj close e leva em conta apenas os dias negociados. A vol anual considera 252 dias
def vol_const(stock,begin,end):
    df = web.DataReader(stock,'yahoo',begin,end)
    df = df['Adj Close']
    df = df.reset_index()
    df = df.set_index('Date')

    df['ln'] = np.log(df['Adj Close']/df['Adj Close'].shift())
    df = df.iloc[1:]

    n = len(df['Adj Close'])+1

    termo_1 = (1/(n-1))
    termo_1 = termo_1*sum(df['ln']**2)

    termo_2 = (1/(n*(n-1)))
    termo_2 = termo_2*(sum(df['ln']))**2

    S = math.sqrt(termo_1-termo_2)

    print(df)
    print('--------------------------------------------------')
    print('Estimativa da vol diária {}'.format(S))
    print('Estimativa da vol anual {}'.format(S*math.sqrt(252)))
    

vol_const('CYRE3.sa',begin='02-24-2021',end='03-24-2021')
