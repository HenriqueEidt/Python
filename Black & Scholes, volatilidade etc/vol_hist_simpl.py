from numpy.lib.arraysetops import ediff1d
import pandas as pd
import numpy as np
import pandas_datareader as web





#Determinação da volatilidade histórica toma como base o adj close e leva em conta apenas os dias negociados. A vol anual considera 252 dias
#Nessa formulação a média é tomada como 0
#Considera-se M e não M-1, isso faz com que o estimador passe de não viesado para de maxima verosemelhança
#Leva-se em conta o retorno simples

def vol_hist_simpl(ativo,start,end):
    df = web.DataReader(ativo,'yahoo',start,end)
    df = df['Adj Close']
    df = df.reset_index()
    df = df.set_index('Date')
    df['simpl_return'] = df['Adj Close']/df['Adj Close'].shift()
    n=len(df['Adj Close'])
    


vol_hist_simpl('CYRE3.SA',start='02-24-2021',end='03-24-2021')

