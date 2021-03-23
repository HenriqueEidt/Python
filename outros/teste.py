import numpy as np
from numpy.core.defchararray import count, mod
from pandas.core.frame import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import statistics

#REG COM SOBESPECIFICAÇÃO

B1 = []
B2 = []
for i in range(0,10000):

    u = np.random.normal(loc=0,scale=1,size = 100)
    X2 = (np.random.normal(loc=0,scale=1,size = 100) + 10)
    X3 = (np.random.normal(loc=0,scale=1,size = 100) + 20)

    Y = 10 + 10*X2 + 10*X3 + u

    X2 = sm.add_constant(X2)

    mod = sm.OLS(endog = Y,exog = X2)
    res = mod.fit()

    B1.append(res.params[0])
    B2.append(res.params[1])


fig = px.histogram(B1)
fig.show()

fig = px.histogram(B2)
fig.show()


print('--------------------REG MULTIPLA COM SOBESPECIFICAÇÃO---------------')
print('B1')
print(statistics.mean(B1))
print(statistics.stdev(B1))
print(min(B1))
print(max(B1))

print('B2')
print(statistics.mean(B2))
print(statistics.stdev(B2))
print(min(B2))
print(max(B2))
print('---------------------------------------------------------------------------------------')