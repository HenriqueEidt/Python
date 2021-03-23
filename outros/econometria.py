import pandas_datareader as wb
import pandas as pd
import numpy as np
from scipy import stats
import scipy as sy
from scipy.stats import linregress

df_ibov = wb.DataReader('^BVSP','yahoo','01/02/2019',end='12/31/2019')
df_itsa3 = wb.DataReader('ITSA3.SA','yahoo','12/28/2018',end='12/31/2019')


df_reg = pd.DataFrame()
df_reg['IBOV'] = df_ibov['Close']
df_reg['ITSA3'] = df_itsa3['Close']

df_reg['txIBOV'] = np.log(df_reg['IBOV']/df_reg['IBOV'].shift())
df_reg['txITSA3'] = np.log(df_reg['ITSA3']/df_reg['ITSA3'].shift())

df_reg.to_csv('reg.csv')



#Faz a regressão linear de duas variáveis, calculando β1 e β2 baseado em um csv contendo x e y 
def regression(nome,column_sep,decimal_sep,alpha,Xname,Yname,skipRows):


    df = pd.read_csv(nome,sep=column_sep,decimal=decimal_sep,skiprows=[skipRows],header=0)

    print(df)

    X_list = df[Xname].to_list()
    Y_list = df[Yname].to_list()


    Xmean = sum(X_list)/len(X_list)
    Ymean = sum(Y_list)/len(Y_list)

    x = []
    y = []
    X_times_X = []

    for i in range(len(X_list)):
        x.append(X_list[i]-Xmean)
        y.append(Y_list[i]-Ymean)
        X_times_X.append(X_list[i]*X_list[i])

    x_times_y = []
    x_times_x = []
    y_times_y = []

    for i in range(len(x)):
        x_times_x.append(x[i]*x[i])
        x_times_y.append(x[i]*y[i])
        y_times_y.append(y[i]*y[i])


    sumx_squared = sum(x_times_x)
    sumX_squared = sum(X_times_X)
    sumxy = sum(x_times_y)
    sumy_squared = sum(y_times_y)


    beta_2 = sumxy/sumx_squared
    beta_1 = Ymean - beta_2*Xmean
    
    SSr = (beta_2**2) * sumx_squared                       #Regression sum of squares SQE
    rss = sumy_squared - (beta_2**2)*sumx_squared          #residual sum of squares SQR
    RdgF = len(df[Yname])-2                                  #residual degrees of freedom n-k
    mrss = rss/RdgF                                        #Average residual sum of squares


    varβ2 = mrss/sumx_squared
    varβ1 = (sumX_squared/len(df[Yname]))*varβ2

    t_calc_β2 = beta_2/varβ2**0.5
    t_calc_β1 = beta_1/varβ1**0.5
    
    t_critic = stats.t.ppf(1-alpha/2, len(df[Xname])-2)   #Função sem descrição, retorna um valor da tabela t, para um teste t bicaudal é necessário dividir o a por 2 
    
    print('------------------')
    print('β2 =', beta_2)
    print('β1 =', beta_1)
    print('------------------')
    print('varβ2 =',varβ2)
    print('varβ1=', varβ1)
    print('------------------')
    print('epβ2 =',varβ2**0.5)
    print('epβ1 =',varβ1**0.5)
    print('------------------')
    print('r² =',SSr/sumy_squared,'ou seja, {}% das variações em Y são explicadas por variações em X'.format((SSr/sumy_squared)*100))
    print('t calculado para β1=',t_calc_β1)
    print('t calculado para β2=',t_calc_β2)

    #Teste t 
    if t_calc_β1 > t_critic or t_calc_β1 < t_critic*(-1):
        print('Para β1: Rejeita-se H0, 0 não é um dos possíveis valores para β1')
    else:
        print('Para β1: Aceita-se H0, 0 é um dos possíveis valores para β1')
    
    if t_calc_β2 > t_critic or t_calc_β2 < t_critic*(-1):
        print('Para β2: Rejeita-se H0, 0 não é um dos possíveis valores para β2')
    else:
        print('Para β2: Aceita-se H0, 0 é um dos possíveis valores para β2')
    print('---------------------------------------')

regression('reg.csv',column_sep=',',decimal_sep='.',alpha=0.5,Xname='txIBOV',Yname='txITSA3',skipRows=1)






#Testando a regressão pelo linregress
x = df_reg['txIBOV'].tolist()
x = x[2:]

y = df_reg['txITSA3'].tolist()
y = y[2:]

regressão = linregress(x,y)
print(regressão)


