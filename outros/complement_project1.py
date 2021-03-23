from operator import le
from numpy.core.defchararray import array
from numpy.core.fromnumeric import var
import pandas as pd
import math
from scipy import stats
import scipy
import numpy as np


#Faz a regressão linear de duas variáveis, calculando β1 e β2 
def regression(nome,column_sep,decimal_sep,alpha):
    df = pd.read_csv(nome,sep=column_sep,decimal=decimal_sep)
    
    X_list = df['X'].to_list()
    Y_list = df['Y'].to_list()

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
    RdgF = len(df['Y'])-2                                  #residual degrees of freedom n-k
    mrss = rss/RdgF                                        #Average residual sum of squares


    varβ2 = mrss/sumx_squared
    varβ1 = (sumX_squared/len(df['Y']))*varβ2

    t_calc_β2 = beta_2/varβ2**0.5
    t_calc_β1 = beta_1/varβ1**0.5
    
    t_critic = stats.t.ppf(1-alpha/2, len(df['X'])-2)   #Função sem descrição, retorna um valor da tabela t, para um teste t bicaudal é necessário dividir o a por 2 
    
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
    





#Regressão multipla, utilizando abordagem matricial
def multiple_regression(name,column_sep,decimal_sep):
    df = pd.read_csv(name,sep=column_sep,decimal=decimal_sep)

    variables_number = len(df.columns)

  

    array_row_1_list = [len(df.iloc[:,0])]
    array_row_2_list = [sum(df.iloc[:,1])]
    array_row_3_list = [sum(df.iloc[:,2])]
    array_row_4_list = [sum(df.iloc[:,3])]
    array_row_5_list = [sum(df.iloc[:,4])]
    array_row_6_list = [sum(df.iloc[:,5])]

    xty = [sum(df.iloc[:,0])]

    for i in range(1,variables_number):
        array_row_1_list.append(sum(df.iloc[:,i]))
        array_row_2_list.append(sum(df.iloc[:,1]*df.iloc[:,i]))
        array_row_3_list.append(sum(df.iloc[:,2]*df.iloc[:,i]))
        array_row_4_list.append(sum(df.iloc[:,3]*df.iloc[:,i]))
        array_row_5_list.append(sum(df.iloc[:,4]*df.iloc[:,i]))
        array_row_6_list.append(sum(df.iloc[:,5]*df.iloc[:,i]))
        xty.append(sum(df.iloc[:,0]*df.iloc[:,i]))

    array_row_1=np.array(array_row_1_list)
    array_row_2=np.array(array_row_2_list)
    array_row_3=np.array(array_row_3_list)
    array_row_4=np.array(array_row_4_list)
    array_row_5=np.array(array_row_5_list)
    array_row_6=np.array(array_row_6_list)


    print(xty)

    xtx = np.array([array_row_1,array_row_2,array_row_3,array_row_4,array_row_5,array_row_6])
    print(xtx)
    
    print('------------------------------------')

    xtx_inv = np.linalg.inv(xtx)
    print(xtx_inv)



multiple_regression(name='dados_mult.csv',column_sep=';',decimal_sep=',')

