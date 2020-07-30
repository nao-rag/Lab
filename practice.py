# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:30:36 2019

@author: 直己
"""
"""
単変量のときのピアソン残差、deviance　完了
To do list
*nによるグラフ
*多変量
**他のいろんな尺度
*AUC
*likelihood base
*estimated m_pp
"""
import collections
import pandas as pd
import numpy as np
import numpy.linalg as LA
from scipy import stats
import matplotlib.pyplot  as plt
import math
from scipy import optimize
from sklearn.datasets import load_iris

def main():
    n=1000
    df = data(n)
    
    initial_para = np.random.rand(len(df.values[1]))#初期パラメータ  
    mle = estimate_para(df, initial_para) #最尤推定の実行
    
    df_cat = categorize(mle, df)#共変量パターンによるデータフレーム(カテゴリ化),もとのデータフレームから作成
    print(df_cat)
    print("ピアソンカイ二乗統計量",chi_square(df_cat))#カテゴリ化データフレームからカイ二乗統計量
    print("逸脱度",deviance(df_cat))#逸脱度
    
    df_grp = grouping(10, df_cat)#g個のグルーピング,カテゴリ化データフレームから作成
    print(df_grp)
    print("HL統計量",Hosmer_Lemeshow(df_grp))#自由度g-2のカイ二乗分布
    
    #ROC(df_cat)#ROC曲線の図示
    print("AUC",AUC(df_cat))#AUCの値
    '''
    est_prob = np.zeros(n)
    for i in range(n):
        est_prob[i] = prob(df.x1[i], mle)
    
    print(est_prob)
    draw_split_line(mle, 'black', 'solid')#mleの図示
    
    
    scatter(df)#散布図
    
    print('mle',mle)
    print('mleでの尤度', likelihood(mle, df))
    print('mle標準誤差', std_error(mle, df))
    '''
def TP_TN(c, df):
    y_hat = np.zeros(len(df))
    table =np.zeros((2,2))
    for i in range(len(df)):
        if df.pi_j[i] > c:
            y_hat[i] = 1
        else:
            y_hat[i] = 0
            
        if df.y_j[i] == 1:
            if y_hat[i] == 1:
                table[0,0] = table[0,0] + 1
            else:
                table[1,0] = table[1,0] + 1
        if df.y_j[i] == 0:
            if y_hat[i] == 1:
                table[0,1] = table[0,1] + 1
            else:
                table[1,1] = table[1,1] + 1
       
    #df_table = pd.DataFrame(table,index=['y_est=1','y_est=0'],columns=['y_true=1','y_true=0'])
    
    sens_spec = [table[0,0] / (table[0,0] + table[1,0]),table[1,1] / (table[1,1] + table[0,1])]
    return sens_spec

def AUC(df):
    
    df = pd.DataFrame(df.values,columns = ['x1', 'm_j', 'y_j', 'pi_j'])
    c = collections.Counter(df.y_j)
    den = c[0] * c[1]
    num = 0
    for i in range(len(df)):
        if df.y_j[i] == 1:
            for j in range(i):
                if df.y_j[j] == 0:
                    num = num + 1
    auc = num / den
    return auc

def ROC(df):
    x = np.zeros(1000)
    y = np.zeros(1000)
    c = 0.0
    for i in range(len(x)):
        x[i] = 1 - TP_TN(c, df)[1]
        y[i] = TP_TN(c, df)[0]
        c = c + 1 / len(x)
        
    plt.title("ROC")
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.plot(x,y)
    
def data(n):
    '''データセットの用意
    '''
    x1 = np.random.binomial(1,0.5,n)#ベルヌーイ分布
    x2 = np.random.poisson(3,n) #ポアソン分布
    x3 = np.random.binomial(1,0.7,n)#ベルヌーイ分布
    x4 = np.random.normal(0,3,n)#正規分布
    #X = np.array([x0,x1,x2,x3])#共変量の行列
    #z =np.dot(para,X)#線形予測子
    y = np.zeros(n)
    y_1 = np.zeros(n)
    
    np.random.seed(3)
    
    i = 0
    for i in range(n):
        if x4[i] > 0:
            y_1[i] = np.random.binomial(1,0.6,1)
        else:
            y_1[i] = np.random.binomial(1,0.1,1)
    
    y = np.random.binomial(1,0.6,n)
    
    df_1 = pd.DataFrame({'x1': x4, 'y': y_1})
    df_2 = pd.DataFrame({'x1': x2, 'y': y})
    df_data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})
    
    return df_1

def scatter(df):
    '''
    i = 0
    for x1, y in zip(df.x1, df.y):
        if i == 5:
            plt.scatter(x1,y,c = 'r',marker = 'o')
        else:
            plt.scatter(x1,y,c = 'black',marker = 'o')
        i = i + 1
    '''
    plt.scatter(df.x1,df.y,c = 'black',marker = 'o')
    plt.xlabel('x')
    plt.ylabel('y')
def deviance_residual(y, m, prod):
    res = 0
    if y == 0:
        res = math.sqrt(2 *((m -y) *np.log((m -y) /(m *(1 -prod)))))
    elif y == m:
        res = math.sqrt(2 *(y *np.log(y /(m *prod))))
    else:
        res = math.sqrt(2 *(y *np.log(y /(m *prod)) +(m -y) *np.log((m -y) /(m *(1 -prod)))))
    
    if (y -(m * prod)) < 0:
        res = -res
    
    return res
def deviance(*args):
    df = args[0]
    stat = 0
    for i in range(len(df)):
        stat = stat + math.pow(deviance_residual(df.y_j[i], df.m_j[i], df.pi_j[i]),2)
    
    return stat

def pearson_residual(y, m, prob):
    res = (y - m * prob)/math.sqrt(m * prob * (1 - prob))
    
    return res
    
def chi_square(*args):
    df = args[0]
    stat = 0
    for i in range(len(df)):
        stat = stat + math.pow(pearson_residual(df.y_j[i], df.m_j[i], df.pi_j[i]),2)
    
    return stat
    
def grouping(g, df):
    mem = len(df) / g
    data = np.zeros((g, len(df.values[0])))
    n = 0
    for i in range(len(df)):
        data[n] = data[n] + df.values[i]
        if i >= mem * (n + 1) - 1:
            n = n + 1
    data[:,3] = data[:,3] / mem
    df_grouping = pd.DataFrame(data,columns = ['x1','m_k','y_k','pi_k'])
    
    return df_grouping

def categorize(para, *args):
    df = args[0]
    c = collections.Counter(df.x1)
    values, counts = zip(*c.most_common())
    x1 = values
    mem = counts
    y_count = np.zeros(len(c), dtype = int)
    est_p =np.zeros(len(c))
    for i in range(len(c)):
        est_p[i] = prob(x1[i], para)
        for j in range(len(df)):
            if df.values[j,0] == x1[i]:
                y_count[i] = y_count[i] + df.values[:,1][j]   
        
    categorize_df = pd.DataFrame({'x1': x1, 'm_j': mem, 'y_j': tuple(y_count.tolist()), 'pi_j': est_p})
    cat_df = categorize_df.reindex(columns=['x1', 'm_j', 'y_j', 'pi_j'])
    return cat_df.sort_values(by = 'pi_j')

def Hosmer_Lemeshow_residual(x, n, pi):
    res = math.pow((x -(n *pi)), 2)/(n *pi *(1- pi))
    return res

def Hosmer_Lemeshow(*args):
    df = args[0]
    stat = 0
    for i in range(len(df)):
        stat = stat + Hosmer_Lemeshow_residual(df.y_k[i], df.m_k[i], df.pi_k[i])
    
    return stat
    
def sigmoid(z):
    '''内積に対して、シグモイド関数の値を返す
    '''
    return 1.0 / (1.0 + np.exp(-z))

def prob(x1, para):
    '''P(Y=1|X)を返す
    '''
    feature_vector =  np.array([x1, 1.0])
    z = np.dot(feature_vector, para)

    return sigmoid(z)

def limited_prob(x1, para):
    '''P(Y=1|X)を返す,シグモイド関数の値に制限
    '''
    sigmoid_range = 39.14394658089878#log((1 - 10^-17)/10^-17))
    feature_vector =  np.array([x1, 1.0])
    z = np.dot(feature_vector, para)
    
    if z <= -sigmoid_range:
        return 1e-15
    if z >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-z))

def likeli(para,*args):
    '''対数尤度の和を返す
    '''
    sum = 0
    df_data = args[0]
    for x1,y in zip(df_data.x1, df_data.y):
        p=prob(x1, para)
        if y == 1:
            sum = sum - np.log(p)
        else:
            sum = sum - np.log(1-p)
    
    return sum            

def likelihood(para,*args):
    '''尤度を返す
    '''
    sum = 0
    df_data = args[0]
    for x1,y in zip(df_data.x1, df_data.y):
        p=prob(x1, para)
        if y == 1:
            sum = sum + np.log(p)
        else:
            sum = sum + np.log(1-p)
    
    return np.exp(sum)
 
def estimate_para(df_data,initial_para):
    '''学習用のデータとパラメータの初期値を受け取って、
    最尤推定の結果の最適パラメータを返す関数
    '''        
    parameter = optimize.minimize(likeli, initial_para, 
                                  args=(df_data), method='Nelder-Mead')
    
    return parameter.x

def infor_matrix(para,*args): 
    '''データフレームとパラメータを受け取って、情報行列を返す関数
    '''
    df_data = args[0]
    
    diag_array = np.zeros(len(df_data))
    i = 0
    for x1, y in zip(df_data.x1, df_data.y):
        p = limited_prob(x1, para)
        diag_array[i] = p * (1.0 - p)
        i = i + 1
    
    df = args[0]
    Xdf = df.iloc[:,[0]]
    Xold = Xdf.as_matrix()#共変量行列(x1,x2)
    array_ones = np.reshape(np.ones(Xdf.shape[0]),(Xdf.shape[0],1))
    X = np.concatenate((array_ones,Xold), axis = 1)#(1,x1,x2)を要素
    diag_matrix = np.diag(diag_array)#対角成分がp_i(1-p_i)となる行列
    infor_matrix = np.dot(np.dot(X.T, diag_matrix), X)#情報行列
    
    return infor_matrix

def std_error(para,*args):
    '''推定値の標準誤差を返す関数(通常ver)
    '''
    infor_max = infor_matrix(para,*args)
    var_matrix = inverse_matrix(infor_max)
    std_error = np.diag(var_matrix)
    std =np.zeros(len(std_error))
    for i in range(len(std_error)):
        std[i]=math.sqrt(std_error[i])
    
    return std

def inverse_matrix(X):
    '''逆行列を返す
    '''
    inv_X = np.linalg.inv(X)
    
    return inv_X

def determinant(X):
    '''行列式を返す
    '''
    det_X = LA.det(X)
    
    return det_X

def draw_split_line(para,color,ls):
    '''分離線を描画する関数
    '''
    a,b = para
    x = np.linspace(-10, 10, 1000)

    y =1.0/(1.0 + np.exp(-(b + a * x))) 
    plt.plot(x, y, color=color,ls=ls, alpha=0.5)    

if __name__ == "__main__":
    main()