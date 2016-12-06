# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 04:18:41 2016

@author: Sergey
"""

import numpy as np
import random
import copy
from pandas import read_csv
import sklearn.linear_model


def fun1(t): 
    z = np.exp(-t)
    return np.matrix([z[0,i] / (z[0,i] + 1.0) for i in range(z.shape[1])])
    
def normalize(X):
    
    for i in range(1, X.shape[0]):
        X[i,:] = (X[i, :] - X[i,:].mean())/X[i,:].var()
        
    return X
        
def download_data(N = 100, reg = 'train' , index = [i for i in range(891)]):
    
    data = read_csv('train.csv')

    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    dicts = {}
    
    label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
    dicts['Sex'] = list(label.classes_)
    data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов 
    
    
    label.fit(data.Embarked.drop_duplicates())
    dicts['Embarked'] = list(label.classes_)
    data.Embarked = label.transform(data.Embarked)
    
    
    data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    
    data.Age[data.Age.isnull()] = data.Age.mean()
    data.Fare[data.Fare.isnull()] = data.Fare.median() #заполняем пустые значения средней ценой билета
    
    #data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
    
    data = np.matrix(data)
    
    data = data[index, :]

    if (reg == 'train'):
        data = data[:N,:]
    elif (reg == 'test'):
        data = data[N:,:]
    else:
        raise Exception('Invalid dataset!')
        
    A = np.matrix([[1 for i in range(data.shape[1])] for j in range(data.shape[0])])
    data = np.matrix(data)
    
    A[:, 1:] = copy.copy(data[:, 1:])
    y = copy.copy(data[:, 0].T)
    
    for i in range(y.shape[1]):
        if (y[0, i] == 0.0):
            y[0, i] = -1.0 
            
    X = A.T
    X = normalize(X)
    
    return   X, y


def log_reg_gradient(X, y, epsilon = 0.1, num_iterations = 1, regularization = 0, random = 1, reg_const = 0.25, ordre = 1):
    
    if (random == 1):
        w =  np.random.rand(X.shape[0])
    else:
        w = np.ones(X.shape[0])
        
    w = np.matrix(w)
    
    if (ordre == 1):

        for j in range(num_iterations):
            
            grad = (fun1(w * X) - y) * X.T
            
            if (regularization == 0):    
                w =  w - epsilon * grad
            else:
                w = w - epsilon * (grad + w * reg_const)
            print np.linalg.norm(w)
            if np.linalg.norm(grad, 2) < 0.0001:
                break
    
    return w
    
def predict(W, X, y):
    
    z = [0 for i in range(X.shape[1])]
    
    for i in range(X.shape[1]):
        if (W * X[:,i] <= 0.0):
            z[i] = - 1.0
        else:
            z[i] = 1.0
            
    return np.matrix(z) 
    
def test(X, y, w):
    z = predict(w, X, y)  
    a = np.linalg.norm(z - y)
    return a * a / 4.0         

def plots(Size, regularize = 1, min_grad = 1.0, ind = [i for i in range(891)]):

    L = [1500]#[20, 100, 200, 300]+ [500 * i for i in xrange(1,5)]
    
    M_test = [0 for i in xrange(len(L))]
    M_train = [0 for i in xrange(len(L))]
    
    X_train, y_train = download_data(N = Size, reg = 'train') 
    X_test, y_test = download_data(N = Size, reg = 'test')
    
    size_test = X_test.shape[1]
    size_train = X_train.shape[1]
    
    for i in range(len(L)):
        print('n = ' + str(L[i]))
        
        if (regularize == 0):
            w = log_reg_gradient(X_train, y_train, epsilon = 1e-3, num_iterations = L[i], regularization = 0, reg_const = 5, ordre = 1)
        else:
            w = log_reg_gradient(X_train, y_train, epsilon = 1e-3, num_iterations = L[i], regularization = 1, reg_const = 5, ordre = 1)
            
        M_test[i] = test(X_test, y_test, w) / size_test
        M_train[i] = test(X_train, y_train, w) / size_train
     
    return (M_train, M_test)
    
    
    
def np_regression(T, ind = [i for i in range(891)]):
       
    
    X_train, y_train = download_data(reg = 'train', N = T, index = ind)
    X_test, y_test = download_data(reg = 'test', N = T, index = ind)
    
    size_train = X_train.shape[1]
    size_test = X_test.shape[1]
    
    #print(y_test)
    L = [1500]#, 50, 100, 200] + [500 * i for i in xrange(1,4)]
    M_test = [0 for i in xrange(len(L))]
    M_train = [0 for i in xrange(len(L))]
    
    for i in range(len(L)):
        m = sklearn.linear_model.LogisticRegression(max_iter = L[i], random_state = 1)
        m.fit(X_train[1:,:].T, np.array(y_train.T))
        z = m.predict(X_test[1:,:].T)
        a = np.linalg.norm(z - y_test)
        #print(z - y_test)
        M_test[i] = a * a / 4.0 / size_test#m.score(X_test[1:,:].T, np.array(y_test.T))
        z = m.predict(X_train[1:,:].T)
        a = np.linalg.norm(z - y_train)
        M_train[i] = a * a / 4.0 / size_train#m.score(X_train[1:,:].T, np.array(y_train.T))#= a * a / 4.0 / size_train

    return (M_train, M_test)
    
    
    
def compare():
    
    L = [i * 100 for i in range(1,8)]
    in_built_res_train = [0 for i in range(len(L))]
    in_built_res_test = [0 for i in range(len(L))]
    my_res_train = [0 for i in range(len(L))]
    my_res_test = [0 for i in range(len(L))]
    
    for i in xrange(len(L)):
        
        indices = np.random.permutation(891)
        plotts = plots(Size = L[i], ind = indices)
        nump = np_regression(L[i], ind = indices)
        
        in_built_res_test[i] =  nump[1][0]
        
        my_res_test[i] = plotts[1][0]
        
        in_built_res_train[i] =  nump[0][0]
        
        my_res_train[i] = plotts[0][0]
        
    import matplotlib.pyplot as plt    
    line1 = plt.plot(L, in_built_res_train, 's', linestyle = '--', color = 'r', label = 'Numpy_Train');
    line2 = plt.plot(L, in_built_res_test, '*', linestyle = '-', color = 'r', label = 'Numpy_Test');
    
    line3 = plt.plot(L, my_res_train, 's', linestyle = '--', color = 'b', label = 'My_Train');
    line4 = plt.plot(L, my_res_test, '*', linestyle = '-', color = 'b', label = 'My_Test');
    
    plt.xlabel('TrainSetSize')    
    plt.legend(loc='down center')
    plt.savefig('compare.png')