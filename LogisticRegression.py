# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 00:32:18 2016

@author: Sergey
"""
import numpy as np
import random
import copy

n = 10



def fun1(t):
    
    z = np.exp(t)
    #return np.matrix([z[0,i] / (z[0,i] + 1) for i in range(z.shape[1])])
    return z / (1 + z )
    
#Загрузка и обработка данных    

def download_data(reg = 'train'):
    from pandas import read_csv
    # reg = 'test'######
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
    
    data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
    
    data = np.matrix(data)
    
    if (reg == 'train'):
        data = data[:600,:]
    elif (reg == 'test'):
        data = data[600:,:]
    else:
        raise Exception('Invalid dataset!')
        
    A = np.matrix([[-1 for i in range(data.shape[1])] for j in range(data.shape[0])])
    data = np.matrix(data)
    
    A[:, 1:] = copy.copy(data[:, 1:])
    y = copy.copy(data[:, 0].T)
    
    for i in range(y.shape[1]):
        if (y[0, i] == 0.0):
            y[0, i] = -1.0 
            
    X = A.T
    
    return X, y
    
    
#1. Градиентный спуск


def log_reg_gradient(X, y, epsilon = 0.1, num_iterations = 1, regularization = 0, random = 1, reg_const = 0.25):
        
    if (random == 1):
        w = np.random.rand(X.shape[0])
    else:
        w = np.ones(X.shape[0])
        
    w = np.matrix(w)
    
    for j in range(num_iterations):
        if (j % 20 == 0):
            print(j)
        for i in range(X.shape[1]):
            
            t = w * X[:, i] * y[0, i]
            q =  X[:, i] * y[0, i]
            
            if (regularization == 0):
                w = w + epsilon * fun1(-t) * q.T
            else:
                w = w + epsilon * ( fun1(-t) * q.T + w * reg_const)
    """
    while(difference > epsilon):
        
        t = np.multiply( w * X, y)
        
        if (regularization == 0):
            grad = X * (np.multiply(fun1(t),y)).T 
        else:
            grad = X * (np.multiply(fun1(t),y)).T + reg_constant * w.T

        
        difference = np.linalg.norm(grad, 2)
        print(difference)
        w = w - grad.T * alpha
        
        """
    
    return w
    
def predict(W, X):
    
    z = [0 for i in range(X.shape[1])]
    
    for i in range(X.shape[1]):
        if (W * X[:,i] <= 0):
            z[i] = -1.0
        else:
            z[i] = 1.0
            
    return np.matrix(z)
    
    
def test(X, y, w):
    z = predict(w, X)  
    a = np.linalg.norm(z - y)
    return a * a / 4
    
#X,y = download_data()
#w = log_reg_gradient(X, y, epsilon = 0.01, num_iterations = 300, regularization = 1, reg_const = 0.001)  

#test(X_test, y_test, w)

    
def plots(regularize = 0):
    """
    import matplotlib.pyplot as plt
    line1 = plt.plot(Numbers, G_MLE_test, 's', linestyle = '--', color = 'b', label = 'G_MLE'); 
    line2 = plt.plot(Numbers, L_MLE_label_test, '*', linestyle = '-', label = 'L_MLE'); 
    line3 = plt.plot(Numbers, KL_AVG_label_test, 's',linestyle = '-', color = 'g', label = 'KL_AVG' );
    line4 = plt.plot(Numbers, L_AVG_label_test , 's',linestyle = '-', color = 'r', label = 'L_AVG' );
    plt.xlabel('Test LL(label-wise partition)')
    plt.xscale('log')
    plt.legend(loc='down left')
    plt.savefig('label_test.png')
    
    """
    L = [20, 50, 100, 200] + [500 * i for i in xrange(1,7)]
    
    M_test = [0 for i in xrange(len(L))]
    M_train = [0 for i in xrange(len(L))]
    
    X_train, y_train = download_data(reg = 'train')
    X_test, y_test = download_data(reg = 'test')
    
    
    for i in range(len(L)):
        print('n = ' + str(L[i]))
        
        if (regularize == 0):
            w = log_reg_gradient(X, y, epsilon = 0.01, num_iterations = L[i], regularization = 0, reg_const = 0.001)
        else:
            w = log_reg_gradient(X, y, epsilon = 0.01, num_iterations = L[i], regularization = 1, reg_const = 0.001)
            
        M_test[i] = test(X_test, y_test, w)
        M_train[i] = test(X_train, y_train, w)
    
    import matplotlib.pyplot as plt
    
    if (regularize == 0):
        line1 = plt.plot(L, M_test, 's', linestyle = '-', color = 'b', label = 'Test_non-reg');
        line2 = plt.plot(L, M_train, '*', linestyle = '-', color = 'g', label = 'Train_non-reg');
        plt.xlabel('Without Regularization')
        plt.legend(loc='down left')
        plt.savefig('non-reg.png')
    else:
        line1 = plt.plot(L, M_test, 's', linestyle = '-', color = 'b', label = 'Test_reg');
        line2 = plt.plot(L, M_train, '*', linestyle = '-', color = 'g', label = 'Train_reg');
        plt.xlabel('With Regularization')
        plt.legend(loc='down left')
        plt.savefig('reg.png')
    