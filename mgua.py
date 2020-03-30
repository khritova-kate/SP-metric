#Варвара Васильева, 16.03.2020
#https://github.com/VarvaraVasilyeva/QSAR/blob/master/MGUA.ipynb


import numpy as np
import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import math
from sklearn.metrics import r2_score

def corr(x, y):
    up = sum(x*y)
    down = math.sqrt(sum(x*x)*sum(y*y))
    return up/down

class MGUA:
    def __init__(self, Q, C, I, model):
        self.Q = Q #размер буфера
        self.C = C #порог корреляции
        self.I = I #количество итераций
        self.model = model #model = LinearRegression(normalize=True)
        self.EPS = 1e-14
    def fit(self, X, y_train):
        N = X.shape[0]
        M = X.shape[1]
        buf_val = 0
        for i in range(M):
            for j in range(i+1, M):
                X_train = X.iloc[:, [i, j]]
                self.model.fit(X_train, y_train)
                pred = self.model.predict(X_train) #Попробовать тут Х?
                if buf_val == 0:
#                     print(type(pred))
                    buf = [pred] #buf[0] = [pred]???
                    buf_coef = [[[i, j]]]
                    buf_val += 1
                else:
                    buf_corr = [corr(col, pred.reshape(-1, )) for col in buf[0].T]
                    if buf_val<self.Q and max(buf_corr)-self.C < self.EPS:
                        buf[0] = np.c_[buf[0], pred]
                        buf_coef[0].append([i, j])
                        buf_val += 1
                    elif buf_val>=self.Q and max(buf_corr)<self.C:
                        buf_r2 = [r2_score(y_train, buf[0][:, col]) for col in range(self.Q)]
                        if r2_score(y_train, pred) > min(buf_r2):
                            buf[0] = np.delete(buf[0], buf_r2.index(min(buf_r2)), axis = 1)
                            del buf_coef[0][buf_r2.index(min(buf_r2))]
                            buf[0] = np.c_[buf[0], pred]
                            buf_coef[0].append([i, j])

        for k in range(1, self.I):
            #print("iter = ", k)
            buf_val = 0
            X_train = X.iloc[:, [0]]
            X_train = X_train.assign(new = buf[k-1][:, 0])
            self.model.fit(X_train, y_train)
            pred = self.model.predict(X_train) 
            
            buf.append(pred)
            buf_coef.append([[0, 0]])
            buf_val += 1
            for i in range(M):
#                 print('i = ', i)
                for j in range(1, buf[k-1].shape[1]):
                    X_train = X.iloc[:, [i]]
                    X_train = X_train.assign(new = buf[k-1][:, j])
                    self.model.fit(X_train, y_train)
                    pred = self.model.predict(X_train) 

                    buf_corr = [corr(col, pred.reshape(-1, )) for col in buf[k].T]
#                     if buf_val>=self.Q:
#                         print('buf_val=', buf_val, 'max(buf_corr)=', max(buf_corr), max(buf_corr)<self.C)
                    if buf_val<self.Q and max(buf_corr)-self.C < self.EPS:
#                         print('1. (i, j) = ', i, j)
                        buf[k] = np.c_[buf[k], pred]
                        buf_coef[k].append([i, j])
                        buf_val += 1
                    elif buf_val>=self.Q and max(buf_corr)<self.C:
                        buf_r2 = [r2_score(y_train, buf[k][:, col]) for col in range(self.Q)]
                        if r2_score(y_train, pred) > min(buf_r2):
#                             print('(i, j) = ', i, j)
                            buf[k] = np.delete(buf[k], buf_r2.index(min(buf_r2)), axis = 1)
                            del buf_coef[k][buf_r2.index(min(buf_r2))]
                            buf[k] = np.c_[buf[k], pred]
                            buf_coef[k].append([i, j])
        self.buf_coef = buf_coef
        self.buf = buf
        self.X_train = X
        self.y_train = y_train
        return buf_coef
    
    def predict(self, X):
        result = []
        index = []
        for i in range(len(self.buf_coef[-1])): #может быть последний буфер не полностью заполнен, тогда меньше...
            ind_pred = i
            index.append([])
            for k in reversed(range(self.I)):
                index[i].append(self.buf_coef[k][ind_pred])
                ind_pred = self.buf_coef[k][ind_pred][1]
            X_t = self.X_train.iloc[:, index[i][-1]]
            self.model.fit(X_t, self.y_train)
            pred = self.model.predict(X.iloc[:, index[i][-1]])
            for k in range(1, self.I):
                X_t = self.X_train.iloc[:, [index[i][self.I-1-k][0]]]
                X_t = X_t.assign(new = self.buf[k-1][:, index[i][self.I-1-k][1]])
                self.model.fit(X_t, self.y_train)
                X_pred = X.iloc[:, [index[i][self.I-1-k][0]]]
                X_pred = X_pred.assign(new = pred)
                pred = self.model.predict(X_pred)
            result.append(pred)
            
        return result #Q столбцов - Q предсказаний
