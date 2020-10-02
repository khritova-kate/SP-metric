import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#кластерный анализ
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#import mgua

from sklearn.model_selection import train_test_split

def in_markers (markers, lenth, cl_mark):
    for i in range (lenth):
        if (markers[i] == cl_mark):
            return True
    return False

def clustnum (markers, cl_mark):
    i = 0
    while (markers[i] != cl_mark):
        i += 1
    return i

def cluster_metric_DB_AMN(dim, X, Y, y_alg, y_len):
    clust_n = len( set(y_alg) )
    markers = [None]*clust_n
    
    #общая характеристика вектора свойств
    # ищем среднее, максимкм и минимум для каждого кластера
    
    M       = [None]*clust_n
    MAX     = [None]*clust_n
    MIN     = [None]*clust_n 
    
    print("information obout cluster propeties:")
    for i in range (clust_n):
        nop = m = beg = 0
        
        while (in_markers(markers,i,y_alg[beg])):
            beg += 1
            
        markers[i] = y_alg[beg]
        MAX[i] = Y[beg]
        MIN[i] = Y[beg]
        
        for j in range (beg, y_len):
            if (y_alg[j] == markers[i]):
                m += Y[j]
                nop += 1
                if (MIN[i] > Y[j]):
                    MIN[i] = Y[j]
                if (MAX[i] < Y[j]):
                    MAX[i] = Y[j]
                
        M[i] = m/nop
        print("cluster ", markers[i], " : mean ", round( M[i], 3), 
              " , n of points ", nop,
              " , [", round( MIN[i], 3), " , ", round( MAX[i], 3), "]")
                
    err = [0]*clust_n
    for i in range (clust_n):
        for j in range (y_len):
            pnt = clustnum(markers, y_alg[j])
            err [pnt] += abs(Y[j] - M[pnt])
    print("Markers : ", markers)
    
    # ошибка - сумма модулей отклонения от среднего
    print("Errors  :  [", end = '')
    for i in range(clust_n-1):
        print(round(err[i], 3), end = ', ')
    print(round(err[clust_n - 1], 3), end = '')
    print(']\n')
          



"""
Возвращает матрицу признаков и вектор свойств для точек кластера
"""
def get_cluster_points(X, Y, klust_mark, y_alg, y_len):
    ret_X = list()
    ret_Y = list()
    for i in range (y_len):
        if (y_alg[i] == klust_mark):
            ret_X += [ X[i] ]
            ret_Y += [ Y[i] ]
    return ret_X, ret_Y

def plot_clustProfile(X, y_alg):
    X_len   = len(X)
    clust_n = len( set(y_alg) )
    markers = [None]*clust_n
    profile = [list()]*clust_n
    
    beg = 0
    for i in range(clust_n):
        while (in_markers(markers,i,y_alg[beg])):
            beg += 1    
        markers[i] = y_alg[beg]
        
        profile[i] = list()
        for k in range(X_len):
            if y_alg[k] == markers[i]:
                profile[i].append(1)
            else:
                profile[i].append(0)
    
    fig, axs = plt.subplots(1, clust_n)
    for i in range(clust_n):
        axs[i].bar( np.arange(X_len),  profile[i][:X_len], width = 1)
        axs[i].set_title("cluster " + str(i+1) )
    
    plt.show()

""" 
Изображает проекции на dim главных осей 
"""
def plot_PC (dim, PrincipalComp):
    if (dim == 1):
        plt.scatter(PrincipalComp[:,0], PrincipalComp[:,0]*0)
        plt.show()
    elif (dim == 2):
        plt.scatter(PrincipalComp[:,0], PrincipalComp[:, 1])
        plt.show()
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], PrincipalComp[:, 2])
        plt.show()
    else:
        print("exepted dim = 1,2,3")
        
        

""" 
Изображает проекции на dim главных осей после применения алгоритмы кластерного анализа
"""        
def plot_clusters(dim, label, train_pnt, test_pnt, y_alg_train, y_alg_test, centers = np.array(0)):
    if (dim == 1):
        fig, ax = plt.subplots()
        ax.set_title(label)
        ax.scatter(train_pnt[:,0], train_pnt[:,0]*0, c = y_alg_train, s = 25)
        ax.scatter(test_pnt[:,0], test_pnt[:,0]*0, c = y_alg_test, s = 100)
        if (centers.any()):
            ax.scatter(centers[:, 0], centers[:, 0]*0, c = 'black', s = 200, alpha = 0.5);
        plt.show()
        return y_alg_train
    elif (dim == 2):
        fig, ax = plt.subplots()
        ax.set_title(label)
        ax.scatter(train_pnt[:,0], train_pnt[:,1], c = y_alg_train, s = 25)
        ax.scatter(test_pnt[:,0], test_pnt[:,1], c = y_alg_test, s = 100)
        if (centers.any()):
            ax.scatter(centers[:, 0], centers[:, 0]*0, c = 'black', s = 200, alpha = 0.5);
        plt.show()
        return y_alg_train
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(label)
        ax.scatter(train_pnt[:,0], train_pnt[:,1], train_pnt[:,2], c = y_alg_train, s = 25)
        ax.scatter(test_pnt[:,0], test_pnt[:,1], test_pnt[:,2], c = y_alg_test, s = 100)
        if (centers.any()):
            ax.scatter(centers[:, 0], centers[:, 0]*0, c = 'black', s = 200, alpha = 0.5);
        plt.show()
        return y_alg_train
    else:
         print("expected dim = 1,2,3")
         return None
    
    
    
def plot_kmeans (dim, train_pnt, test_pnt, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(train_pnt)
    y_alg_train = kmeans.predict(train_pnt)
    y_alg_test  = kmeans.predict(test_pnt)
    #centers = kmeans.cluster_centers_
    
    return plot_clusters(dim, "KMeans-algorithm", train_pnt,\
                         test_pnt, y_alg_train, y_alg_test)#, centers)
    
    
        
def plot_EM (dim, train_pnt, test_pnt, n_clusters):
    clust_alg = GaussianMixture(n_components = n_clusters, covariance_type='full')
    clust_alg.fit(train_pnt)
    y_alg_train = clust_alg.predict(train_pnt)
    y_alg_test  = clust_alg.predict(test_pnt)
    
    return plot_clusters(dim, "EM-algorithm", train_pnt,\
                         test_pnt, y_alg_train, y_alg_test)
    

        
def plot_DBSCAN (dim, train_pnt, test_pnt, eps, min_samples, alg_id, p):
    if(alg_id == 1):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    if(alg_id == 2):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = 'minkowski', p = p)
    if(alg_id == 3):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = 'chebyshev')
    
    clusters = dbscan.fit_predict( np.vstack( (train_pnt, test_pnt) ) )
    y_alg_train = clusters[:len(train_pnt)]
    y_alg_test  = clusters[len(train_pnt):]
    
    return plot_clusters(dim, "DBSCAN-algorithm", train_pnt,\
                         test_pnt, y_alg_train, y_alg_test)
        

        
    
    
        
        