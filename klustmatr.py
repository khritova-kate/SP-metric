import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#кластерный анализ
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mgua

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
        
    # линейная регрессия + МГУА
    
    inp = input("   MGUA (y/n):  ")
    if inp.startswith('y'):
        for i in range(clust_n):
            print("cluster ", markers[i], " MGUA: ",
                  MGUA_cluster(X,Y, markers[i],y_alg,y_len))
        print("\n")
    elif not inp.startswith('n'):
        print("continue with 'n'")
     
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

def MGUA_cluster(X,Y, clust_mark, y_alg, y_len):
    X_cl, Y_cl = get_cluster_points(X,Y, clust_mark, y_alg, y_len) 
    if(len(Y_cl) > 5):
        Mgua = mgua.MGUA(4, 0.997, 5, LinearRegression(normalize=True))
        
        train_size = int (0.7 * len(Y_cl))
        X_train = pd.DataFrame(X_cl[:train_size])
        X_test  = pd.DataFrame(X_cl[train_size:])
        Y_train = pd.DataFrame(Y_cl[:train_size])
        Y_test  = pd.DataFrame(Y_cl[train_size:])
        
        Mgua.fit(X_train, Y_train)
        res  = Mgua.predict(X_test)
        
        best_res = -1
        for i in range(len(res)):
            if r2_score(Y_test, res[i]) > best_res:
                best_res = r2_score(Y_test, res[i])
        
        return best_res
    else:
        return "failure (too few points)"

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
    else:
        print("exepted dim = 1,2,3")

""" 
Изображает проекции на dim главных осей после применения k-means для k =  n_clusters 
"""        
def plot_kmeans (dim, PrincipalComp, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(PrincipalComp)
    y_kmeans = kmeans.predict(PrincipalComp)
    centers = kmeans.cluster_centers_
    
    if (dim == 1):
        fig, ax = plt.subplots()
        ax.set_title("KMeans-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,0]*0, c = y_kmeans)
        ax.scatter(centers[:, 0], centers[:, 0]*0, c = 'black', s = 200, alpha = 0.5);
        plt.show()
        return y_kmeans
    elif (dim == 2):
        fig, ax = plt.subplots()
        ax.set_title("KMeans-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], c = y_kmeans)
        ax.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5);
        plt.show()
        return y_kmeans
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("KMeans-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], PrincipalComp[:, 2], c = y_kmeans)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c = 'black', s = 200, alpha = 0.5)
        plt.show()
        return y_kmeans
    else:
         print("expected dim = 1,2,3")
         return None
        
def plot_EM (dim, PrincipalComp, n_clusters):
    gmm = GaussianMixture(n_components = n_clusters, covariance_type='full')
    gmm.fit(PrincipalComp)
    y_gmm = gmm.predict(PrincipalComp)
    
    if (dim == 1):
        fig, ax = plt.subplots()
        ax.set_title("EM-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,0]*0, c = y_gmm)
        plt.show()
        return y_gmm
    elif (dim == 2):
        fig, ax = plt.subplots()
        ax.set_title("EM-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], c = y_gmm)
        plt.show()
        return y_gmm
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("EM-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], PrincipalComp[:, 2], c = y_gmm)
        plt.show()
        return y_gmm
    else:
         print("expected dim = 1,2,3")
         return None
        
def plot_DBSCAN (dim, PrincipalComp, eps, min_samples):
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    dbscan.fit(PrincipalComp)
    y_dbscan = dbscan.labels_
    
    if (dim == 1):
        fig, ax = plt.subplots()
        ax.set_title("DBSCAN-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,0]*0, c = y_dbscan)
        plt.show()
        return y_dbscan
    elif (dim == 2):
        fig, ax = plt.subplots()
        ax.set_title("DBSCAN-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], c = y_dbscan)
        plt.show()
        return y_dbscan
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("DBSCAN-algorithm")
        ax.scatter(PrincipalComp[:,0], PrincipalComp[:,1], PrincipalComp[:, 2], c = y_dbscan)
        plt.show()
        return y_dbscan
    else:
         print("expected dim = 1,2,3")
         return None
        

        
    
    
        
        