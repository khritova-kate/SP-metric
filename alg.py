import inputdata
import klustmatr

import numpy as np
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dir_name = inputdata.read_dir()
file_name = inputdata.read_file()
dim = inputdata.read_PCA()

x_len, y_len = inputdata.get_propeties(dir_name + "propeties.txt", file_name)
X = inputdata.get_matr(dir_name + file_name, x_len, y_len)

if ("DB_AMN" in dir_name):
    Y = inputdata.get_propities_DB_AMN(dir_name + "propeties.txt", y_len)
elif ("DB_CANCERF4" in dir_name):
    Y = inputdata.get_propeties_DB_CANCERF4_A1(dir_name + "propeties.txt", y_len)
elif ("DB_GLASS" in dir_name):
    Y = inputdata.get_propeties_DB_GLASS(dir_name + "propeties.txt", y_len)
    

if dim>0:
    pca = PCA(n_components = dim)
    PrincipalComp = pca.fit_transform(X)
    
    klustmatr.plot_PC(dim, PrincipalComp)
    
    enter = input("cluster analysis algorithm (y/n): ")
    while enter.startswith('y'):
        alg_id, p1, p2 = inputdata.read_clustAlg(dim, PrincipalComp)
    
        if alg_id == 1:
            y_alg = klustmatr.plot_kmeans(dim,PrincipalComp,p1)
            print("K-Means:")
            klustmatr.cluster_metric_DB_AMN(dim, X, Y, y_alg, y_len)
            klustmatr.plot_clustProfile(X,y_alg)
            
        
        if alg_id == 2:
            y_alg = klustmatr.plot_EM(dim,PrincipalComp,4)
            print("EM:")
            klustmatr.cluster_metric_DB_AMN(dim, X, Y, y_alg, y_len)
            klustmatr.plot_clustProfile(X,y_alg)
            
        if alg_id == 3:
            y_alg = klustmatr.plot_DBSCAN(dim,PrincipalComp,2,2)
            print("DBSCAN:")
            klustmatr.cluster_metric_DB_AMN(dim, X, Y, y_alg, y_len)
            klustmatr.plot_clustProfile(X,y_alg)
                
        enter = input("cluster analysis algorithm (y/n): ")
        if not(enter.startswith('y')) and not(enter.startswith('n')):
            print("continue with 'n'")

input("enter smth to finish : ")





        