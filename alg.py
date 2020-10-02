import inputdata
import klustmatr
import ODR

#import numpy as np
from sklearn.decomposition import PCA

#from sklearn.cluster import KMeans

#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def build_model(PrincipalComp, Y, X_train, X_test, y_train, y_test,dim):
    alg_id, p1, p2 = inputdata.read_clustAlg()

    #without ODR
    if alg_id == -1:
        print("PCA: ", dim, "components\ncluster analysis algorithm: None\nResult:\n")
        ODR.LinearModel_collection(PrincipalComp, Y)
            
    #ODR with K-Means algorithm
    if alg_id == 1:
        enter_1 = input("information about clusters (y/n)? ")
        if enter_1.startswith('y'):
            y_alg_train = klustmatr.plot_kmeans(dim,X_train, X_test,p1)
            print("K-Means:")
            print("train-clusters propeties:")
            klustmatr.cluster_metric_DB_AMN(dim, X_train, y_train, y_alg_train, len(y_train))
            klustmatr.plot_clustProfile(X_train,y_alg_train)
        elif not enter_1.startswith('n'):
            print("continue with 'n'\n")
            
        result = ODR.kmeans_model(X_train, y_train, X_test, y_test,p1)
        print("PCA: ", dim, "components\ncluster analysis algorithm: K-Means")
        print("Correctness on testing  set: {:.2f}\n".format(result))
        
    #ODR with EM algorithm
    if alg_id == 2:
        enter_1 = input("information about clusters (y/n)? ")
        if enter_1.startswith('y'):
            y_alg_train = klustmatr.plot_EM(dim,X_train, X_test,p1)
            print("EM:")
            print("train-clusters propeties:")
            klustmatr.cluster_metric_DB_AMN(dim, X_train, y_train, y_alg_train, len(y_train))
            klustmatr.plot_clustProfile(X_train,y_alg_train)
        elif not enter_1.startswith('n'):
            print("continue with 'n'\n")
            
        result = ODR.EM_model(X_train, y_train, X_test, y_test,p1)
        print("PCA: ", dim, "components\ncluster analysis algorithm: K-Means")
        print("Correctness on testing  set: {:.2f}\n".format(result))
        
    #ODR with DBSCAN algorithm
    if alg_id == 3:
        alg_id, p = inputdata.get_metric()
        
        enter_1 = input("information about clusters (y/n)? ")
        if enter_1.startswith('y'):
            y_alg_train = klustmatr.plot_DBSCAN(dim,X_train, X_test,p1,p2,alg_id,p)
            print("DBSCAN:")
            print("train-clusters propeties:")
            klustmatr.cluster_metric_DB_AMN(dim, X_train, y_train, y_alg_train, len(y_train))
            klustmatr.plot_clustProfile(X_train,y_alg_train)
        elif not enter_1.startswith('n'):
            print("continue with 'n'\n")
        
        result =  ODR.DBSCAN_model(X_train, y_train, X_test, y_test,p1, p2, alg_id,p)
        print("PCA: ", dim, "components\ncluster analysis algorithm: K-Means")
        print("Correctness on testing  set: {:.2f}\n".format(result))


def user_models(X,Y,dim):
    #divide data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split( \
                      X, Y, test_size=0.25, random_state = 42)
    
    enter = input("cluster analysis algorithm (y/n): ")
    if enter.startswith('n'):
        ODR.LinearModel_collection(X, Y)
        enter = input("another algorithm (y/n): ")
        
    if not(enter.startswith('y')) and not(enter.startswith('n')):
        print("continue with 'n'\n\n")
        ODR.LinearModel_collection(X, Y)
        enter = input("another algorithm (y/n): ")
    
    while enter.startswith('y'):
        build_model(X, Y, X_train, X_test, y_train, y_test,dim)
        
        enter = input("another algorithm (y/n): ")
        if not(enter.startswith('y')) and not(enter.startswith('n')):
            print("continue with 'n'\n")
            print("PCA: ", dim, "components\ncluster analysis algorithm: None\nResult:\n")
            ODR.LinearModel_collection(X, Y) 
    
        





        