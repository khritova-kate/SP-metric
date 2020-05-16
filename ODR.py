from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.linear_model    import Ridge
from sklearn.linear_model    import Lasso

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import r2_score
from scipy.spatial.distance import minkowski, chebyshev

import klustmatr
import numpy as np
    
def LinearModel_collection (X, Y):
    X_train, X_test, y_train, y_test = train_test_split( \
                      X, Y, test_size=0.25, random_state = 42)
    
    model = LinearRegression().fit(X_train, y_train);
    print("LinearRegression:")
    print("Correctness on training set: {:.2f}".format(model.score(X_train, y_train)))
    print("Correctness on testing  set: {:.2f}\n".format(model.score(X_test, y_test)))
    
    res = 0
    res_train = 0
    opt_alfa = 0.001
    for alfa in {0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100}:
        model = Ridge(alpha = alfa).fit(X_train, y_train);
        current_res = model.score(X_test, y_test) 
        if current_res > res:
            opt_alfa = alfa
            res = current_res
            res_train = model.score(X_train, y_train)
    
    print("Ridge (optimal alfa ", opt_alfa, "):")
    print("Correctness on training set: {:.2f}".format(res_train))
    print("Correctness on testing  set: {:.2f}\n".format(res))
    

def ridge(X_train, X_test, y_train, y_test):
    res = 0
    first = True
    for alfa in {0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100}:
        model = Ridge(alpha = alfa).fit(X_train, y_train);
        if (len(y_test) == 1):
           current_res = 1 - (y_test[0] - model.predict(X_test)[0])**2 / (4*y_test[0])
        else:        
            current_res = model.score(X_test, y_test)
    
    if current_res > res or first:
                res = current_res
                prediction =  model.predict(X_test)
                first = False
    
    return res, prediction    
   
def ridge_model_cluster(n_clusters, y_alg_train, y_alg_test, X_train, X_test,\
                        Y_train, Y_test):
    Y_res     = list()
    Y_compare = list()
    markers = [None]*n_clusters
    n = 0
    
    for i in range(n_clusters):
        if (n < len(y_alg_test)):
            beg = 0
            while (klustmatr.in_markers(markers,i,y_alg_test[beg]) and beg < len(y_alg_test)-1):
                beg += 1
            markers[i] = y_alg_test[beg]
        
            X_clust_train, y_clust_train = klustmatr.get_cluster_points(\
                  X_train, Y_train, markers[i], y_alg_train, len(y_alg_train))
            X_clust_test,  y_clust_test  = klustmatr.get_cluster_points(\
                                X_test, Y_test, markers[i], y_alg_test, len(y_alg_test))
            
            if (len(y_clust_test) > 0 and len(y_clust_train) > 0):
                Y_compare += y_clust_test
                ridge_res, y_res = ridge(X_clust_train, X_clust_test,
                                               y_clust_train, y_clust_test)
                Y_res += y_res.tolist()
                n += len(y_res)
    
    if (len(Y_res) > 1):        
        return r2_score(Y_compare, Y_res)
    elif(len(Y_res) == 1):
        return 1 - (Y_compare[0] - Y_res[0])**2 / (4*Y_compare[0])
    else:
        return -10000


def DBSCAN_model(X_train, Y_train, X_test, Y_test, eps, min_samples, alg_id, p):
    #alg_id = 1 -> 'euclid'
    #alg_id = 2 -> 'minkowski', p = p
    #alg_id = 3 -> 'chebyshev'
    
    if(alg_id == 1):
        clust_alg = DBSCAN(eps = eps, min_samples = min_samples)
    if(alg_id == 2):
        clust_alg = DBSCAN(eps = eps, min_samples = min_samples, metric = 'minkowski', p = p)
    if(alg_id == 3):
        clust_alg = DBSCAN(eps = eps, min_samples = min_samples, metric = 'chebyshev')
                
    clusters = clust_alg.fit_predict( np.vstack( (X_train, X_test) ) )
    y_alg_train = clusters[:len(X_train)]
    y_alg_test  = clusters[len(X_train):]
    
    return ridge_model_cluster(len(set(clusters)), y_alg_train, y_alg_test,\
                                            X_train, X_test, Y_train, Y_test)


def kmeans_model(X_train, Y_train, X_test, Y_test, n_clusters):

    clust_alg = KMeans(n_clusters = n_clusters, random_state = 42)
    clust_alg.fit(X_train)
    y_alg_train = clust_alg.predict(X_train)
    y_alg_test  = clust_alg.predict(X_test)
    
    return ridge_model_cluster(len(set(y_alg_test)), y_alg_train, y_alg_test,\
                                            X_train, X_test, Y_train, Y_test)

def EM_model(X_train, Y_train, X_test, Y_test, n_clusters):
    
    clust_alg = GaussianMixture(n_components = n_clusters, covariance_type='full', random_state = 42)
    clust_alg.fit(X_train)
    y_alg_train = clust_alg.predict(X_train)
    y_alg_test  = clust_alg.predict(X_test)
    
    return ridge_model_cluster(len(set(y_alg_test)), y_alg_train, y_alg_test,\
                                            X_train, X_test, Y_train, Y_test)

        
        
        
    
    
    


    
        
    