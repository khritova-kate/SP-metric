import alg

# get input data (information about model and data)
dir_name = inputdata.read_dir()
file_name = inputdata.read_file()
dim = inputdata.read_PCA()

#get data
x_len, y_len = inputdata.get_propeties(dir_name + "propeties.txt", file_name)
X = inputdata.get_matr(dir_name + file_name, x_len, y_len)
if ("DB_AMN" in dir_name):
    Y = inputdata.get_propities_DB_AMN(dir_name + "propeties.txt", y_len)
elif ("DB_CANCERF4" in dir_name):
    Y = inputdata.get_propeties_DB_CANCERF4_A1(dir_name + "propeties.txt", y_len)
elif ("DB_GLASS" in dir_name):
    Y = inputdata.get_propeties_DB_GLASS(dir_name + "propeties.txt", y_len)
    
#build model with ODR
if dim>0:
    #scale data and build a PCA projection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  
    pca = PCA(n_components = dim)
    PrincipalComp = pca.fit_transform(X_scaled)
    
    #plot all data
    enter = input("plot result (y/n): ")
    if enter.startswith('y'):
        klustmatr.plot_PC(dim, PrincipalComp)
    
    alg.user_models(PrincipalComp, Y, dim)
            
#build model without ODR            
else:
    alg.user_models(X,Y,dim)
                
#input("enter smth to finish : ")
