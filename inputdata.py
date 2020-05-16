import numpy as np

""" считывание данных с командной строки """
def read_dir():
# выбор директории
    while(True):
        print("select data directory:\n",
              "1 - DB_AMN\n",
              "2 - DB_CANCERF4\n",
              "3 - DB_GLASS")
        in_dir = input()
        if (in_dir.startswith("1")):
            dir_name = "Data\\DB_AMN\\"
            break
        if (in_dir.startswith("2")):
            dir_name = "Data\\DB_CANCERF4\\"
            break
        if (in_dir.startswith("3")):
            dir_name = "Data\\DB_GLASS\\"
            break
        else:
            print("Incorrect input (expected 1, ...)")
    return dir_name

def read_file():
#выбор файла
    while(True):
        print("select file:\n",
              "1 - matrk1m3.txt\n",
              "2 - matrk2m1.txt\n",
              "3 - matrk2m2.txt\n",
              "4 - matrk2m3.txt\n",
              "5 - matrk3m1.txt\n",
              "6 - matrk3m2.txt\n",
              "7 - matrk3m3.txt")
        in_file = input()
        if (in_file.startswith("1")):
            file_name = "matrk1m3.txt"
            break
        elif (in_file.startswith("2")):
            file_name = "matrk2m1.txt"
            break
        elif (in_file.startswith("3")):
            file_name = "matrk2m2.txt"
            break
        elif (in_file.startswith("4")):
            file_name = "matrk2m3.txt"
            break
        elif (in_file.startswith("5")):
            file_name = "matrk3m1.txt"
            break
        elif (in_file.startswith("6")):
            file_name = "matrk3m2.txt"
            break
        elif (in_file.startswith("7")):
            file_name = "matrk3m3.txt"
            break
        else:
            print("Incorrect input (expected 1, ...)")
    return file_name

def read_PCA():            
#выбор числа главных компонент
    while(True):
        print("select dimension of feature space:\n",
              "1 - PCA (1 principal component)\n",
              "2 - PCA (2 principal components)\n",
              "3 - PCA (3 principal components)\n",
              "4 - all features")
        in_metr = input()
        if(in_metr.startswith("1")):
            dim = 1
            break
        elif(in_metr.startswith("2")):
            dim = 2
            break
        elif(in_metr.startswith("3")):
            dim = 3
            break
        elif(in_metr.startswith("4")):
            dim= -1
            break
        else:
            dim = -1
            print("continue whith 'Euclidean metric'")
            break
    return dim

def read_clustAlg():
#выбор алгоритма кластеризации
    print("select cluster analysis algorithm:\n",
          "1 - k-means\n",
          "2 - EM\n",
          "3 - DBSCAN\n",
          "4 - None")
    in_metr = input()
    
    if(in_metr.startswith("1")):
        n_comp = int( input("   number of clusters: ") )
        return 1, n_comp, -1
    
    elif(in_metr.startswith("2")):
        n_comp = int( input("   number of clusters: ") )
        return 2, n_comp, -1
    
    elif(in_metr.startswith("3")):
        eps = int( input("   eps (for ex. 2): ") )
        min_samples = int( input("   min_samples (for ex. 2): ") )
        return 3, eps, min_samples
    elif(in_metr.startswith("4")):
        return -1, -1, -1
    else:
        print("continue with 'None'")
        return -1, -1, -1

""" интерпритация данных """

""" 
Из файла name извлекает количество строк и столбцов матрицы признаков
"""
def get_propeties(name, file_name):
    with open(name, "r") as f:
        file_info = f.readlines()
        
    descriptor_len = int ( file_name[5] ) 
    marker_n = int( file_name[7] ) 
    
    i = x_len = y_len = 0
    while (file_info[i] != "\n"):
        pnt = file_info[i].find("matrk")
        if (pnt != -1):
            same_descriptor = ( int (file_info[i][pnt + len("matrk")] ) == descriptor_len)
            same_marker = (int  (file_info[i][pnt + len("matrk") + 2] ) == marker_n)
            if (same_descriptor & same_marker):
                pnt = file_info[i].find("(")
                str = file_info[i][pnt+1 :]
                str = str.strip(",)\n").split(", ")
                y_len = int(str[0])
                x_len = int(str[1])
        i += 1
    return x_len, y_len

""" 
Считывает матрицу признаков 
"""
def get_matr(name,x_len, y_len):
    with open(name, "r") as f:
        file_info = f.readlines()
    
    X = np.zeros( (y_len, x_len) )
    for y in range (y_len):
        for x in range (x_len):
            X[y][x] = float( file_info[y].split()[x] )
            
    return X

""" 
Считывает матрицу свойств для каталога DB_AMN 
"""
def get_propities_DB_AMN (name, y_len):
    with open(name, "r") as f:
        file_info = f.readlines()
        
    i = 0
    while (file_info[i] != "\n"):
        i += 1
        
    Y = np.zeros( (y_len) )
    i += 1
    for j in range(y_len):
        Y[j] = float( file_info[i+j].split()[1] )
    return Y
               
def get_propeties_DB_CANCERF4_A1 (name, y_len):
    with open(name, "r") as f:
        Y = np.zeros( (y_len) )
        j = 0
        not_stop = True
        
        for line in f:
            if line != "\n" and not_stop: 
                continue
            elif not_stop:
                not_stop = False
                continue
        
            A1_line = line.split()[1].split(',')
            if (len(A1_line) == 2):
                Y[j] = int(A1_line[0]) + int(A1_line[1]) / (10**(len(A1_line[1]) ))
            elif (len(A1_line) == 1):
                Y[j] = int(A1_line[0])   
            j += 1 
    return Y

def get_propeties_DB_GLASS (name, y_len):
    with open(name, "r") as f:
        Y = np.zeros( (y_len) )
        j = 0
        not_stop = True
        
        for line in f:
            if line != "\n" and not_stop: 
                continue
            elif not_stop:
                not_stop = False
                continue
            
            #print(line.split())
            Y[j] = int( line.split()[3] )
            j += 1
    return Y
        
def get_metric():
    print("select cmetric:\n",
          "1 - Euclidean\n",
          "2 - Minkowski\n",
          "3 - Chebyshev\n")
    in_metr = input()
    
    p = -1
    if in_metr.startswith('1'):
        alg_id = 1
    elif in_metr.startswith('2'):
        alg_id = 2
        p = float(input("p = "))
    elif in_metr.startswith('3'):
        alg_id = 3
    else:
        print("continue with '1'")
        alg_id = 1
    return alg_id, p
        
        