# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:13:06 2017

@author: denya-rog
"""



import sys

import pandas as pd
import datetime as dt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def  tax_sys(data_seria, unic_list):
    """Change sting from pandas Serias to its numer in list of unique strings"""
    for i in range(len (unic_list)):
        data_seria[data_seria == unic_list[i]] = i
    
    return data_seria

            
def how_old(data) :
    """ Take pandas Series.
    For making from continioudsly datafor making prediction more easy 
    and for avoiding overfitting
    
    Make numerical parametrs from date:
            0 - less then year
            1 - less then 2 years
            2 - less then 5 years
            3 - less then 10 years
            4 - more then 10 years 
            None - if there is no date or it has illegal format
    """

    NUMBER_DAYS = 365
    out = data.fillna("01.01.0001") 
    now = dt.datetime.now()

  
    for i in range(len(data)):
       
        try:           
            date = dt.datetime.strptime(str(out[i]), "%Y-%m-%d" )
            
        except ValueError: 
            
            try:
                date = dt.datetime.strptime(str(out[i]), "%d.%m.%Y" )
                
            except ValueError:
                #if item has illegal format
                 date = dt.datetime(1,1,1)  
                 
          
        difference = now - date
        
        dif_years = difference.days // NUMBER_DAYS
        
        if dif_years < 1:
            out[i] = 0
            
        elif dif_years < 2 :
            out[i] = 1
            
        elif dif_years < 5 :
            out[i] = 2 
            
        elif dif_years < 10 :
            out[i] = 3
            
        elif dif_years > 10 : 
            out[i] = 4
            
        else:
            out[i] = None
               
    return out


def find_best_estimator(norm_data,target):
    """Finding best predictor on train set. 
    Takes args: train set 2d-array like, target list-like.
    Returns classifier with best bapams with best accuracy"""
    
    RAND_STATE = 42    
    
    X_train,X_test,y_train,y_test = train_test_split(norm_data,target, \
                                        test_size=0.4, random_state=RAND_STATE)
     
    best_estimtor = []
    best_est_skore = []
    params = []             
    list_est = []
    
#    knn
    list_est.append( KNeighborsClassifier(n_neighbors=3,) )    
    ALG = ['ball_tree','kd_tree']  
    leafes = [int(i) for i in range(10,40,5)]
    params.append(dict(leaf_size=leafes, algorithm=ALG))
    
#    logistic regr
    list_est.append( LogisticRegression() )    
    c =[i for i in range(2,10)] 
    t = [(0.1)**i for i in range(6)]
    params.append(dict(C=c, tol=t))
        
#    linear regression
    list_est.append( LinearSVC() )    
    c =[i for i in range(2,10)] 
    t = [(0.1)**i for i in range(6)]
    params.append(dict(C=c, tol=t))
             
#    ridge clasfier
    list_est.append( RidgeClassifier() )    
    alp =[(0.1)**i for i in range(-6,1,1)] 
    t = [(0.1)**i for i in range(6)]
    params.append(dict(alpha=alp, tol=t))
    
#    decidion tree
    list_est.append( DecisionTreeClassifier() ) 
    dept=[i for i in range(1,31,5)]
    samp_spl=[i for i in range(2,11)]
    params.append(dict(max_depth=dept, min_samples_split=samp_spl))
   
#    random forest
    list_est.append( RandomForestClassifier() ) 
    dept=[i for i in range(1,31,5)]
    samp_spl=[i for i in range(2,11)]
    params.append(dict(max_depth=dept, min_samples_split=samp_spl))
    
   

    for i in range(len (params)):   
            
        est = GridSearchCV(estimator=list_est[i], param_grid=params[i], n_jobs=-1)
        est.fit(X_train, y_train)
    
        best_estimtor.append(est.best_estimator_)
        best_est_skore.append(est.best_score_)
        
    
    best_ind = best_est_skore.index(max(best_est_skore))
    print (best_est_skore)
    return best_estimtor[best_ind]
        
        
def data_prediction(name1 , name2):
    
    """main functon, take two names of files, preparing data,
    choosing best predictor, creates new file with predicted data """
    
    
    #do i need try exept???
    
    test = pd.read_csv(name1, delimiter= "\t", encoding= "cp1251")
    train = pd.read_csv(name2, delimiter= "\t",encoding= "cp1251")
    
 
    print ("load_success")
    
    """"making numeric tax system, because predictor can work with
     only nummerical data. find all unique  taxs sustems, and give them category.
    """
    unic_tax_sys_train = train["taxactionSystem"].unique() 
    unic_tax_sys_test = test["taxactionSystem"].unique()
    print(unic_tax_sys_train,unic_tax_sys_test)
    
    unic_tax_sys = list(set(unic_tax_sys_test).union(unic_tax_sys_train)) 
#    see, that there is not so many tax systems, we can cetegorise them dirrect
                                           
    
    train["num_tax_sys"] = tax_sys(train["taxactionSystem"], unic_tax_sys)
    test["num_tax_sys"] = tax_sys(test["taxactionSystem"], unic_tax_sys)
    
    print ("tax_changing_successful")
    
    # new features with numeric parametr of year of registration and creation
    """for getting not so much informatin,  that is useles, 
    from continusly data date make kategorical.  It will make our life easier,
    and save us from mistakes of overtraining.  Also it will speedup programm.
    """

    train["age_of_reg"] = how_old(train["regdt"])
    test["age_of_reg"] = how_old(test["regdt"]) 
    
    train["age_of_adding"] = how_old(train["OrgCreationDate"])
    test["age_of_adding"] = how_old(test["OrgCreationDate"]) 
    
    print("changing_years_success")
       
     #dropong useless columns
       
    train = train.drop(["regdt","OrgCreationDate", "taxactionSystem"], axis = 1)
    test = test.drop(["regdt","OrgCreationDate", "taxactionSystem"], axis = 1)
     
    print("drping_useless_success")
    
    #filling missing data with median
    
    for col in train:
        mediana = int(round(train[col].median()))
        train[col] = train[col].fillna(mediana).astype(int)
    
    for col in test:
        mediana = int(round(test[col].median()))
        test[col] = test[col].fillna(mediana).astype(int)

    print("fillna_with_median_success")
   
   #finding corelation among  features
            
    KOEF_CORR =0.7  #max value of koeficien of corelation
    
    cor = train.astype(float).corr().abs()
    indices = np.where(cor > KOEF_CORR)   
    indices = [(cor.index[x], cor.columns[y]) for x, y in zip(*indices) if x != y and x < y]
    
 
    # id containers
    id_train = train["id"]
    id_test = test["id"]
    
    drop_list=["id"] 
        
    for i in indices:
        
        if cor[i[0]].sum() > cor[i[1]].sum() and i[0] not in drop_list:
            drop_list.append(i[0])
            
        else:
            drop_list.append(i[1])
            
            
    print ("drop",drop_list ,"success")
    
    train = train.drop(drop_list, axis=1)
    test = test.drop(drop_list, axis=1)
    
    # data for training
    norm_data = train.drop("is_prolong", axis=1)
    
    print("features used for prediction:", norm_data.columns.values.tolist())
    
    est = find_best_estimator(norm_data,train["is_prolong"])


    print("best estimaor :",est)
    
    
    est.fit(norm_data, train["is_prolong"])
    print (est.score(norm_data, train["is_prolong"]))
    
#    prediction = est.predict(test)
#    
#    out = prediction
    test["is_prolong"] = est.predict(test)
    test["id"] = id_test


    test.sort_values(by='id',inplace=True)
    test.to_csv("test_predict.csv", columns = ["id", "is_prolong"], encoding="utf-8",index=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_prediction('test.csv', 'train.csv')
        
    elif len(sys.argv) == 3:
        data_prediction(sys.argv[1],sys.argv[2])
        
    else:
        print ("something wrong with input filenames")
