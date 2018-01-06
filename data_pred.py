# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:13:06 2017

@author: denya-rog
"""


import re
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


def  tax_sys(train, li):
    """Change stings from pandas Serias to its numer in list of unique strings"""
    for i in range(len (li)):
        train[train == li[i]] = i
    
    return train

            
def how_old(data) :
    """ Take pandas Series.
    For making from continioudsly datafor making prediction more easy 
    and for avoiding overfitting
    
    Make numerical parametrs from date .
            0- less then year
            1- less then 2 years
            2- less then 5 years
            3- less then 10 years
            4- more then 10 years 
            None- if there is no date or it has illegal format"""


    out = data.fillna("01.01.0001") 
    now = dt.datetime.now()
    
    for i in range(len(data)):
        
        try:           
            date = dt.datetime.strptime(str(out[i]), "%Y-%m-%d" )
            
        except ValueError:            
            try:
                date = dt.datetime.strptime(str(out[i]), "%d.%m.%Y" )
                
            except ValueError:
                 date = dt.datetime(1,1,1)   
            
        difff = now - date
        dif = difff.days//365
        
        if dif < 1:
            out[i] = 0
            
        elif dif < 2 :
            out[i] = 1
            
        elif dif < 5 :
            out[i] = 2 
            
        elif dif < 10 :
            out[i] = 3
            
        elif dif > 10 : 
            out[i] = 4
            
        else:
            out[i] = None
               
    return out


def estimate(norm_data,target):
    """Finding best predictor on train set. 
    Takes args: train set 2d-array like, target list-like.
    Returns classifier with best bapams with best accuracy"""
    
    # making train /test split from train set for choosing best classificator
    RAND_STATE = 42         #for some random
    X_train,X_test,y_train,y_test = train_test_split(norm_data,target,test_size=0.4, random_state=RAND_STATE)
    
#    lists will contain best params and esimators with them
    list_estim = []
    list_best_param = []
    
#    making many classificators
    print("finding best classificator: ...")    
    
    #KNN
    
    neighbours={}
    ALG={0:'ball_tree',1:'kd_tree'}
    
    for  i in range(10,40,5):
        for j in range(2):
            
            neigh = KNeighborsClassifier(n_neighbors=3, leaf_size=i,algorithm =ALG[j])
            neigh.fit(X_train, y_train)
            neighbours[str(i)+" alg="+str(j)] = neigh.score(X_test,y_test)
            
    list_best_param,list_estim = fil_best(neighbours, list_best_param,list_estim)        


    print ("knn checked ...")
    
    #logistic Regression
    
    log_regr={}
    
    for c in  range(2,10,2):
        for t  in [(0.1)**i for i in range(6)]:
            
            log = LogisticRegression(C=c, tol=t)
            log.fit(X_train, y_train)
            log_regr[str(c)+" Tol="+str(t)] = log.score(X_test,y_test)  
                   
    list_best_param,list_estim = fil_best(log_regr, list_best_param,list_estim)

    
    print ("logistic regression checked ...")
    
    #linear Regression
    
    svc={}
    
    for c in  range(2,10,2):
        for t  in [(0.1)**i for i in range(6)]:
            
            log = LinearSVC(C=c, tol=t)
            log.fit(X_train, y_train)
            svc[str(c)+" Tol="+str(t)] = log.score(X_test,y_test)  
                   
    list_best_param,list_estim = fil_best(svc, list_best_param,list_estim)

    
    print ("linear regression checked ...")
    
    #Ridge Classifier
    
    ridge={}
    
    for a in  [(0.1)**i for i in range(-6,1,1)]:
        for t  in [(0.1)**i for i in range(6)]:
            
            log = RidgeClassifier(alpha=a, tol=t)
            log.fit(X_train, y_train)
            ridge[str(a)+" Tol="+str(t)] = log.score(X_test,y_test) 
                    
    list_best_param,list_estim = fil_best(ridge, list_best_param,list_estim)

    
    print ("ridge cassifier checked ...")
    
    # Decision Tree
    
    tree={}
    dept=[i for i in range(1,31,5)]
    samp_spl=[i for i in range(2,11)]
    
    for a in  dept:
        for t  in samp_spl:
            
            log = DecisionTreeClassifier(max_depth=a, min_samples_split=int(t))
            log.fit(X_train, y_train)
            tree[str(a)+" Tol="+str(t)] = log.score(X_test,y_test)                         
    
    list_best_param,list_estim = fil_best(tree, list_best_param,list_estim)

    
    print ("decision tree checked ...")
    
    #Random Forest
    
    forest = {}
    dept = [i for i in range(1,32,5)]
    samp_spl = [i for i in range(2,11)]
    
    for a in  dept:
        for t  in samp_spl:
            
            log = RandomForestClassifier(max_depth=a, min_samples_split=int(t))
            log.fit(X_train, y_train)
            forest[str(a)+" Tol="+str(t)] = log.score(X_test,y_test)     
                
    list_best_param,list_estim = fil_best(forest, list_best_param,list_estim)

    
    print ("random forest  checked ...")
    #chosing best estimatot, and fitting it
    best_ind = list_best_param.index(max(list_best_param))
    
    return list_estim[best_ind]
              
   
def fil_best(dic, list_best_param,list_estim):
    
    """take dictionary with params of classifier, list with best parameters 
    and list with clasificators, returns list with new best parameters 
    and list with new clasificator """
    
    best_key = [key for key,val in dic.items() if val == max(dic.values())]
    best_value = [val for key,val in dic.items() if val == max(dic.values())]       
    
    list_best_param.append(best_value[0])    
    A = float(re.findall(r'(^.*)\s',best_key[0])[0])
    B =  float(re.findall(r'[=](.*)$',best_key[0])[0]  )    
    list_estim.append(RandomForestClassifier(max_depth=A, min_samples_split=int(B)))
    
    return list_best_param,list_estim
        
        
def data_prediction(name1 , name2):
    
    """main functon, take two names of files, preparing data,
    choosing best predictor, creates new file with predicted data """
    
    
    import os 
    name1 = os.getcwd() + name1 
    name1 = os.getcwd() + name1
    try:
        test = pd.read_csv(name1, delimiter= "\t", encoding= "cp1251")
    except:
         raise NameError('check  file or name of file for test')
         
    try:
        train = pd.read_csv(name2, delimiter= "\t",encoding= "cp1251")
    except:
         raise NameError('check  file or name of file for train')
 
    print ("load_success")
    
    #making numeric tax system
    
        #creating list with unique tax systems
    li1 = train["taxactionSystem"].unique() 
    li2 = test["taxactionSystem"].unique()
    li = list(set(li1).union(li2))
    
    #creating numerical taxsystem feature     
    
    train["tax_sys"] = tax_sys(train["taxactionSystem"],li)
    test["tax_sys"] = tax_sys(test["taxactionSystem"],li)
    
    print ("tax_changing_successful")
    
    # making new feature with numeric parametr with year of registerion and creation
    
    train["age_of_reg"] = how_old(train["regdt"])
    test["age_of_reg"] = how_old(test["regdt"]) 
    
    train["age_of_adding"] = how_old(train["OrgCreationDate"])
    test["age_of_adding"] = how_old(test["OrgCreationDate"]) 
    
    print("changing_years_success")
       
     #dropong useless columns
       
    train = train.drop(["regdt",u"OrgCreationDate", "taxactionSystem"],axis = 1)
    test = test.drop(["regdt","OrgCreationDate", "taxactionSystem"],axis = 1)
    
    print("drping_useless_success")
    
    #filling missing data with median
    
    for col in train:
        train[col] = train[col].fillna(int(round(train[col].median()))).astype(int)
    
    for col in test:
        test[col] = test[col].fillna(int(round(test[col].median()))).astype(int)

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
        if cor[i[0]].sum()>cor[i[1]].sum() and i[0]not in drop_list:
            drop_list.append(i[0])
        else:
            drop_list.append(i[1])
            
            
    print ("drop",drop_list ,"success")
    
    train = train.drop(drop_list, axis=1)
    test = test.drop(drop_list, axis=1)
    
    # data for training
    norm_data = train.drop("is_prolong", axis=1)
    
    print("features used for prediction:", norm_data.columns.values.tolist())
    
    est = estimate(norm_data,train["is_prolong"])

#    est = list_estim[best_ind]
    
    print("best estimaor :",est)
    
    
    est.fit(norm_data, train["is_prolong"])
    prediction = est.predict(test)
    out = prediction
    test["is_prolong"] = out
    test["id"] = id_test


    test.sort_values(by='id',inplace=True)
    test.to_csv("test_predict.csv", columns = ["id", "is_prolong"], encoding="utf-8",index=False)


if __name__=="__main__":
    if len(sys.argv) == 1:
        data_prediction("test.csv", "train.csv")
        
    elif len(sys.argv) == 3:
        data_prediction(sys.argv[1],sys.argv[2])
        
    else:
        print ("something wrong with input filenames")