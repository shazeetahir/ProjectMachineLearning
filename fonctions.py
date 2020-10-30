import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import request
import json



#le dataset
# train=pd.read_csv('heart_failure.csv')
# train.rename(columns=lambda x: x.replace('DEATH_EVENT', 'Mort'), inplace=True)
train=pd.read_csv('application_record.csv')
train.rename(columns=lambda x: x.replace('CODE_GENDER', 'Sex'), inplace=True)
train.rename(columns=lambda x: x.replace('FLAG_OWN_CAR', 'Own a car'), inplace=True)
train.rename(columns=lambda x: x.replace('NAME_FAMILY_STATUS', 'Family Status'), inplace=True)
train.rename(columns=lambda x: x.replace('NAME_INCOME_TYPE', 'Source of income'), inplace=True)


# train["age"] = train["age"]
# bins = [39, 50, 65, 70, np.inf]
# labels = ['39-50', '50-65', '65-70', '70-95']
# train['AgeGroup'] = pd.cut(train["age"], bins, labels = labels)
# train['AgeGroup']=train['AgeGroup'].map({'39-50':1, '50-65':2, '65-70':3, '70-95':4})

#Male is 0 female is 1
train['Sex']=train['Sex'].astype(str).map({'M':'0','F':'1'})
train['Sex']

#Own a car is 0 no is 1
train['Own a car']=train['Own a car'].astype(str).map({'Y':'0','N':'1'})
train['Own a car']

#Changing the family status to 0,1,2,3
#Civil marriage or married will be married
train['Family Status']=train['Family Status'].astype(str).map({'Civil marriage':'0',"Married":'0',"Single / not married":'1',"Separated":'2',"Divorced":'3'})
train['Family Status']

#Changing the source of income to 0,1,2,3
train['Source of income']=train['Source of income'].astype(str).map({'Working':'0',"Commercial associate":'1',"Pensioner":'2'})
train['Source of income']



def prediction(param):

    param=np.array(param).reshape(1,-1) 
    
    #col_1=float(request.form.get(param1, False))
    #col_2=float(request.form.get(param2, False))
    #a=[]
    #cls=pickle.load(open("cls_heart_attack.pkl", "rb"))
    #return a"""

    # cls=pickle.load(open("cls_heart_attack.pkl", "rb"))
    cls=pickle.load(open("cls_accounts.pkl", "rb"))

    #return json.dumps({'user':cls.predict(np.array(a).reshape(1,-1))})
    return (cls.predict(param))


  
def entrainement():

    # predictors = train.drop(['Mort'], axis=1)
    # target = train["Mort"]
    predictors = train.drop(['Sex'], axis=1)
    target = train["Sex"]

    # x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
    # cls=RandomForestClassifier(max_depth=12,n_estimators=300).fit(x_train,y_train)

    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
    cls=RandomForestClassifier(max_depth=12,n_estimators=300).fit(x_train,y_train)
    # cls.score(x_val,y_val)

    #sauver cls
    filename = 'cls_accounts.pkl'
    pickle.dump(cls, open(filename, 'wb'))

    return(cls.score(x_val,y_val))




