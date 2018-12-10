# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:39:38 2018

@author: zouco
"""

import pandas as pd 
import numpy as np
from pandas import Series,DataFrame


import os

def change_dir_to_DAT():
    path = os.getcwd()
    while(os.path.basename(path) != 'Data-Analysis-Tools'):
        path = os.path.dirname(path)
    os.chdir(path)
    
change_dir_to_DAT()

from pandas_extend import columns_with_na


os.chdir(os.getcwd() + '/Projects')

from k000_Titanic_0 import load_data
from k000_Titanic_0 import save_result

data_train, data_test = load_data()

DATA_PATH='\\datasets\\k000_titanic\\'



# ---------------------------------------------------------------------------

# data preparation
print(columns_with_na(data_train))
print(columns_with_na(data_test))
data_train.info()
data_test.info()


from sklearn.preprocessing import StandardScaler

def data_preprocessing(df, rfr):
    # df is the df
    # 1 PassengerID : x
    # 2 Plass       : get_dummies
    # 3 Name        : x
    # 4 Sex         : get_dummies
    # 5 Age         : use RF to predict it by ['Age','Fare', 'Parch', 'SibSp', 'Pclass']
    # 6 SibSp       : ---  (use it directly)
    # 7 Parch       : ---
    # 8 Ticket      : x
    # 9 Fare        : fill the NA with 0
    # 10 Cabin      : diffienciate the passengers by the one who has Cabin info and those who dont
    # 11 Embarked   : get_dummies
    
    '''
        So interesting preprocessing are ages, Fare, Cabin.
        age missed a lot.
        Fare missed one.
        with or withou cabin shows huge difference.
    '''
    
    # Age
    if np.sum(df.Age.isnull()) != 0:
        df = fill_null_age(df, rfr)
    
    # Fare
    df.Fare.fillna(0, inplace=True)
    # df.loc[df.Fare.isnull(), 'Fare' ] = 0
    
    # Cabin
    df.loc[df.Cabin.notnull(), 'Cabin' ] = 1
    df.loc[df.Cabin.isnull(), 'Cabin' ] = 0
    
    # get_dummies
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')
    
    df = pd.concat([df, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
    df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin|Embarked_.*|Sex_.*|Pclass_.*')
    
    # Scale the value : Age, Fare
    # here is huge problem, can you find it?
    
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df.loc[:,'Age'].values.reshape(-1,1))
    df['Fare'] = scaler.fit_transform(df.loc[:,'Fare'].values.reshape(-1,1))
    
    return df



# fill the age
def fill_null_age_rfr(df):
    columns = ['Age','Fare', 'Parch', 'SibSp', 'Pclass']
    known_age = df.loc[df.Age.notnull(), columns].values
    
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(known_age[:, 1:], known_age[:, 0])
    
    return rfr
    
def fill_null_age(df, rfr):
    columns = ['Age','Fare', 'Parch', 'SibSp', 'Pclass']
    unknown_age = df.loc[df.Age.isnull(), columns].values
    df.loc[ (df.Age.isnull()), 'Age' ] = rfr.predict(unknown_age[:, 1::])
    return df

rfr = fill_null_age_rfr(data_train)




'''
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(list(df.Sex.value_counts().index)).transform(df.Sex)
'''




# ------------------------------------------------------------------------
# prepare for fit
df = data_preprocessing(data_train, rfr)
x_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values  

# prepare for predict
df_test = data_preprocessing(data_test, rfr)
x_test = df_test.values # the form of input of the model, ready for model.predict(x_test)


# ---------------------------------------------------------------------------
# logistic model:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=2, penalty='l1', tol=1e-8)
lr.fit(x_train,y_train)


############################################################################
# evaluate the model
print(lr.score(x_train,y_train))

from sklearn.model_selection import cross_val_score
print(cross_val_score(lr, x_train, y_train, cv=5))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(lr.predict(x_train), y_train))

save_result('prediciton_lr', lr, x_test)

# see coefs:
pd.DataFrame({"columns":list(df.columns)[1:], "coef":list(lr.coef_.T)})
###########################################################################



# use Bagging to do model fusion
from sklearn.ensemble import BaggingRegressor

lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(lr, n_estimators=20, max_samples=0.8, max_features=1.0, n_jobs=-1) 
# here we can even set bootstrap=false to get duplicate samples
bagging_clf.fit(x_train, y_train)

save_result('prediciton_bagginglr', bagging_clf, x_test)













