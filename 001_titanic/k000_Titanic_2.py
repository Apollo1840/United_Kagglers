# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import  DataFrame
from patsy import dmatrices
from patsy import dmatrix
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib

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



def clean_and_munge_data(df):
    # 1 PassengerID : x
    # 2 Plass       : ---
    # 3 Name        : get title out of Name
    # 4 Sex         : labelEncoder
    # 5 Age         : use title to predict age by average, then cut it
    # 6 SibSp       : combine it with Parch to create family size
    # 7 Parch       : x
    # 8 Ticket      : x
    # 9 Fare        : predict Fare by pclass
    # 10 Cabin      : diffienciate the passengers by the one who has Cabin info and those who dont
    # 11 Embarked   : x
    
    # this time do not use get_dummies because we going to use dmatrix
    
    
    # Title
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].apply(lambda x: substrings_in_string(x, title_list))

    # all to mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme','Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms','Miss']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title =='':
            if x['Sex']=='Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title']=df.apply(replace_titles, axis=1)
    
    
    # Age
    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'Age'] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'Age'] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'Age'] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'Age'] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    
    df['AgeCat'] = df['Age']
    df.loc[ (df.Age<=10) ,'AgeCat'] = 'child'
    df.loc[ (df.Age>60),'AgeCat'] = 'aged'
    df.loc[ (df.Age>10) & (df.Age <=30) ,'AgeCat'] = 'adult'
    df.loc[ (df.Age>30) & (df.Age <=60) ,'AgeCat'] = 'senior'
    # df['AgeCat']=pd.cut(df['Age'], bins=[0, 10, 30, 60, 120], labels=['child', 'adult', 'senior', 'aged']
    
    
    # Family
    df['Family_Size']=df['SibSp'] + df['Parch']
    
    # Fare
    df.Fare = df.Fare.apply(lambda x: np.nan if x==0 else x)
    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    # Cabin
    df.loc[df.Cabin.notnull(), 'Cabin' ] = 1
    df.loc[df.Cabin.isnull(), 'Cabin' ] = 0

    
    # new attributes:
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
    df['AgeClass']=df['Age']*df['Pclass']
    df['ClassFare']=df['Pclass']*df['Fare_Per_Person']
    
    df['HighLow']=df['Pclass']
    df.loc[ (df.Fare_Per_Person<8) ,'HighLow'] = 'Low'
    df.loc[ (df.Fare_Per_Person>=8) ,'HighLow'] = 'High'


    # labelEncode the attributes
    le = preprocessing.LabelEncoder()
    # enc=preprocessing.OneHotEncoder()
    
    df['Sex']=le.fit_transform(df['Sex'])
    df['Title']=le.fit_transform(df['Title'])
    df['AgeCat']=le.fit_transform(df['AgeCat'])

    return df


def substrings_in_string(big_string, substrings):
    'return the first substring in string, or np.nan'
    
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    print(big_string)
    return np.nan




#  prepare the data for training 
    
full_X = data_train.append(data_test)
df = clean_and_munge_data(full_X)

formula_ml='Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 
# formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 
# train_Y, train_X = dmatrices(formula_ml, data=df, return_type='dataframe')
# see https://patsy.readthedocs.io/en/latest/formulas.html#formulas
# C means change it to dummies

all_x = dmatrix(formula_ml, data=df, return_type='dataframe')

train_X = all_x[:891] 
train_Y = data_train.Survived
test_X = all_x[891:] 

train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size=0.2)


# ----------------------------------------------------------------------------------
# modeling 
print('logistic regression')
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(C=1,tol=1e-8)
lr.fit(train_x, train_y)

model = lr
print(model.score(train_x, train_y))
print(model.score(test_x, test_y))

model.fit(train_X, train_Y)
save_result('predition_lr_strange', model, test_X)



print('grid Search')
param_grid = {'lr__C': [1e-1, 1.0, 10, 1e2, 1e3, 1e4, 1e5]}
pipeline=Pipeline([ ('lr',lr) ])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',
cv=StratifiedShuffleSplit(train_Y.values.ravel(), n_iter=10, test_size=0.2, train_size=None)).fit(train_X, train_Y.values.ravel())

print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
# report(grid_search.grid_scores_)

model = grid_search.best_estimator_
print(model.score(train_x, train_y))
print(model.score(test_x, test_y))

model.fit(train_X, train_Y)
save_result('predition_lr_grid_strange', model, test_X)


# more for grid_search

# report grid_score
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


clf=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1.0,
  min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
  verbose=0)



param_grid = dict()
pipeline=Pipeline([ ('clf',clf) ])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',
cv=StratifiedShuffleSplit(train_Y, n_iter=10, test_size=0.2, train_size=None)).fit(train_X, train_Y)

print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)


''' 
print('-----grid search end------------')
print ('on all train set')
scores = cross_val_score(grid_search.best_estimator_, train_x, train_y, cv=3,scoring='accuracy')
print (scores.mean(), scores)
print ('on test set')
scores = cross_val_score(grid_search.best_estimator_, test_x, test_y, cv=3,scoring='accuracy')
print (scores.mean(), scores)


print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))

model_file=MODEL_PATH+'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)
'''

# grid_search.best_estimator_.fit(train_X, train_Y)
save_result('predition_gs_strange', grid_search.best_estimator_, test_X)



'''
print('bagging regression')

from sklearn.ensemble import BaggingRegressor

lr = LogisticRegression(C=1.0,  tol=1e-5)
bagging_clf = BaggingRegressor(lr, n_estimators=20, max_samples=0.9, max_features=1.0, n_jobs=-1) 
# here we can even set bootstrap=false to get duplicate samples
bagging_clf.fit(train_x, train_y)


model = bagging_clf
print(model.score(train_x, train_y))
print(model.score(test_x, test_y))
'''



