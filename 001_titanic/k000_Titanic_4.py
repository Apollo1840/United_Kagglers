# -*- coding: utf-8 -*-

# warining: this script has low quality

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
pd.options.display.max_columns = 100
import pandas_extend as pde

DATA_PATH='Projects\\datasets\\k000_titanic\\'

# ------------------------------------------------------------------------
import os

def change_dir_to_DAT():
    path = os.getcwd()
    while(os.path.basename(path) != 'Data-Analysis-Tools'):
        path = os.path.dirname(path)
    os.chdir(path)
    
change_dir_to_DAT()

# from pandas_extend import columns_with_na


os.chdir(os.getcwd() + '/Projects')

from k000_Titanic_0 import load_data
from k000_Titanic_0 import save_result

data_train, data_test = load_data()

full = data_train.append( data_test , ignore_index = True )
titanic = full[ :891 ]


from sklearn.preprocessing import StandardScaler

def generate_features(full, scaled=False):
    # 1 PassengerID : x
    # 2 Plass       : get_dummies
    # 3 Name        : get title out of Name
    # 4 Sex         : labelEncoder
    # 5 Age         : predict Age by median of Plass, Title, Sex
    # 6 SibSp       : ---, and combine it with Parch to create family_size, and discretenize it
    # 7 Parch       : ---
    # 8 Ticket      : x
    # 9 Fare        : predict Fare by average
    # 10 Cabin      : predict cabin with U and take the first char
    # 11 Embarked   : get_dummies
    
    
    # --- title ---
    # we extract the title from each name
    # full['Name']
    
    title = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
    # title['Title'].value_counts()
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
    
                        }
    
    # we map each title
    title = title.map( Title_Dictionary )
    full['Title'] = title # for age
    
    # --- sex ---
    # Transform Sex into binary values 0 and 1
    sex = full.Sex.map({'male': 1,'female':0})
    # sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
    
    
    # --- Age ---#
    grouped_median = full.groupby(['Sex','Pclass','Title'])['Age'].agg(np.median)
    grouped_median = grouped_median.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    
    def fill_age(row):
        condition = (
            (grouped_median['Sex'] == row['Sex']) & 
            (grouped_median['Title'] == row['Title']) & 
            (grouped_median['Pclass'] == row['Pclass'])
        ) 
        return grouped_median[condition]['Age'].values[0]
    
    Age = full.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    
    # -- Family ---
    family_size = full[ 'Parch' ] + full[ 'SibSp' ] + 1
    
    # (discretelization) introducing other features based on the family size
    family = pd.DataFrame()
    family[ 'Family_Size'] = family_size
    family[ 'Family_Single' ] = family_size.apply( lambda s : 1 if s == 1 else 0 )
    family[ 'Family_Small' ]  = family_size.apply( lambda s : 1 if 2 <= s <= 4 else 0 )
    family[ 'Family_Large' ]  = family_size.apply( lambda s : 1 if 5 <= s else 0 )
    
    
    # --- ticket ---
    full.Ticket.value_counts()
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket( ticket ):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = [t.strip() for t in ticket.split()]
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    
    # Extracting dummy variables from tickets:
    ticket = full[ 'Ticket' ].apply( cleanTicket )
    # ticket[ 'Ticket' ].value_counts()
    
     # --- Fare ---
    Fare = full.Fare.fillna( full.Fare.mean() )
    
        
    # --- cabin ---
    # replacing missing cabins with U (for Uknown)
    cabin = full.Cabin.fillna( 'U' )
    
    # mapping each Cabin value with the cabin letter
    cabin = cabin.apply( lambda c : c[0] )
    
    
    # ---- embarked ----
    embarked = full.Embarked.fillna('S')
    
    pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
    embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
    
    ticket = pd.get_dummies( ticket, prefix = 'Ticket' )
    cabin = pd.get_dummies( cabin, prefix = 'Cabin' )
    title = pd.get_dummies( title, prefix='Title')
    
    full_X = pd.concat( [ pclass, title, sex, Age, family, full.Parch, full.SibSp, ticket, Fare , cabin, embarked] , axis=1 )
    # print(full_X.columns)
    # print(full_X.head())
    
    if scaled==True:
        ss = StandardScaler()
        full_X.loc[:,pde.choose_columns(full_X,'float64')]=ss.fit_transform(full_X.loc[:,pde.choose_columns(full_X,'float64')])
        full_X.loc[:,'Family_Size']=ss.fit_transform(full_X.loc[:,'Family_Size'].values.reshape(-1,    1))

    full_X.shape
    return full_X



full_X = generate_features(full)

from sklearn.cross_validation import train_test_split

train_X = full_X[ 0:891 ]
train_Y = titanic.Survived.values.reshape(-1,1)
test_X = full_X[ 891: ]
train_x , test_x , train_y , test_y = train_test_split( train_X , train_Y , train_size = .7 )

print(full_X.shape , train_X.shape, train_Y.shape, test_X.shape)
print(train_x.shape , train_y.shape , test_x.shape, test_y.shape)




from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV





# pre-selection
pre_selection = False
if pre_selection:
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train_x, train_y)
    
    '''
    
    
    # see feature importance
    features = pd.DataFrame()
    features['feature'] = train_x.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    
    features.plot(kind='barh', figsize=(25, 25))
    '''
    
    selector = SelectFromModel(clf, prefit=True)
    train_x = selector.transform(train_x)
    test_y = selector.transform(test_X)



# try out the models

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()


models = [logreg, logreg_cv, rf, gboost]
for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_X, y=train_Y, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')
    
    
    
    
# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train_X, train_Y)
    model = grid_search.best_estimator_
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': True, 'min_samples_leaf': 8, 'n_estimators': 10, 
                  'min_samples_split': 10, 'max_features': 'log2', 'max_depth': 8}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train_X, train_Y)

# save_result('prediction_grid_rf', model, test_X)




models = [logreg, logreg_cv, rf, gboost, model, model, model, model, model]

predictions = []
for model in models:
    model.fit(train_X, train_Y)
    predictions.append(model.predict_proba(test_X)[:, 1])
    
predictions_df = pd.DataFrame(predictions).T
predictions_df['out'] = predictions_df.mean(axis=1)
predictions_df['PassengerId'] = data_test['PassengerId']
predictions_df['out'] = predictions_df['out'].map(lambda s: 1 if s >= 0.5 else 0)

predictions_df = predictions_df[['PassengerId', 'out']]
predictions_df.columns = ['PassengerId', 'Survived']

predictions_df.to_csv(DATA_PATH + 'prediction_blending3.csv', index=False)

p3 = pd.read_csv(DATA_PATH + 'prediction_blending3.csv')
p2 = pd.read_csv(DATA_PATH + 'prediction_blending2.csv')
p1 = pd.read_csv(DATA_PATH + 'prediction_blending0.csv')
p0 = pd.read_csv(DATA_PATH + 'prediction_blending2_80.csv')

print(p0.loc[p0.Survived != p3.Survived, 'PassengerId'])





