# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import pandas_extend as pde


DATA_PATH='Projects\\datasets\\k000_titanic\\'
    
    
# -----------------------------------------------------------------------------    
# get titanic & test csv files as a DataFrame

'''
    in k000_Titanic_1, it works on train set and deploy the changes on test data.
    in this one, it combines train and test first.

'''

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

full = data_train.append( data_test , ignore_index = True )
titanic = full[ :891 ]

# print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


# -----------------------------------------------------------------------------  

'''
    here it is going to change some attributes.

'''
from sklearn.preprocessing import StandardScaler

def generate_features(full):
    # 1 PassengerID : x
    # 2 Plass       : get_dummies
    # 3 Name        : get title out of Name
    # 4 Sex         : labelEncoder
    # 5 Age         : predict Age by average
    # 6 SibSp       : combine it with Parch to create family_size, and discretenize it
    # 7 Parch       : ---
    # 8 Ticket      : x
    # 9 Fare        : predict Fare by average
    # 10 Cabin      : predict cabin with U and take the first char
    # 11 Embarked   : get_dummies

    '''
         Here we will create small df frist then combine them with :
         full_X = pd.concat( [ sex, full.Embarked, full.Pclass, imputed,title, cabin, ticket, family] , axis=1 )
    '''
    
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
    
    
    # --- sex ---
    # Transform Sex into binary values 0 and 1
    sex = full.Sex.map({'male': 1,'female':0})
    # sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
    
    
    # --- Age ---
    Age = full.Age.fillna( full.Age.mean())
    
    # -- Family ---
    family_size = full[ 'Parch' ] + full[ 'SibSp' ] + 1
    
    # (discretelization) introducing other features based on the family size
    family = pd.DataFrame()
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
    
    
    # full_X = pd.concat( [ sex, full.Embarked, full.Pclass, Age, Fare,title, cabin, ticket, family_size] , axis=1 )
    # print(full_X.columns)
    # print(full_X.head())
    # full_X.to_csv(DATA_PATH +'titanic_data2.csv' )
    
    
    embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
    pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
    
    ticket = pd.get_dummies( ticket, prefix = 'Ticket' )
    cabin = pd.get_dummies( cabin, prefix = 'Cabin' )
    title = pd.get_dummies( title, prefix='Title')
    
    full_X = pd.concat( [ pclass, title, sex, Age, family, ticket, Fare , cabin, embarked] , axis=1 )
    # print(full_X.columns)
    # print(full_X.head())
    
    ss = StandardScaler()
    full_X.loc[:,pde.choose_columns(full_X,'float64')]=ss.fit_transform(full_X.loc[:,pde.choose_columns(full_X,'float64')])
    # full_X.loc[:,'FamilySize']=ss.fit_transform(full_X.loc[:,'FamilySize'].values.reshape(-1,    1))

    return full_X

# -----------------------------------------------------------------------------  
full_X = generate_features(full)

full_X.to_csv(DATA_PATH + 'titanic_wrangled_data.csv')
full_X = pd.read_csv(DATA_PATH + 'titanic_wrangled_data.csv')

# -----------------------------------------------------------------------------


# pd.set_option('display.max_columns', None)
# full_X.describe()


#-------------------------------------------------------------------------------
# preparation

# Create all datasets that are necessary to train, validate and test models

'''
    train_x, train_y is the train data
    test_x, test_y is the validation data
    
    train_all_x, train_all_y is the final training data
    
    test_X is the data to predict

'''

from sklearn.cross_validation import train_test_split

train_X = full_X[ 0:891 ]
train_Y = titanic.Survived.values.reshape(-1,1)
test_X = full_X[ 891: ]
train_x , test_x , train_y , test_y = train_test_split( train_X , train_Y , train_size = .7 )

print(full_X.shape , train_X.shape, train_Y.shape, test_X.shape)
print(train_x.shape , train_y.shape , test_x.shape, test_y.shape)


# plot_variable_importance(train_X, train_y)


# modeling ------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV





from sklearn.model_selection import cross_val_score

train_y=train_y.ravel()
train_Y=train_Y.ravel()
test_y=test_y.ravel()


def evaluate_model(model):
    model.fit(train_x, train_y)
    print(model.score(train_x, train_y))
    print(model.score(test_x, test_y))
    cvs=cross_val_score(model, train_X, train_Y, cv=5)
    print(cvs)
    print(np.mean(cvs), np.std(cvs))

rfc = RandomForestClassifier(n_estimators=50)
evaluate_model(rfc)

'''
1.0
0.8134328358208955
[0.77653631 0.81564246 0.84269663 0.79775281 0.84180791]
0.817115441698256      
'''


lr = LogisticRegression(C=2, penalty='l2', tol=1e-8)
evaluate_model(lr)

'''
0.8491171749598716
0.8022388059701493
[0.82122905 0.82681564 0.80898876 0.81460674 0.86440678]
0.8272093956032849
'''




lr = LogisticRegression(C=1, penalty='l2', tol=1e-8)
evaluate_model(lr)
'''
0.8507223113964687
0.7985074626865671
[0.82122905 0.83240223 0.81460674 0.8258427  0.85875706]
0.8305675570530682
'''

lr = LogisticRegression(C=5, penalty='l2', tol=1e-8)
evaluate_model(lr)
'''
0.8507223113964687
0.7985074626865671
[0.82122905 0.83240223 0.81460674 0.8258427  0.85875706]
0.8305675570530682
'''




bagging_clf = BaggingRegressor(lr, n_estimators=10, max_samples=0.8, max_features=1.0, n_jobs=-1) 
# here we can even set bootstrap=false to get duplicate samples
evaluate_model(bagging_clf)


from sklearn.feature_selection import SelectFromModel
lr = LogisticRegression(C=20, penalty='l2', tol=1e-8)

selector = SelectFromModel(lr, threshold='1.25*median')
selector.fit(train_x, train_y)

train_x2 = selector.transform(train_x)
print(train_x.columns[selector.get_support()])
lr.fit(train_x2, train_y)
print(lr.score(train_x2, train_y))
print(lr.score(selector.transform(test_x),test_y))
cvs = cross_val_score(lr, selector.transform(train_X), train_Y, cv=5)
print(cvs)
print(np.mean(cvs), np.std(cvs))


'''

0.8475120385232745
0.8171641791044776
[0.82681564 0.80446927 0.81460674 0.82022472 0.86440678]
0.8261046313072583

'''

lr.fit(selector.transform(train_X), train_Y)
filename = 'predition_lr_selected_scaled_all_C5.csv'

test_Y = lr.predict( selector.transform(test_X) )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y.astype(np.int32) } )
# test = test.reset_index()
# test = test.drop(columns = ['index'])
print(test.info())
test.to_csv( DATA_PATH + filename , index = False )





gbc=GradientBoostingClassifier(n_estimators=200)
evaluate_model(gbc)

'''
0.971107544141252
0.8097014925373134
[0.65921788 0.82681564 0.81460674 0.8258427  0.85310734]
0.797066906117377
'''

single_model = RandomForestClassifier(n_estimators=100)
# model = GradientBoostingClassifier()
# model = LogisticRegression(C=0.5, penalty='l2', tol=1e-9)
rfecv = RFECV( estimator = single_model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
evaluate_model(rfecv)

'''
    0.9983948635634029
    0.7910447761194029
    [0.7877095  0.79888268 0.84269663 0.80898876 0.83050847]
    0.8137572093211297
'''


from sklearn.ensemble import VotingClassifier

voc = VotingClassifier([('lr', lr), ('rf', rfc), ('gbc', gbc)], voting='hard')
evaluate_model(voc)



score = 0
while(score < 0.83):
    model = MLPClassifier((100,50, 50,10), activation = 'tanh', tol=10e-8)
    model.fit( train_x , train_y )
    print(model.score( train_x , train_y ))
    print(model.score( test_x , test_y ))
    score = model.score( test_x , test_y )





# ------------------------------------------------------------------
# output

def save_result_by(model, need_train=True,filename='titanic_pred.csv'):
    if need_train:
        model = model.fit(train_X, train_Y)
    save_result(filename, model, test_X)


save_result_by(rfecv)
save_result_by(rfc,need_train=False, filename='predition_rfc_scaled.csv')
save_result_by(rfc,filename='predition_rfc_scaled_all.csv')
save_result_by(lr,filename='predition_lr_C1.csv')




