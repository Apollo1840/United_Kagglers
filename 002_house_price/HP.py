import pandas as pd 
import numpy as np
from pandas import Series,DataFrame
import os

DATA_PATH='Projects\\datasets\\k001_house_price\\'


def change_dir_to_DAT():
    path = os.getcwd()
    while(os.path.basename(path) != 'Data-Analysis-Tools'):
        path = os.path.dirname(path)
    os.chdir(path)

def load_data():
    change_dir_to_DAT()
    
    # data_train = pd.read_csv(os.path.dirname(__file__)+'\\datasets\\k000_titanic\\train.csv')
    data_train = pd.read_csv(DATA_PATH + 'train.csv')
    data_test = pd.read_csv(DATA_PATH + 'test.csv')
    
    return data_train, data_test


data_train, data_test = load_data()

data_train.info()
data_train.shape
data_test.info()
data_test.shape


full_X = data_train.append(data_test, ignore_index=True)


#Checking for missing data
NAs = pd.concat([data_train.isnull().sum(), data_test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]

'''
# Prints R2 and RMSE scores
from sklearn.metrics import r2_score, mean_squared_error
def get_score(prediction, labels):    
    print('R2: {}'.format(r2_score(prediction, labels)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))
'''


from sklearn.preprocessing import StandardScaler
def preprocess(df):
    
    df.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
    
    df = filling_NA(df)
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
    
    df = get_dummie(df)
    
    # scale some attributes
    need_scale_columns = ['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']
    ss =  StandardScaler()
    df.loc[:, need_scale_columns] = ss.fit_transform(df.loc[:, need_scale_columns])
     
    return df
    
def filling_NA(df):
    
    # some need categorize
    num2str_columns = ['YrSold', 'MoSold', 'OverallCond','KitchenAbvGr']
    for c in num2str_columns:
         df[c] = df[c].astype(str)
         
    # some take mean
    fillWithMean_columns = ['LotFrontage']
    for c in fillWithMean_columns:
        df[c] = df[c].fillna(df[c].mean())    
    
    # some take mode
    fillWithMode_columns = ['MSZoning', 'MasVnrType', 'Electrical', 'KitchenQual', 'SaleType']
    for c in fillWithMode_columns:
        df[c] = df[c].fillna(df[c].mode()[0])   
    
    # some marked as none
    fillWithNA_columns = ['Alley','BsmtQual', 'BsmtCond', 
                          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'FireplaceQu', 'GarageType', 'GarageFinish', 
                          'GarageQual']
    for c in fillWithNA_columns:
        df[c] = df[c].fillna('No_such_thing')   
    
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['GarageCars'] = df['GarageCars'].fillna(0.0)
        
    return df
    
    

def get_dummie(df):
    # Getting Dummies from Condition1 and Condition2
    conditions = set([x for x in df['Condition1']] + [x for x in df['Condition2']])
    dummies = pd.DataFrame(data=np.zeros((len(df.index), len(conditions))),
                           index=df.index, columns=conditions)
    for i, cond in enumerate(zip(df['Condition1'], df['Condition2'])):
        dummies.loc[i, cond] = 1
        
    df = pd.concat([df, dummies.add_prefix('Condition_')], axis=1)
    df.drop(['Condition1', 'Condition2'], axis=1, inplace=True)
    
    
    # Getting Dummies from Exterior1st and Exterior2nd
    exteriors = set([x for x in df['Exterior1st']] + [x for x in df['Exterior2nd']])
    dummies = pd.DataFrame(data=np.zeros((len(df.index), len(exteriors))),
                           index=df.index, columns=exteriors)
    for i, ext in enumerate(zip(df['Exterior1st'], df['Exterior2nd'])):
        dummies.loc[i, ext] = 1
        
    df = pd.concat([df, dummies.add_prefix('Exterior_')], axis=1)
    df.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)
    
    # Getting Dummies from all other categorical vars
    for col in df.dtypes[df.dtypes == 'object'].index:
        for_dummy = df.pop(col)
        df = pd.concat([df, pd.get_dummies(for_dummy, prefix=col)], axis=1)
    
    return df



# scale the outcome
# ax = sns.distplot(train_labels)
# the SalePrice is shifted(skwed to left) we need to transform it a bit
# df.loc[df.SalePrice.notna(), 'SalePrice_log'] = np.log(df.loc[df.SalePrice.notna(),'SalePrice'])
y_train_full = data_train.SalePrice.apply(np.log)
del full_X['SalePrice']

full_X = preprocess(full_X)

x_train_full = full_X[ 0:data_train.shape[0]]
x_test_full = full_X[ data_train.shape[0]:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x_train_full, y_train_full, train_size = .7)

print(full_X.shape , x_train_full.shape, x_train_full.shape, x_test.shape)
print(x_train.shape , y_train.shape , x_test.shape, y_test.shape)




# Shows scores for train and validation sets  
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score  

def train_test(estimator, train_x , test_x , train_y , test_y, train_X, train_Y):
    
    s_trn = estimator.score(train_x, train_y)
    l_trn = np.sqrt(mean_squared_error(estimator.predict(train_x), train_y))
    
    s_tst = estimator.score(test_x, test_y)
    l_tst = np.sqrt(mean_squared_error(estimator.predict(test_x), test_y))
    
    scores = cross_val_score(estimator, train_X, train_Y, cv=5)
    
    print(estimator)
    print(' Train: {} \t {} \n test: {} \t {}'.format(s_trn, l_trn, s_tst,l_tst))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    
def check_model(model):
    model.fit(x_train, y_train)
    train_test(model, x_train, x_test, y_train, y_test, x_train_full, y_train_full)
    
def use_model2output(model):    
    model_trained = model.fit(x_train_full, y_train_full)
    pd.DataFrame({'Id': data_test.Id, 'SalePrice': np.exp(model_trained.predict(x_test_full)) }).to_csv(DATA_PATH + 'prediction_GBest.csv', index =False) 


    
    
from sklearn.ensemble import GradientBoostingRegressor
GBest = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, 
                                  max_features='sqrt', min_samples_leaf=15, 
                                  min_samples_split=10, 
                                  loss='huber')

check_model(GBest)
use_model2output(GBest)








