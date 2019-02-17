# -*- coding: utf-8 -*-
'''
    There are 3 steps of data analysis
        1, load data
        2, check column
        3, check column correlation

'''

###########################################################################
# 1 load data
PROJECT_NAME = '001_titanic'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

from tools.data_loader import load_data
data_train, data_test = load_data(DATA_PATH)


'''
    how to load data?
    1) download the data to datasets, put it into the project folder
    2) in py file, define the 2 variables above
    3) from tools.data_loader import load_data
    4) data_train, data_test = load_data(DATA_PATH)
    
'''



###########################################################################
# 2 check column
import matplotlib.pyplot as plt


from ploters import plot_value_counts
plot_value_counts(data_train, 'Survived')
plot_value_counts(data_train, 'Pclass')
plot_value_counts(data_train, 'Embarked')








###########################################################################
# 3 check column relationship
import numpy as np

os.chdir(os.getcwd()+'\\tools')


# first of the first: corr
from ploters import plot_corr
plot_corr(data_train)


# scatter for deep corr
plt.title('Age and survival ratio')
plt.scatter(data_train.Survived, data_train.Age, alpha=0.05)
plt.ylabel("Age")                        
plt.grid(axis='y',b=True, which='major') #?
plt.show()

plt.scatter(data_train['Age'], data_train['Fare'], c=data_train['Survived'], s=data_train['Fare'], cmap='seismic', alpha=0.8) 
#https://matplotlib.org/examples/color/colormaps_reference.html


# -----------------------------------------------------------

from ploters import KdePloter
KdePloter(data_train).plot('Age', lcol='Survived', row='Sex')


from ploters import StackedPloter
StackedPloter(data_train).plot('Fare', ycol='Survived', bins=20) # histogram
StackedPloter(data_train).plot('Sex', ycol='Survived', normalized=True)


from ploters import plot_crosstab_NA
plot_crosstab_NA(data_train, 'Age', ycol='Survived')
plot_crosstab_NA(data_train, 'Cabin', ycol='Survived')


# -----------------------------------------------------------

# for 0,1 target column (calculate ratio for different categories)
# simple one
data_train.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')
data_train.groupby('Sex')['Survived'].agg(np.mean).plot('bar')
data_train.groupby('Embarked')['Survived'].agg(np.mean).plot('bar')
data_train.groupby('SibSp')['Survived'].agg(np.mean).plot('bar')
data_train.groupby(['Pclass','Sex'])['Survived'].agg(np.mean).plot('bar')

# more beautiful
from ploters import plot_ratio
plot_ratio( data_train, ycol='Survived', xcol = 'Embarked')














# ======================================================================
# 4 data cleanning
df = data_train

# 4.1 fill NA

# simple approach
data_train.Fare.fillna(0, inplace=True)
data_train.Fare.fillna(method='ffill', inplace=True)
data_train.Fare.fillna(method='bfill', inplace=True)


# more complex approach:
df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

# or
grouped_median = df.groupby(['Pclass'])['Fare'].agg(np.median).reset_index()
Fare = df.apply(lambda row: grouped_median[grouped_median['Pclass']==row['Pclass']]['Age'].values[0] 
                    if np.isnan(row['Age']) else row['Age'], axis=1)


# take age for example
grouped_median = df.groupby(['Sex','Pclass','Title'])['Age'].agg(np.median).reset_index()
def fill_age(row):
    condition = (
        (grouped_median['Sex'] == row['Sex']) & 
        (grouped_median['Title'] == row['Title']) & 
        (grouped_median['Pclass'] == row['Pclass'])
    ) 
    return grouped_median[condition]['Age'].values[0]
Age = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)


# more complex, use a model to predict    
from pandas_extend import fillna_model
df = fillna_model(data_train, rfr, 'Age', cols)

# sometimes, we need subsitute 0 to NA
df.Fare = df.Fare.apply(lambda x: np.nan if x==0 else x)  



    
# 4.2 get dummies

# do it like this:
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

ticket = pd.get_dummies( ticket, prefix = 'Ticket' )
cabin = pd.get_dummies( cabin, prefix = 'Cabin' )
title = pd.get_dummies( title, prefix='Title')

full_X = pd.concat( [ pclass, title, sex, Age, family, ticket, Fare , cabin, embarked] , axis=1 )
  
    







# ======================================================================
# 5 build model













