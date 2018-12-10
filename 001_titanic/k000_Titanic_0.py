# -*- coding: utf-8 -*-
'''
    There are:
        Basic functions used in Kaggle project Titanic
        visualization functions

'''


import pandas as pd 
import numpy as np
from pandas import Series,DataFrame
import os

DATA_PATH='Projects\\datasets\\k000_titanic\\'


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


'''
    1, PassengerId: and id given to each traveler on the boat
    2, Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)
    3, The Name of the passeger
    4, The Sex
    5, The Age
    6, SibSp: number of siblings and spouses traveling with the passenger
    7, Parch: number of parents and children traveling with the passenger
    8, The ticket number
    9, The ticket Fare
    11, The cabin number
    12, The embarkation. This describe three possible areas of the Titanic from which the people embark. Three possible values S,C,Q

'''


def save_result(file_name, trained_model, test_data):
    predictions = trained_model.predict(test_data)
    result = pd.DataFrame({'PassengerId': list(range(892, 1310)), 'Survived':predictions.astype(np.int32)})
    result.to_csv(DATA_PATH + file_name + ".csv", index=False)




# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


def draw_plots():
    
    data_train, data_test = load_data()

    import matplotlib.pyplot as plt
    
    # value_counts
    plt.title('dead/survive(0/1)')
    data_train.Survived.value_counts().plot(kind='bar')
    plt.ylabel('people')  
    plt.show()
    
    plt.title('pclass distribution')
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel('people')
    plt.show()
    
    plt.title("embark location")
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.ylabel("people")
    plt.show()
    
    # logistic scatter
    plt.title('Age and survival ratio')
    plt.scatter(data_train.Survived, data_train.Age, alpha=0.05)
    plt.ylabel("Age")                        
    plt.grid(axis='y',b=True, which='major') 
    plt.show()
    
    
    # kde
    plt.title('Age and pclass')
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel("Age")# plots an axis lable
    plt.ylabel("density") 
    plt.legend(('1', '2','3'),loc='best') # sets our legend for our graph.
    
    

    # primary analysis
    data_train.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('Sex')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('Embarked')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('SibSp')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby(['Pclass','Sex'])['Survived'].agg(np.mean).plot('bar')

    
    # stacked bar
    attr = 'Age'
    withit = data_train.Survived[pd.notnull(data_train[attr])].value_counts()
    without = data_train.Survived[pd.isnull(data_train[attr])].value_counts()
    df=pd.DataFrame({'not null': withit, 'null': without}).transpose()
    print(df)
    df.plot(kind='bar', stacked=True)

    attr = 'Cabin'
    withit = data_train.Survived[pd.notnull(data_train[attr])].value_counts()
    without = data_train.Survived[pd.isnull(data_train[attr])].value_counts()
    df=pd.DataFrame({'not null': withit, 'null': without}).transpose()
    print(df)
    df.plot(kind='bar', stacked=True)


    #---------------------------------------------
    
    full = data_train.append( data_test , ignore_index = True )
    titanic = full[ :891 ]

    from ploters import plot_distribution, plot_categories, plot_correlation_map

    plot_correlation_map( titanic )
    
    # Plot distributions of Age of passangers who survived or did not survive
    plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
    
    # Plot survival rate by Embarked
    plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )



    #---------------------------------------------

    data = data_train.copy()
    
    # deal with age for plots
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    
    # some plots
    data['Died'] = 1 - data['Survived']
    data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot( kind='bar', figsize=(25, 7), stacked=True, colors=['g', 'r'])
    data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),stacked=True, colors=['g', 'r'])                                                         
    fig = plt.figure(figsize=(25, 7))
    
    
    sns.violinplot(x='Sex', y='Age', 
                   hue='Survived', data=data, 
                   split=True,
                   palette={0: "r", 1: "g"})
    
    figure = plt.figure(figsize=(25, 7))
    
    plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
             stacked=True, color = ['g','r'],
             bins = 50, label = ['Survived','Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()
    
    plt.figure(figsize=(25, 7))
    ax = plt.subplot()
    
    ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], 
               c='green', s=data[data['Survived'] == 1]['Fare'])
    ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], 
               c='red', s=data[data['Survived'] == 0]['Fare'])
    
    ax = plt.subplot()
    ax.set_ylabel('Average fare')
    data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax)
    
    fig = plt.figure(figsize=(25, 7))
    sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"})








