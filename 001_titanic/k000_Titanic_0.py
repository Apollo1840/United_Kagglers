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

from ploters import plot_value_counts, plot_kde_on, plot_category_by_NA

def draw_plots():
    
    data_train, data_test = load_data()
    
    # value_counts
    plot_value_counts(data_train, 'Survived')
    plot_value_counts(data_train, 'Pclass')
    plot_value_counts(data_train, 'Embarked')
    
    # logistic scatter
    plt.title('Age and survival ratio')
    plt.scatter(data_train.Survived, data_train.Age, alpha=0.05)
    plt.ylabel("Age")                        
    plt.grid(axis='y',b=True, which='major') #?
    plt.show()
    
    
    # kde
    plot_kde_on(data_train, 'Age','Pclass')
    
    
    # primary analysis
    data_train.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('Sex')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('Embarked')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby('SibSp')['Survived'].agg(np.mean).plot('bar')
    data_train.groupby(['Pclass','Sex'])['Survived'].agg(np.mean).plot('bar')

    
    # stacked bar
    plot_category_by_NA(data_train, 'Age', 'Survived')
    plot_category_by_NA(data_train, 'Cabin', 'Survived')


    #---------------------------------------------
    
    full = data_train.append( data_test , ignore_index = True )
    titanic = full[ :891]

    from ploters import plot_distribution, plot_categories, plot_correlation_map

    plot_correlation_map( titanic )
    
    # Plot distributions of Age of passangers who survived or did not survive
    # upgrade of plot_kde_on : kde of 'var' by 'target'
    plot_distribution( titanic , var = 'Age' , target = 'Survived')
    plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
    
    # Plot survival rate by Embarked
    plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
    # By default the height of the bars/points shows the mean and 95% confidence interval



    #---------------------------------------------

    data = data_train.copy()
    
    df = data
    plt.scatter(df['Age'], df['Fare'], c=df['Survived'], s=df['Fare'], cmap='seismic', alpha=0.8) 
    #https://matplotlib.org/examples/color/colormaps_reference.html
    
    from ploters import StackedPloter
    sp = StackedPloter(data)

    # some plots   
    sp.plot('Sex', 'Survived')
    
    plt.figure()
    sp.plot('Fare', 'Survived')
    
    
    sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, palette={0: "r", 1: "g"})
        fig = plt.figure(figsize=(25, 7))
    
    sns.violinplot(x='Sex', y='Age', hue='Survived', 
                   data=data, 
                   split=True,
                   palette={0: "r", 1: "g"})
    # it is like plot distribution with row=x, var=y, target=hue
    # personal perfers the plot_distribution
    








