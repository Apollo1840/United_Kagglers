# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:08:02 2018

@author: zouco
"""
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.style.use( 'ggplot' )
sns.set_style( 'white' )

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# single column
    
def plot_value_counts(df, column_name):   
    plt.title('the value counts of {}'.format(column_name))
    df[column_name].value_counts().plot(kind='bar')
    plt.xlabel(column_name)  
    plt.show()




# Kde ---------------------------------------------------------------------

def plot_kde_on(df, kde_column_name, category_column_name):
    plt.title('{} and {}'.format(kde_column_name, category_column_name))
    for label in df[category_column_name].unique():
        df.loc[df[category_column_name] == label, kde_column_name].plot(kind='kde')   
    plt.xlabel(kde_column_name) # plots an axis lable
    plt.ylabel("density") 
    plt.legend(df[category_column_name].unique(),loc='best') # sets our legend for our graph.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

#------------------------------------------------------------------------


def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()


    
def plot_category_by_NA(df, mis_col,cate_col):
    # mis_col means the name of the column which has NA value, cate_col is the category column
    withit = df.loc[pd.notnull(df[mis_col]), cate_col].value_counts()
    without = df.loc[pd.isnull(df[mis_col]), cate_col].value_counts()
    pd.DataFrame({'not null': withit, 'null': without}).transpose().plot(kind='bar', stacked=True)





class StackedPloter:
    
    def __init__(self, df):
        self.df = df
    
    def plot(self, xcol, ycol, **kwargs):
        if self.df[xcol].dtype == 'O':
            
            '''use crosstab maybe?'''
            
            normalized = kwargs.get('normalized', False)
            
            smalldf = pd.get_dummies(self.df[ycol],prefix=ycol)
            smalldf = pd.concat([self.df[xcol], smalldf], axis=1)
            if normalized:
                smalldf.groupby(xcol).agg(np.mean).plot(kind='bar', stacked=True)
            else:
                smalldf.groupby(xcol).agg(np.sum).plot(kind='bar', stacked=True)
        else:
            hist_data = []
            for label in self.df[ycol].unique():
                hist_data.append(self.df.loc[self.df[ycol]==label, xcol])
            
            plt.hist(hist_data, stacked=True, bins=50, label=self.df[ycol].unique())
            plt.xlabel(xcol)
            plt.ylabel('Number')
            plt.legend()
            plt.show()    
                
        
'''
    
def plot_stacked_barchart(df, xcol, ycol, normalized=False):
    # xcol, ycol are column names
    # assume the number of rows is the val_col (value_column)
    
    'use crosstab maybe?'
    
    smalldf = pd.get_dummies(df[ycol],prefix=ycol)
    smalldf = pd.concat([df[xcol], smalldf], axis=1)
    if normalized:
        smalldf.groupby(xcol).agg(np.mean).plot(kind='bar', stacked=True)
    else:
        smalldf.groupby(xcol).agg(np.sum).plot(kind='bar', stacked=True)


def plot_stacked_hist(df, var, cat):
    hist_data = []
    for label in df[cat].unique():
        hist_data.append(df.loc[df[cat]==label, var])
    
    plt.hist(hist_data, stacked=True, bins=50, label=df[cat].unique())
    plt.xlabel(var)
    plt.ylabel('Number')
    plt.legend()
    plt.show()
    
'''


# -------------------------------------------------------------------------



def plot_variable_importance( X , y ):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    
    
    
def plot_contract_bar(df, colname, target, scaled=False):
    # df[colname] is a binary column, target is a category column
    df[colname+'_rev']= 1 - df[colname]
    the_type = 'mean' if scaled else 'sum'
    df.groupby(target)[[colname, colname+'_rev']].agg(the_type).plot(kind='bar', figsize=(25, 7),
                                                        stacked=True, colors=['g', 'r']);

def plot_contract_hist(df, colname, target):
    # df[colname] is a binary column, target is a numeric column
    plt.hist([df[df[colname] == 1][target], df[df[colname] == 0][target]], 
         stacked=True, color = ['g','r'],
         bins = 50)
    
def plot_contract_scatter(df, colname, target1, target2):
    # df[colname] is a category column, targets are numeric columns
    plt.scatter(df[target1], df[target2],c=df[colname]) 

def scatter_with_color(df, xcol, ycol, scol, colcol, collist):
        ax = plt.subplot()   
        for item,color in zip(df[colcol].unique(),collist):
            tempdf = df.loc[df[colcol] == item,:]
            ax.scatter(tempdf[xcol], tempdf[ycol], s=tempdf[scol], c=color)


if __name__=='__main__':
    
    ax = plt.subplot()
    ax.set_ylabel('Average fare')
    data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax)
    

