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

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()


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


