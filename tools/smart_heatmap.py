# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:29:23 2018

@author: zouco
"""
 
# main topic: give df, return meaningful heatmap

import pandas as pd
import numpy as np


# step 1 enumerate all the possible heatmap
# step 2 use heatmap_rubik to modify the heatmap  

import pandas_extend as pde

def heatmap_rubik(m):
    '''
    take a matrix, return rubik transformation of the matrix which has special properties
    
    '''
    
    ref_list=[]
    for i in range(m.shape[0]):
        ref_list.append(np.ma.median(m[i,:]))
    l_row = [i for i,j in sorted(enumerate([-i for i in ref_list]), key = lambda x: x[1])]
    m=np.array([m[i,:] for i in l_row])
    
    ref_list=[]
    for i in range(m.shape[1]):
        ref_list.append(np.ma.median(m[:,i]))
    l_column = [i for i,j in sorted(enumerate(ref_list), key = lambda x: x[1])]
    m=np.array([m[:,i] for i in l_column]).T
    return m

def heatmap_rubik_df(df):
    # only works for denoted pivot table created by pde.pivot_table_df
    
    ref_list=[]
    for i in range(df.shape[0]):
        ref_list.append(np.ma.median(df.iloc[i,1:]))
    l_row = [i for i,j in sorted(enumerate([-i for i in ref_list]), key = lambda x: x[1])]
    
    df2=pd.DataFrame.copy(df)
    
    j=0
    for i in l_row:
        df2.iloc[j] = df.iloc[i]
        j+=1
    
    ref_list=[]
    for i in range(1, df.shape[1]):
        ref_list.append(np.ma.median(df.iloc[:,i]))
    l_column = [i+1 for i,j in sorted(enumerate(ref_list), key = lambda x: x[1])]
    
    df3=pd.DataFrame.copy(df2)
    
    j=1
    for i in l_column:
        df3.iloc[:,j] = df2.iloc[:,i]
        df3.columns[j] = df2.columns[i]
        j+=1
    
    return df3

def heatmap_index(m):
    s=0
    for i in range(m.shape[0]-1):
        for j in range(1, m.shape[1]):
            s += 2*m[i,j] - m[i+1,j] - m[i,j-1]
    for i in range(1, m.shape[0]):
        s += m[i,1] + m[i-1,0] - 2*m[i,0]
    for j in range(1, m.shape[1]-1):
        s += m[-1,j+1] + m[-2,j] - 2*m[-1,j]
    return s/(2*float(np.max(m)))


if __name__=='__main':
    
    df=pd.read_csv('movie_metadata.csv')
    df.dropna(inplace=True)
    df2=pde.pivot_table_df(df, index='actor_1_facebook_likes_level', columns = 'facenumber_in_poster_level', values='gross', aggfunc=np.mean )
    print(df2)
    df3=heatmap_rubik_df(df2)
    print(df3)
    print(heatmap_index(df3.values[:,1:]))
    
     
    
    
    m=np.array([[1,3,1],[2,4,2],[5,2,8]])
    m2=np.array([[5,4,6],[2,1,3],[8,7,9]])
    m3=heatmap_rubik(m)
    m4=heatmap_rubik(m2)
    print(m)
    print(heatmap_index(m))
    print(m3)
    print(heatmap_index(m3)) 
    print(m2)   
    print(heatmap_index(m2))
    print(m4)
    print(heatmap_index(m4)) 
    
    df2.iloc[:,1:].values
    df2.shape
    
    
    

    

    