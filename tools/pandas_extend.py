# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


class pde():
    def __init__(self, df):
        self.df = df


# NA    
class NA_refiner(pde):
    '''
    How to us NA_refiner:

    '''
    
    def show(self):
        # print the names of columns with NA value and the number of NA values
        
        dfna = self.df.isnull().sum()/self.df.shape[0]*100
        dfna = dfna[dfna > 0].sort_values(ascending = False)
        print(dfna)
        try:
            dfna.plot("bar")
        except:
            pass
        return dfna
        
         #return list(self.df.columns[self.df.isna().any()])
    
    def fill_by(self, col, method, **kwargs):
        if not isinstance(method, str):
            # probably the trained model
            
            features = kwargs.get('features', None)
            # feature to predict
            
            cols = kwargs.get('cols', None)
            # cols used to predict
            
            na_rows_index = self.df[features].isnull().any(axis=1)
            unknown = self.df.loc[ na_rows_index, cols].values
            self.df.loc[na_rows_index, features ] = method.predict(unknown)
            return self.df
            
        elif method == 'mean':
            self.df[col].fillna(self.df[col].mean(),inplace=True)
        elif method =='mode':
            self.df[col].fillna(self.df[col].mode()[0],inplace=True)
        elif method == 'None':
            self.df[col].fillna('None',inplace=True)
        elif method == 'meanBy':
            cols = kwargs.get('cols', None)
            self.df[col] = self.df.groupby(cols)[col].transform(
                    lambda x: x.fillna(x.mean()))
        elif method == 'medienBy':
            cols = kwargs.get('cols', None)
            self.df[col] = self.df.groupby(cols)[col].transform(
                    lambda x: x.fillna(x.median()))
            


class da_opt():
    """
    example:  
        import da_opt as dp
        dp.trust_amount = 10
        dp.mean()
    """
    
    trust_amount = 10
    replace_num = np.nan    
    
    @classmethod
    def mean(cls, x):
        if len(x) >= cls.trust_amount:
            return np.mean(x)
        else:
            return cls.replace_num
    
    @classmethod
    def deviation_level(cls, x):
        # x-u/segma
        m=np.nanmean(x)
        segma = np.std([j for j in x if j >= x.quantile(0.1) and j <= x.quantile(0.9)])
        return np.array([np.abs(j-m)/segma for j in x])
    

def choose_columns(df, dt_type='float64'):
    # pick all the column names which contains dt_type data 
    # example : choose_columns(df2, np.number)
    
    return list(df.iloc[0:1,:].select_dtypes(include = dt_type).columns)

    
def quantile_cut_column(df, column_name, quantile_ratio=[0,0.2,0.4,0.6,0.8,1], labels = ['small','medium-small','medium','medium-large','large']):
    # cut the column by quantile, returns the new df (with a new column  '..._level')
    
    # use case:
    # df2=quantile_cut_column(df, 'title_year')
    # print(df2)
    
    bins = list(df[column_name].quantile(quantile_ratio))
    print(bins)
    bins[0]=bins[0]-0.1
    df[column_name + '_level']=list(pd.cut(df[column_name], bins=bins, labels=labels))
    return df




def df_to_hm_data(df):
    # df to heatmap data of pyecharts, enumerate the df by row and column
    
    value = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value.append([i,j,df.iloc[i,j]])
    return value



def pivot_table_df(df, index, column, value, aggfunc):
    # columns can only be str. aggfunc can only have one func
        df2 = pd.pivot_table(df, index = index, columns = column, values = value, aggfunc = [aggfunc])
        df2 = pd.DataFrame(df2.to_records())
        df2.columns = [hdr.replace("('"+aggfunc.__name__+"', ", column+".").replace(")", "") for hdr in df2.columns]
        return df2 


def sortDF_column(df, column_order):
    df2 = df.T
    df2['columns'] =  df2.index
    df2.index = column_order
    df2 = df2.sort_index()
    columns = df2['columns']
    del df2['columns']
    df3 = df2.T
    df3.columns = columns
    return df3


def drop_outliers(df, list_column, level = 4):
    """
     here the outlier is defined as x-u < level*std
     
    """
    
    for i in list_column:
        m=np.nanmean(df[i])
        segma = np.std([j for j in df[i] if j >= df[i].quantile(0.1) and j <= df[i].quantile(0.9)])
        outlier_index = [np.abs(j-m)/segma for j in df[i]]
        df = df.drop(df.index[list(np.where(np.array(outlier_index) >= level))])
    return df

class multi_content_column():
    
    """
    how to use:
            mcc = multi_content_column(df.genres,'|')
            mcc2 = multi_content_column(df.plot_keywords,'|')
            print(mcc.num_classes)
            # print(mcc.list_of_classes)
            # print(mcc2.list_of_classes[:10])
    """
    
    def __init__(self,column,sep=' '):
        self.column=column
        self.sep=sep
    
    @property
    def value(self):
        # break the content into several elements in list
        value = []
        for i in self.column:
            value.append([j.strip() for j in i.split(self.sep)])
        return value
    
    @property
    def list_levels(self):
        return [len(i) for i in self.value]
    
    @property
    def classes(self):
        content = []
        for i in self.column:
            content.extend([j.strip() for j in i.split(self.sep)])  
        return pd.Series(content).value_counts()
    
    @property
    def num_classes(self):
        return self.classes.shape[0]
    
    @property
    def list_classes(self):
        return sorted(list(self.classes.index))
        
    def contains(self, content):
        # returns a list of bool, length equal to number of rows, True means the content is in that row.
        return [content in i for i in self.value]
    


def more_info(df):
    var = [] ; l = [] ; na=[]; t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
        na.append(sum(df[x].isnull()))
    info = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'missing_value': na,'Datatype' : t } )
    info.sort_values( by = 'Levels' , inplace = True )
    return info



# deprecated
# -------------------------------------------------------------------------
def columns_with_na(df):
    # return names of columns who has NA
    
    return list(df.columns[df.isna().any()])

def fillna_by(method, df, vcol):
    if method == 'mean':
        df[vcol].fillna(df[vcol].mean(),inplace=True)
    elif method =='mode':
        df[vcol].fillna(df[vcol].mode()[0],inplace=True)
    elif method == 'None':
        df[vcol].fillna('None',inplace=True)
    
    
def fillna_model(df, model, features=[], cols=[]):
    na_rows_index = df[features].isnull().any(axis=1)
    unknown = df.loc[ na_rows_index, cols].values
    df.loc[na_rows_index, features ] = model.predict(unknown)
    return df








 
if __name__=='__main__':

    
    a=[[1,2],[3,5],[5]]
    [5 in i for i in a]






 
