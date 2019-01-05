
# introduction

Alice.py is your primary teacher of data analysis and Kaggle.

In this md, we will introduce how to deal with Kaggle task. It is like a 武功秘籍.


# 1, Frist step

Frist we need to know the data, understand the meaning of attributes and visualize some performance and relations.
This step is helpful for us to choose and generate features, sometimes totally reform the problem.

## visualization

from technology perspective we have:

#### 1)value_counts

    df.column.value_counts.plot('bar')

to valuecount the column.

#### 2)ratio compare by:
    
    df.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')

#### 3)kde plot 
Plot kde of something for different category.

    serie.plot('kde')
    
use plot_distribution or violin plot, or multi-boxplot
    

# 2, Clean the data

## fillNA

we have several ways to fill the NA:

##### a.Trival approaches: 
0, forward fill, backward fill


##### b. Categoral approach: 
Find the most related columns, use the mean or median in this category to predict.


##### c. Model approach: 
Use some model trained on some related columns to fill the NA.



# 3, Feature engineering

##### a. get_dummies

##### b. get_dummies_na
differentiate the entry with and without information in this column.

##### c. cut
cut the continues value to discrete

##### d. String comprehension
dig information out of string


# 4, Preprocessing

##### a. normalize the data

# 5, Build Model
Things we probably need to adjust:
##### a.C
##### b.penalty
##### c.tol


# 6, Evaluate the model
Use cross validation

# 7, Model Augmentation

##### a. Bagging

##### b. votingClassifier

##### c. grid search

















