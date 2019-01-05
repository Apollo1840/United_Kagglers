# Heading
## Subheads
### 1, section
#### 1) subsection
##### a. point



# introduction

Alice.py is your primary teacher of data analysis and Kaggle.

# Frist

frist we need to know the data


# visualize some thing


from technology perspective we have:

## value_counts

    df.column.value_counts.plot('bar')

to valuecount the column.

## ratio compare by:
    
    df.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')

## plot kde of something for different category.

    serie.plot('kde')
    
    use plot_distribution or violin plot.
    
    or multi-boxplot
    

# clean the data

## fillNA

we have several ways to fill the NA:

##### a.trival approaches: 
0, forward fill, backward fill

##### b. categoral approach: 
find the most related columns, use the mean or median in this category to predict.

##### c. model approach: 
use some model trained on some related columns to fill the NA.


# feature engineering

## get_dummies

## get_dummies_na
differentiate the entry with and without information in this column.

## cut
cut the continues value to discrete

## String comprehension
dig information out of string


# preprocessing

## normalize the data

# Build Model
Things we probably need to adjust:
##### a.C
##### b.penalty
##### c.tol


# evaluate the model
use cross validation

# Model Augmentation

## Bagging

## votingClassifier

## grid search

















