# Frist
frist we need to know the data

visualize some thing

from technology perspective we have:

## 1, value_counts

    df.column.value_counts.plot('bar')

to valuecount the column.

## 2, ratio compare by:
    
    df.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')

## 3, plot kde of something for different category.

    serie.plot('kde')
    
    use plot_distribution or violin plot.
    
    or multi-boxplot



