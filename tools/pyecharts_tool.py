# -*- coding: utf-8 -*-
"""
extension of pyecharts

"""
from pyecharts import Boxplot,Sankey



# --------------------------------------------------------------------------
# more for boxplot

def boxplot_of_2_attr(df, value, attr1, attr2):
    
    boxplot = Boxplot("箱形图")
    data = data_of_2_attr(df, value, attr1, attr2)
    # print(data_of_2_attr(df,'Kills','Gender','Place'))
    
    name = data['name']
    dt = data['data']
    
    for i in range(len(set(df[attr1]))):
        boxplot.add(attr1+":{}".format(list(set(df[attr1]))[i]), name[i], 
                prepare_data(dt[i]), is_more_utils=True)
    boxplot.render('b2({}-{}).html'.format(attr1,attr2))

    
def data_of_2_attr(df,value,attr1,attr2):
    """
        attr1 and attr2 are all objects
        this is like df.groupby(attr1, attr2).value
    
    """
       
    name = []
    data = []
    for i in set(df[attr1]):
        data_inner = []
        name_inner = []
        for j in set(df[attr2]):
            slot=list(df.loc[(df[attr1]==i) & (df[attr2]==j),value])
            name_inner.append(attr2+':{}'.format(j))
            data_inner.append(slot)
        data.append(data_inner)
        name.append(name_inner)
    return {'name': name, 'data': data}

def prepare_data(list_x):
    # prepare data for boxplot
    
    list_y = []
    for i in list_x:
        if len(i) >=3:
            list_y.extend(Boxplot.prepare_data([i]))
        if len(i) == 2:
            list_y.extend(Boxplot.prepare_data([[i[0]]+i+[i[1]]]))
        if len(i) == 1:
            list_y.extend(Boxplot.prepare_data([[i[0]]+i]))
        if len(i) == 0:
            list_y.extend(Boxplot.prepare_data([[0,0,0]]))
    return list_y





# --------------------------------------------------------------------------
    
def cross_sankey(df, attr1, attr2, func=len, value=None):
    if value==None:
        df2=df.groupby([attr1, attr2])[attr1].agg([func])
    else:
        df2=df.groupby([attr1, attr2])[value].agg([func])
    df2=df2.reset_index()
    nodes = []
    for j in [attr1, attr2]:
        for i in set(df[j]):
            nodes.append({'name': str(i)})
    links = []
    for i in range(df2.shape[0]):
        if value==None:
            links.append({
                    'source':str(df2.loc[i,attr1]),
                    'target':str(df2.loc[i,attr2]),
                    'value':df2.loc[i,'len'],
                    })
        else:
            links.append({
                    'source':str(df2.loc[i,attr1]),
                    'target':str(df2.loc[i,attr2]),
                    'value':df2.loc[i,func.__name__],
                    })
    
    sankey = Sankey("桑基图示例", width=1200, height=600)
    sankey.add("sankey", nodes, links, line_opacity=0.8,
           line_curve=0.5, line_color='source',
           is_label_show=True, label_pos='right')
    sankey.render()