import matplotlib.pylab as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
#import heat_map
import pickle
# import fine_tune_rcnn
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import cv2
from tabulate import tabulate

df = pd.read_excel (r'C:\Users\michael\Desktop\project\xl_IG_ID.xlsx')
print(df)

options1 = ['no pain']
options2 = ['pain']


wrong_pred = df[(df['GT'].isin(options1)&df['PREDICTION'].isin(options2)) | (df['GT'].isin(options2)&df['PREDICTION'].isin(options1))]
print(wrong_pred)
def useful_graph(table,detection_order):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    #table.groupby(detection_order)['CONFIDENCE'].mean().sort_values(ascending=False).plot(kind='bar')
    avg_confidence = table.groupby(detection_order)['CONFIDENCE'].mean().sort_values(ascending=False)
    num_times = table[detection_order].value_counts()
    num_times = pd.DataFrame({detection_order:num_times.index,'times':num_times.values})
    avg_confidence = avg_confidence.to_frame()
    #num_times = num_times.to_frame()
    merged = pd.merge(num_times,avg_confidence,on=detection_order)
    merged['Weighted Score'] = merged['times']*merged['CONFIDENCE']
    merged['Weighted Score'] = 100*merged['Weighted Score']/merged['Weighted Score'].sum()
    print(merged)
    return merged

graph1 = useful_graph(wrong_pred,'main detection loc.')
def res_plot(graph1,detection_order,correctness):
    graph1.groupby(detection_order)['Weighted Score'].mean().sort_values(ascending=False).plot(kind='barh')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(29.5, 10.5, forward=True)
    matplotlib.pyplot.title(correctness,fontsize=26, fontweight = 'bold')
    plt.gca().yaxis.label.set_size(25)
    plt.gca().yaxis.label.set_color('red')
    plt.xlabel('Weighted Score[%]',fontweight = 'bold')
    plt.ylabel(detection_order,fontweight = 'bold')
    plt.gca().xaxis.label.set_color('red')
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    plt.show()

correct_pred_nopain = df[(df['GT'].isin(options1)&df['PREDICTION'].isin(options1))]
correct_pred_pain = df[(df['GT'].isin(options2)&df['PREDICTION'].isin(options2))]
wrong_nopain_pred = df[(df['GT'].isin(options2)&df['PREDICTION'].isin(options1))]
wrong_pain_pred = df[(df['GT'].isin(options1)&df['PREDICTION'].isin(options2))]
correctness = 'Correct NO PAIN Classification Facial Factors'
correctness_pain = 'Correct PAIN Classification Facial Factors'
wrongness_nopain = '{GT:PAIN,PREDICTION:NO PAIN} - Facial Factors'
wrongness_pain = '{GT:NO PAIN,PREDICTION:PAIN} - Facial Factors'

graph_correct_nopain = useful_graph(correct_pred_nopain,'main detection loc.')
wrongPred_secondary = useful_graph(wrong_pred,'main detection loc.')
wrongPred_nopain = useful_graph(wrong_nopain_pred,'main detection loc.')
wrongPred_nopain_secondary = useful_graph(wrong_nopain_pred,'secondary detection loc.')
correctPred_pain = useful_graph(correct_pred_pain,'main detection loc.')
wrongPred_pain = useful_graph(wrong_pain_pred,'main detection loc.')
wrongPred_pain_secondary = useful_graph(wrong_pain_pred,'secondary detection loc.')

res_plot(graph_correct_nopain,'main detection loc.',correctness)
#res_plot(wrongPred_secondary,'main detection loc.',correctness)
#res_plot(correctPred_pain,'main detection loc.',correctness_pain)
res_plot(wrongPred_nopain,'main detection loc.',wrongness_nopain)
#res_plot(wrongPred_pain,'main detection loc.',wrongness_pain)
res_plot(wrongPred_nopain_secondary,'secondary detection loc.',wrongness_nopain)
#res_plot(wrongPred_pain_secondary,'secondary detection loc.',wrongness_pain)

####################################### AVG#######################################################
df = pd.read_excel (r'C:\Users\michael\Desktop\project\xl_AVG_ID.xlsx')
print(df)

correct_pred_nopain = df[(df['GT'].isin(options1)&df['PREDICTION'].isin(options1))]
correct_pred_pain = df[(df['GT'].isin(options2)&df['PREDICTION'].isin(options2))]
wrong_nopain_pred = df[(df['GT'].isin(options2)&df['PREDICTION'].isin(options1))]
wrong_pain_pred = df[(df['GT'].isin(options1)&df['PREDICTION'].isin(options2))]


graph_correct_nopain = useful_graph(correct_pred_nopain,'main detection loc.')
wrongPred_secondary = useful_graph(wrong_pred,'main detection loc.')
wrongPred_nopain = useful_graph(wrong_nopain_pred,'main detection loc.')
wrongPred_nopain_secondary = useful_graph(wrong_nopain_pred,'secondary detection loc.')
correctPred_pain = useful_graph(correct_pred_pain,'main detection loc.')
wrongPred_pain = useful_graph(wrong_pain_pred,'main detection loc.')
wrongPred_pain_secondary = useful_graph(wrong_pain_pred,'secondary detection loc.')

res_plot(graph_correct_nopain,'main detection loc.',correctness)
#res_plot(wrongPred_pain,'main detection loc.',wrongness_pain)
#res_plot(correctPred_pain,'main detection loc.',correctness_pain)
res_plot(wrongPred_nopain,'main detection loc.',wrongness_nopain)
res_plot(wrongPred_nopain_secondary,'secondary detection loc.',wrongness_nopain)
#res_plot(wrongPred_pain_secondary,'secondary detection loc.',wrongness_pain)




record = {
    'Name': ['Ankit', 'Amit', 'Aishwarya', 'Priyanka', 'Priya', 'Shaurya'],
    'Age': [21, 19, 20, 18, 17, 21],
    'Stream': ['Math', 'Commerce', 'Science', 'Math', 'Math', 'Science'],
    'Percentage': [88, 92, 95, 70, 65, 78]}

# create a dataframe
dataframe = pd.DataFrame(record, columns=['Name', 'Age', 'Stream', 'Percentage'])

print("Given Dataframe :\n", dataframe)

options = ['Math', 'Science']

# selecting rows based on condition
rslt_df = dataframe[(dataframe['Age'] == 21) &
                    dataframe['Stream'].isin(options)]

print('\nResult dataframe :\n', rslt_df)
