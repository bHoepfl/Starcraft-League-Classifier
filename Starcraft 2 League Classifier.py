# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:40:25 2019

@author: Brian Hoepfl
Starcraft 2 League Classifier with Deeplearning
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


starcraft_df = pd.read_csv('D:/Owner/Downloads/skillcraft/Skillcraft.csv')


print(starcraft_df.columns)

cols_to_norm = ['Age', 'HoursPerWeek', 'TotalHours', 'APM',
       'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys', 'MinimapAttacks',
       'MinimapRightClicks', 'NumberOfPACs', 'GapBetweenPACs', 'ActionLatency',
       'ActionsInPAC', 'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade',
       'ComplexUnitsMade', 'ComplexAbilitiesUsed']

#Normalized the columns of the data
starcraft_df[cols_to_norm] = starcraft_df[cols_to_norm].apply(lambda x:
    (x - x.min()) / (x.max() - x.min() ))

#Print the data from the Skillcraft.csv file
#print(starcraft_df.head())


#Creating our numeric columns for our continious variables
#We are making the feature columns 
age = tf.feature_column.numeric_column('Age')
hours_per_week = tf.feature_column.numeric_column('HoursPerWeek')
tot_hours = tf.feature_column.numeric_column('TotalHours')
apm = tf.feature_column.numeric_column('APM')
select_by_hotkeys = tf.feature_column.numeric_column('SelectByHotkeys')
assign_to_hotkeys = tf.feature_column.numeric_column('AssignToHotkeys')
unique_hotkeys = tf.feature_column.numeric_column('UniqueHotkeys')
minimap_attacks = tf.feature_column.numeric_column('MinimapAttacks')
minimap_right_clicks = tf.feature_column.numeric_column('MinimapRightClicks')
num_of_PACs = tf.feature_column.numeric_column('NumberOfPACs')
gap_btwn_PACs = tf.feature_column.numeric_column('GapBetweenPACs')
action_latency = tf.feature_column.numeric_column('ActionLatency')
actions_in_PAC = tf.feature_column.numeric_column('ActionsInPAC')
tot_map_explored = tf.feature_column.numeric_column('TotalMapExplored')
workers_made = tf.feature_column.numeric_column('WorkersMade')
unique_units_made = tf.feature_column.numeric_column('UniqueUnitsMade')
complex_units_made = tf.feature_column.numeric_column('ComplexUnitsMade')
complex_abilities_used = tf.feature_column.numeric_column('ComplexAbilitiesUsed')


#Here are my feature columns
feat_cols = [age, hours_per_week, tot_hours, apm, select_by_hotkeys, assign_to_hotkeys,
             unique_hotkeys, minimap_attacks, minimap_right_clicks, num_of_PACs,
             gap_btwn_PACs, action_latency, actions_in_PAC, tot_map_explored, 
             workers_made, unique_units_made, complex_units_made, complex_abilities_used]

#Dropping some features before conducting the train and test data split
x_data = starcraft_df.drop('GameID', axis=1)
x_data = starcraft_df.drop('LeagueIndex', axis=1)
labels = starcraft_df['LeagueIndex']

#print(labels.head())

#Splitting between training data and the test data 

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, 
                                                    test_size=0.3, random_state = 101)
#Now I am going to use an estimator

dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,20,25,20,10], feature_columns=feat_cols,
                                       n_classes=8)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,batch_size=10,
                                                 num_epochs= 1000, shuffle=True)

#Training the dnn_model
dnn_model.train(input_func, steps=2000)

train_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train, batch_size=10,
                                                       num_epochs=1, shuffle=False)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10,
                                                      num_epochs=1, shuffle=False)

train_metrics = dnn_model.evaluate(train_input_func)

test_metrics = dnn_model.evaluate(eval_input_func)

print('Train Metrics') # we are not overfitting at least
print(train_metrics)

print('Test Metrics')
print(test_metrics)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10,
                                                      num_epochs=1, shuffle=False)
predictions = dnn_model.predict(pred_input_func)

my_pred = list(predictions) # this is a list of dictionaries 

#print(my_pred[0:10])

pred_class_ID_arr = []
pred_class_ID = []
for elements in my_pred:
    pred_class_ID_arr.append(elements['class_ids'])

#print(len(pred_class_ID_arr))
for i in range(len(pred_class_ID_arr)):
    pred_class_ID.append(pred_class_ID_arr[i].item(0))


my_pred_df = pd.DataFrame(pred_class_ID, columns=['Prediction'])

y_test_df = pd.DataFrame(y_test.as_matrix(), columns= ['y test'])
pred_and_actual = pd.concat([y_test_df,my_pred_df], axis=1)


diff_df = pd.DataFrame(pred_and_actual['y test'] - pred_and_actual['Prediction'], 
                       columns=['Difference'])


#Initializing counts
count0 = 0
count1 = 0

#This loop will determine the count of predictions within 1 margin of error of
#the correct prediction
for row in diff_df['Difference'].iteritems():

    if(row[1] == 0):
        count1 += 1
    elif(row[1] >=(-1.0) and row[1]<=1.0):
        count0 += 1

#Determines the accuracy for exact and within 1 margin of error
Accuracy_1_away = (float(count1) / len(my_pred)) * 100
Accuracy_0_away = (float(count0) / len(my_pred)) * 100
Accuracy_0_and_1_away = Accuracy_1_away + Accuracy_0_away


print('Accuracy for 1 away: ' + str(Accuracy_1_away))
print('Accuracy for 0 away: ' + str(Accuracy_0_away))
print('Accuracy for 0 and 1 away: ' + str(Accuracy_0_and_1_away))








    
    
