import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import time
import os
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint

model_file = './lstm_model.h5'

dataset = pd.read_csv( index_col = 0,  parse_dates = True, infer_datetime_format = True, 
                         filepath_or_buffer = './data/2018-01-01__2019-01-01__NConservatory__allMerged.csv')

dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('America/Los_Angeles')
orderedSensorList = ( 'co2_1','co2_2', 'co2_3', 'co2_4',                        
                      'temp_1', 'temp_2', 'temp_3', 'temp_4',                     
                      'dew_1','dew_2', 'dew_3', 'dew_4',
                      'relH_1', 'relH_2', 'relH_3', 'relH_4',
                      'externTemp_1', 
                      'externHumid_1', 
                      'externSunrise_1',                      
                      'externCondition_1',
                    )
                    
orderedDataset  = dataset.reindex(index = dataset.index, columns = orderedSensorList)
dayIndexDF = pd.Series(index = orderedDataset.index, 
                       data = np.round(orderedDataset.index.dayofweek/6, decimals=2), 
                       name='dayIndex')
hourIndexDF = pd.Series(index = orderedDataset.index, 
                       data = np.round((orderedDataset.index.hour+(orderedDataset.index.minute/60))/24, decimals=6), 
                       name='hourIndex')
                       
saturdayVal = np.round(5/6,decimals=2)
sundayVal = np.round(6/6,decimals=2)
orderedDatasetTimeReference = pd.concat([orderedDataset, hourIndexDF, dayIndexDF], axis=1)
weekdayData = orderedDatasetTimeReference[ ( dayIndexDF != saturdayVal) &( dayIndexDF != sundayVal) ]
weekendData = orderedDatasetTimeReference[ ( dayIndexDF == saturdayVal) | (dayIndexDF == sundayVal) ]
continuousData = weekdayData.values[:, 0:17]
categoricalData = weekdayData.values[:, 17:]

standardScaler = StandardScaler()
standardScaler.fit(continuousData)

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(categoricalData)

scaledContinuousData = standardScaler.transform(continuousData)
scaledCategoricalData = minMaxScaler.transform(categoricalData)

weekdayData_scaled =  pd.DataFrame(index = weekdayData.index,
                                   data = np.hstack( (scaledContinuousData, scaledCategoricalData)),
                                   columns = weekdayData.columns)

weekdayData_scaled.drop(['externSunrise_1'],inplace=True,axis=1)

model=load_model(model_file)
input = np.expand_dims(weekdayData_scaled, axis=2)
target = model.predict(x=input)
err = np.linalg.norm(np.array(weekdayData_scaled) - target, axis=-1)
print('Mean:', err.mean())
print('Std dev:', err.std())
anomalies=np.argwhere(err>err.mean()+err.std()*3)
print(anomalies)
np.save('lstm_err_vector.pickle',err)