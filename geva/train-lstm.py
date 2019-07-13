import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import time
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint


weight_file = 'lstm_weights.h5'
model_file = 'lstm_model.h5'
LSTM_width = 128
batch_size = 8
epochs = 20
validation_split = 0.2

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

model = Sequential()
model.add(LSTM(units=LSTM_width, input_shape=(weekdayData_scaled.shape[1], 1), return_sequences=False))
model.add(Dense(units=weekdayData_scaled.shape[1], activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print(model.summary())
input = np.expand_dims(weekdayData_scaled, axis=2)
checkpoint = ModelCheckpoint(weight_file)
model.fit(x=input, y=weekdayData_scaled,
                       batch_size=batch_size, epochs=epochs,
                       verbose=1, validation_split=validation_split,
                       callbacks=[checkpoint])
                       
model.save(model_file)