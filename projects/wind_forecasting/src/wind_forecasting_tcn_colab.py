#!/usr/bin/env python
# coding: utf-8

# ## Wind Forecasting TCN
# This is an exploration of the following paper [**Wind Power Forecasting with Deep Learning Networks: Time-Series Forecasting**](https://www.mdpi.com/2076-3417/11/21/10335) for learning purposes :)

# In[3]:


import re, time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE


# In[4]:


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)


# In[5]:


import pandas as pd


# In[6]:


time_series_data = pd.read_csv('T1.csv')
time_series_data["Date/Time"] = pd.to_datetime(time_series_data["Date/Time"], format='%d %m %Y %H:%M')
time_series_data.set_index('Date/Time', inplace=True)


# In[7]:


time_series_data.head(10)


# In[8]:


time_series_data.values


# In[9]:


# Some data pre-processing to fill in the gaps of data using linear interpolation
time_series_data_resampled = time_series_data.resample('10min')
time_series_data_interpolated = time_series_data_resampled.interpolate(method='linear')


# In[10]:


plt.plot(time_series_data_interpolated.loc['2018-01-01 00:00:00':'2018-01-31 23:50:00']['LV ActivePower (kW)'])


# In[11]:


print("Frequency of the index:", time_series_data_interpolated.index.freq)


# In[64]:


# Scale the data to between 0 and 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series_data_interpolated)

scaled_df = pd.DataFrame(scaled_data, columns=time_series_data_interpolated.columns, index=time_series_data_interpolated.index)


# In[65]:


plt.plot(scaled_df.loc['2018-01-01 00:00:00':'2018-12-31 23:50:00']['LV ActivePower (kW)'])


# In[84]:


# # Baby steps: We start with a month of data and predict the last three days
first_month_scaled_data = scaled_df.loc['2018-01-01 00:00:00':'2018-1-31 23:50:00']

print(first_month_scaled_data.shape)

train_timesteps = int(60*24*28/10) # number of 10 minute timestamps in 28 days

# Train + Val
train = first_month_scaled_data[:train_timesteps]

print(train.shape)
print(val.shape)


# In[85]:


plt.plot(train['LV ActivePower (kW)'])


# In[98]:


# Very simple vanilla time series model on TPU

lookback_window = int(7*24*60/10) # number of 10 minute intervals in 10 days
print(lookback_window)

features = train[['Wind Speed (m/s)', 'Wind Direction (°)']].values # np array
predict = train['LV ActivePower (kW)'].values # np array

val_features = first_month_scaled_data[['Wind Speed (m/s)', 'Wind Direction (°)']][train_timesteps-lookback_window+1:train_timesteps+1].values
val_predict = first_month_scaled_data['LV ActivePower (kW)'][train_timesteps-lookback_window+1:train_timesteps+1].values

print("Feature shape:" + str(features.shape))
print("Predict shape:" + str(predict.shape))

x, y = [], []

# x_val, y_val = [], []

# Sliding window time series
for i in range(lookback_window, len(train)):
  x.append(features[i - lookback_window: i])
  y.append(predict[i])

x = np.array(x)
y = np.array(y)

# x_val = np.array(x_val)
# y_val = np.array(y_val)

print("Input shape: " + str(x.shape))
print("Output shape: " + str(y.shape))

print("Input val shape: " + str(val_features.shape))
print("Output val shape: " + str(val_predict.shape))


# In[91]:


# GPT

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Input Layer
input_layer = Input(shape=(lookback_window, 2))

# Initial Conv Layer
tensor = Conv1D(filters=16, kernel_size=11, activation='relu', padding='causal')(input_layer)

# Dilated Conv Layers
dilation_rates = [1, 2, 4, 8]
for dilation_rate in dilation_rates:
    tensor = Conv1D(filters=16, kernel_size=16, dilation_rate=dilation_rate,
               activation='relu', padding='causal')(tensor)

# Dropout Layer
tensor = Dropout(rate=0.0)(tensor)  # Assuming dropout rate is not specified

# Another Conv Layer
tensor = Conv1D(filters=16, kernel_size=16, activation='relu', padding='causal')(tensor)

tensor = Flatten()(tensor)

# Output Dense Layer
output_layer = Dense(1, activation='linear')(tensor)

# Build the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Summary of the model
model.summary()



# In[30]:


m.compile('adam', 'mae')
m.summary()


# In[89]:


print('Train...')

model.fit(x, y, epochs=10, verbose=1)


# In[80]:


val_features.shape


# In[90]:


model.predict(val_features)


# In[78]:


p = np.clip(model.predict(val_features), 0, 1)


# In[24]:


with open('./predict_json.npy', 'rb') as f: 
    p_upload = np.load(f)


# In[26]:


p_clipped = np.clip(p_upload, 0, 1)


# In[77]:


plt.clf()

plt.plot(p)
plt.plot(y)
plt.title('Monthly wind power generated (kW)')
plt.legend(['predicted', 'actual'])


# In[ ]:




