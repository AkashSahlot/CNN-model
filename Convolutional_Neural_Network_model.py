#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import tensorflow as tf
import statsmodels.api as sm
from tensorflow import keras


# In[2]:


df= pd.read_excel(r"C:\Users\akashsahlot\Documents\yamnaya\Yamnaya Project File\cleaning 2021 4.xlsx")
df


# In[3]:


dataset = df.copy()
dataset.loc[(dataset['Day'] == 'Mon'), 'Day'] = 1
dataset.loc[(dataset['Day'] == 'Tue'), 'Day'] = 2
dataset.loc[(dataset['Day'] == 'Wed'), 'Day'] = 3          
dataset.loc[(dataset['Day'] == 'Thu'), 'Day'] = 4
dataset.loc[(dataset['Day'] == 'Fri'), 'Day'] = 5
dataset.loc[(dataset['Day'] == 'Sat'), 'Day'] = 6
dataset.loc[(dataset['Day'] == 'Sun'), 'Day'] = 7

dataset['Day'] = dataset['Day'].astype(str).astype(int)
# Remove bridge and holiday
dataset = dataset.drop(['Bridge'], axis = 1)
dataset = dataset.drop(['Holiday'], axis = 1)
colname = dataset.columns
headers = colname.tolist()
other_features = []
other_features = headers[6:8]
from datetime import date
# Train Test Split Dataset
dataframe = dataset.reset_index(drop=True)
dtlist = dataframe.iloc[:,1]
pdate = []
for cdt in dtlist:
  spdt = cdt.split("/")
  d = date(int(spdt[2]), int(spdt[1]), int(spdt[0]))
  p = pd.to_datetime(d)
  pdate.append(p)

stime = dataframe.iloc[:,2]
pt = pd.to_datetime(stime, errors="coerce", format="%H:%M").dt
ptime = pt.time
dataframe.insert(loc=1, column='date', value=pdate)
dataframe.insert(loc=2, column='time', value=ptime)
# Drop custom date and time column 
dataframe = dataframe.drop(['Date'], axis=1)
dataframe = dataframe.drop(['hour'], axis=1)

data = dataframe

data.insert(loc=1, column='datetime', value=data.date.astype(str)+' '+data.time.astype(str))
data.insert(loc=1,column = 'year', value =data['date'].dt.year)
data.insert(loc=2,column = 'month', value =data['date'].dt.month)
data.insert(loc=3,column = 'dt', value =data['date'].dt.day)
data.insert(loc=4,column = 'hour', value =data['date'].dt.hour)
data = data.reset_index()

data['day'] = data['date'].apply(lambda x :1 if x.weekday() == 6 else x.weekday() + 2)
print("day :", data['day'])
data['season'] = data['date'].apply(lambda mdt: (mdt.month%12 + 3)//3)
data['working_day'] = data['day'].apply(lambda x: 0 if x in [7, 1] else 1)
data['week_end'] = data['day'].apply(lambda x: 1 if x in [7, 1] else 0)

trend = -1
data['idx'] = [i for i in data.index.values]
# Trend and idx_jr
# Helper Functions
def set_trend(x, gap):
  global trend
  date = x['date']
  # print("set trend :", x['date'], type(x['date']))
  # date = datetime.strptime(x['date'], "%Y-%m-%d")
  # print("set trend :", date, type(date))
  if x['idx'] == 0:
    trend = (date - pd.to_datetime('01-01-{}'.format(date.year))).days * gap
  else:
    trend = 0 if [date.day, date.month, date.hour, date.minute, date.second] == [1, 1, 0, 0,0] else trend + 1
  return trend

def set_idx_jr(x):
  global i
  date = x['date']

  if x['idx'] == 0:
      i = (date - pd.to_datetime('{}-{}-{}'.format(date.day, date.month, date.year),
                                  format='%d-%m-%Y')).seconds / 1800
      print()
      indexof_hour = (date - pd.to_datetime('{}-{}-{}'.format(date.day, date.month, date.year),
                                  format='%d-%m-%Y')).seconds / 1800
      # print("i = (date - pd.to_datetime('{}-{}-{}'.format(date.day, date.month, date.year), format='%d-%m-%Y')).seconds / 1800:",  indexof_hour)
  else:
      i = 0 if date.hour + date.minute + date.second == 0 else i + 1
      # print("i = 0 if date.hour + date.minute + date.second == 0 else i + 1:",i)
  return i

def func_j(data, j,x, x_header,gap ):
  """func_j7, func_j14, func_j21 merge into one """
  try:
    # print("func-J OUTPUT :", data.loc[x['idx'] - j * gap, x_header])
    return data.loc[x['index'] - j * gap, x_header]

  except KeyError:
    return float('nan')


# In[4]:


gap = 24
data['trend'] = data.apply(lambda x: set_trend(x, gap), axis=1)
# print("set trend :", data["trend"])
i = -1
gap=24
data['idx_jr'] = data.apply(lambda x: set_idx_jr(x), axis=1)
data['consoj-7'] = data.apply(lambda x: func_j(data, 7,x,"power", gap), axis=1)
data['consoj-7'] = data['consoj-7'].fillna(0)
data.fillna(0)
data = data.iloc[168:]
#Data Normalise
scaling=StandardScaler()
scaling.fit_transform(data[['power','variable 2']])
X = data


# In[5]:


X


# In[6]:


# Drop column from X
dt_time = X["datetime"]
X = X.drop(["power", "datetime","date", "time"], axis=1)
X.replace(np.nan, )

Y = data["power"]

# Get last index for the last availbe power values in given dataset
last_empty_power_index = dataframe["power"].last_valid_index()
print(last_empty_power_index)

test_period = 7*24

x_known = X[X.index <= last_empty_power_index]

x_unknown = X.iloc[len(x_known):len(X)]
y_known = Y[Y.index <= last_empty_power_index]
y_unknown = Y.iloc[len(y_known):len(Y)]

#Test-Train Dataset
x_train_n = x_known.iloc[0:len(x_known)-test_period]
y_train_n = y_known.iloc[0:len(y_known)-test_period]

x_test = x_known.iloc[len(x_known)-test_period: len(x_known)]
y_test = y_known.iloc[len(y_known)-test_period: len(y_known)]


# In[7]:


X


# In[8]:


len(x_train_n)


# In[9]:


len(x_test)


# In[10]:


len(y_train_n)


# In[11]:


len(y_test)


# In[12]:


x_valid,x_train=x_train_n[684:],x_train_n[:684]
y_valid,y_train=y_train_n[684:],y_train_n[:684]
x_test=x_test


# In[13]:


# Model Architecture
np.random.seed(42)
tf.random.set_seed(42)


# In[14]:


x_train_n.shape[1]


# In[15]:


# n_features = 1,


# In[16]:


# X_train_n=x_train_n.reshape((x_train_n.shape[0],x_train_n.shape[1],n_features))


# In[17]:


# Model structuring and Compilation 

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(24, 3, strides=1, padding='valid', activation='relu', input_shape=(x_train_n.shape[1],1))) # 1D convolutional layer with 32 filters, a kernel size of 3, and a ReLU activation function
model.add(keras.layers.MaxPooling1D(2))
# model.add(keras.layers.Conv1D(64, 3, activation='swish'))
# model.add(keras.layers.MaxPooling1D(2))
# model.add(keras.layers.Conv1D(128, 3, activation='relu'))
# model.add(keras.layers.MaxPooling1D(2))
# model.add(keras.layers.Conv1D(64, 3, activation='swish'))
# model.add(keras.layers.MaxPooling1D(2))          
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(150, activation='relu'))
model.add(keras.layers.Dense(150, activation='relu'))
model.add(keras.layers.Dense(1))
model.summary()


# In[18]:


# Compilation
model.compile(loss="huber",
             optimizer=keras.optimizers.Adam(
             learning_rate=0.001,
             beta_1=0.9,
             beta_2=0.999,
             epsilon=1e-03,
             ema_momentum=0.99,
             jit_compile=True,
             name="Adam"),
             metrics=[['mae']])


# In[19]:


# checkpoint_cb=keras.callbacks.ModelCheckpoint("early_stop_model.h5",save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=40,restore_best_weights=True)
model_history= model.fit(
    x_train,
    y_train,
    epochs=488,
    batch_size=32,
    validation_data=(x_valid,y_valid),
    callbacks=[early_stopping_cb]
)


# In[20]:


# Evaluation
mae_test=model.evaluate(x_test,y_test)


# In[21]:


#losses values
model_history.history


# In[22]:


pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
# plt.gca().set_ylim(0,1)
plt.show()


# In[23]:


# Prediction
x_new=x_unknown
y_pred=model.predict(x_new)
plt.plot(y_pred)
plt.show()


# In[24]:


y_pred


# In[25]:


y_pred_1d = y_pred.flatten() #flatten `array_2d`
y_pred_1d


# In[26]:


plt.plot(Y)
plt.plot(y_pred_1d)
plt.show()


# In[27]:



# pd.DataFrame(y_pred_1d).to_excel("cnn_2017_full", sheet_name='power_prediction')


# In[28]:



dlen=len(df)
future_data=df.iloc[dlen-168:]
future_data['power_prediction']=y_pred_1d
future_data.to_excel("cnn_2021-4.xlsx", sheet_name='sheet1')


# In[ ]:




