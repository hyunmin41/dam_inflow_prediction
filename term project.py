#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from tqdm import tqdm
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim


# In[2]:


df = pd.read_csv("/Users/kimhyunmin/dam_soyanggang.csv")


# In[3]:


df.info()


# In[3]:


df.columns = ["Date","inflow","rainfall","temperature","windspeed","humidity","level1","level2"]


# In[4]:


feature = df[["rainfall","temperature","windspeed","humidity","level1","level2"]]
label = df[["inflow"]]

feature_cols = ["rainfall","temperature","windspeed","humidity","level1","level2"]
label_cols = ["inflow"]


# In[5]:


from sklearn.preprocessing import MinMaxScaler


feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

feature = feature_scaler.fit_transform(feature)
feature = pd.DataFrame(feature)

label = label_scaler.fit_transform(label)
label = pd.DataFrame(label)


# In[6]:


from sklearn.model_selection import train_test_split,KFold,cross_val_score

kfold=KFold(10)
cvscores = []


# In[7]:


def make_dataset(df, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(df) - window_size):
        feature_list.append(np.array(df.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


# In[8]:


import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


# CNN_LSTM

# In[99]:


import random
seed = 1234
random.seed(seed)

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,LSTM,Dropout,Conv1D, MaxPooling1D, TimeDistributed


for train, test in kfold.split(feature,label):
    
    train_feature, test_feature = feature[0:2674], feature[2673:3654]
    train_label, test_label = label[0:2674], label[2673:3654]
    pred_feature, pred_label = feature[3653:3820], label[3653:3820]
    
    train_feature, train_label = make_dataset(train_feature, train_label, 6)
    test_feature, test_label = make_dataset(test_feature, test_label, 6)
    pred_feature, pred_label = make_dataset(pred_feature, pred_label, 6)
    
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    
    subsequences = 2
    timesteps = x_train.shape[1]//subsequences
    X_train_series_sub = x_train.reshape((x_train.shape[0], subsequences, timesteps, 6)) 
    X_valid_series_sub = x_valid.reshape((x_valid.shape[0], subsequences, timesteps, 6)) 
    test_feature_series_sub = test_feature.reshape((test_feature.shape[0], subsequences, timesteps, 6))
    pred_feature_series_sub = pred_feature.reshape((pred_feature.shape[0], subsequences, timesteps, 6))
    
    loss="mean_squared_error" 
    
    model_cnn_lstm = Sequential()
    model_cnn_lstm.add(TimeDistributed(Conv1D(filters=128, kernel_size=2, 
                                          activation='relu'),
                                   input_shape=(None, X_train_series_sub.shape[2],
                                                X_train_series_sub.shape[3])))

    model_cnn_lstm.add(TimeDistributed(Dropout((0.5))))
    model_cnn_lstm.add(TimeDistributed(Flatten()))
    model_cnn_lstm.add(LSTM(100, activation='tanh'))
    model_cnn_lstm.add(Dropout(0.25))
    model_cnn_lstm.add(Dense(150))
    model_cnn_lstm.add(Dense(1))
    model_cnn_lstm.compile(loss=loss, optimizer="adam", metrics=["mse"])
    model_cnn_lstm.fit(X_train_series_sub, y_train,
                    epochs=10, 
                    verbose=1, 
                    validation_data=(X_valid_series_sub, y_valid), 
                    callbacks=[early_stop, checkpoint])
    scores = model_cnn_lstm.evaluate(test_feature_series_sub, test_label, verbose=0)
    cvscores = (scores * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) 


# In[100]:


test_pred = model_cnn_lstm.predict(test_feature_series_sub)


# In[101]:


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred)) 


# In[102]:


test_pred = label_scaler.inverse_transform(test_pred)
test_label = label_scaler.inverse_transform(test_label)


# In[103]:


pred = model_cnn_lstm.predict(pred_feature_series_sub)
true = pred_label


# In[104]:


RMSE(true, pred)


# In[105]:


pred = label_scaler.inverse_transform(pred)


# In[106]:


true = label_scaler.inverse_transform(pred_label)
true = np.array(true)


# In[107]:


plt.plot(true, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()


# LSTM

# In[72]:


import random
seed = 1234
random.seed(seed)

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,LSTM,Dropout,Conv1D, MaxPooling1D, TimeDistributed

for train, test in kfold.split(feature,label):
    
    train_feature, test_feature = feature[0:2674], feature[2673:3654]
    train_label, test_label = label[0:2674], label[2673:3654]
    pred_feature, pred_label = feature[3653:3820], label[3653:3820]
    
    train_feature, train_label = make_dataset(train_feature, train_label, 10)
    test_feature, test_label = make_dataset(test_feature, test_label, 10)
    pred_feature, pred_label = make_dataset(pred_feature, pred_label, 10)
    
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    
    loss="mean_squared_error" 
    
    model_lstm = Sequential()
    model_lstm.add(LSTM(200, activation="relu", return_sequences=False ,input_shape=(x_train.shape[1],x_train.shape[2])))
    model_lstm.add(Dropout(0.25))
    model_lstm.add(Dense(120))
    model_lstm.add(Dense(60))    
    model_lstm.add(Dense(1))    
    model_lstm.compile(loss=loss, optimizer="adam", metrics=["mse"])
    model_lstm.fit(x_train, y_train,
                    epochs=10, 
                    verbose=1, 
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])
    scores = model_lstm.evaluate(test_feature, test_label, verbose=0)
    cvscores = (scores * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[73]:


test_pred = model_lstm.predict(test_feature)


# In[74]:


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred)) 


# In[75]:


test_pred = label_scaler.inverse_transform(test_pred)
test_label = label_scaler.inverse_transform(test_label)


# In[76]:


pred = model_lstm.predict(pred_feature)
true = pred_label


# In[77]:


RMSE(true, pred)


# In[78]:


pred = label_scaler.inverse_transform(pred)


# In[79]:


true = label_scaler.inverse_transform(pred_label)
true = np.array(true)


# In[80]:


plt.plot(true, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()


# GRU

# In[81]:


import random
seed = 1234
random.seed(seed)

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,LSTM,Dropout,Conv1D, MaxPooling1D, TimeDistributed
from keras.layers.recurrent import GRU

for train, test in kfold.split(feature,label):
    
    train_feature, test_feature = feature[0:2674], feature[2673:3654]
    train_label, test_label = label[0:2674], label[2673:3654]
    pred_feature, pred_label = feature[3653:3820], label[3653:3820]
    
    train_feature, train_label = make_dataset(train_feature, train_label, 10)
    test_feature, test_label = make_dataset(test_feature, test_label, 10)
    pred_feature, pred_label = make_dataset(pred_feature, pred_label, 10)
    
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    
    loss="mean_squared_error" 
    
    model_gru = Sequential()
    model_gru.add(GRU(200, activation="relu", return_sequences=False ,input_shape=(x_train.shape[1],x_train.shape[2])))
    model_gru.add(Dropout(0.25))
    model_gru.add(Dense(150))
    model_gru.add(Dense(60))    
    model_gru.add(Dense(1))    
    model_gru.compile(loss=loss, optimizer="adam", metrics=["mse"])
    model_gru.fit(x_train, y_train,
                    epochs=10, 
                    verbose=1, 
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])
    scores = model_gru.evaluate(test_feature, test_label, verbose=0)
    cvscores = (scores * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[82]:


test_pred = model_gru.predict(test_feature)
test_pred


# In[83]:


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred)) 


# In[84]:


test_pred = label_scaler.inverse_transform(test_pred)
test_label = label_scaler.inverse_transform(test_label)


# In[85]:


pred = model_gru.predict(pred_feature)
true = pred_label


# In[86]:


RMSE(true, pred)


# In[87]:


pred = label_scaler.inverse_transform(pred)


# In[88]:


true = label_scaler.inverse_transform(pred_label)
true = np.array(true)


# In[89]:


plt.plot(true, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()


# RNN

# In[90]:


import random
seed = 1234
random.seed(seed)

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,LSTM,Dropout,Conv1D, MaxPooling1D, TimeDistributed
from keras.layers import SimpleRNN



for train, test in kfold.split(feature,label):
    
    train_feature, test_feature = feature[0:2674], feature[2673:3654]
    train_label, test_label = label[0:2674], label[2673:3654]
    pred_feature, pred_label = feature[3653:3820], label[3653:3820]
    
    train_feature, train_label = make_dataset(train_feature, train_label, 10)
    test_feature, test_label = make_dataset(test_feature, test_label, 10)
    pred_feature, pred_label = make_dataset(pred_feature, pred_label, 10)
    
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    
    loss="mean_squared_error" 
    
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(120, activation="relu", return_sequences=False ,input_shape=(x_train.shape[1],x_train.shape[2])))
    model_rnn.add(Dropout(0.25))
    model_rnn.add(Dense(60))    
    model_rnn.add(Dense(1))    
    model_rnn.compile(loss=loss, optimizer="adam", metrics=["mse"])
    model_rnn.fit(x_train, y_train,
                    epochs=10, 
                    verbose=1, 
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])
    scores = model_rnn.evaluate(test_feature, test_label, verbose=0)
    cvscores = (scores * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[91]:


test_pred = model_rnn.predict(test_feature)
test_pred


# In[92]:


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred))


# In[93]:


test_pred = label_scaler.inverse_transform(test_pred)
test_label = label_scaler.inverse_transform(test_label)


# In[94]:


pred = model_rnn.predict(pred_feature)
true = pred_label


# In[95]:


RMSE(true, pred)


# In[96]:


pred = label_scaler.inverse_transform(pred)
pred


# In[97]:


true = label_scaler.inverse_transform(pred_label)
true = np.array(true)
true


# In[98]:


plt.plot(true, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()

