#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and functions
# 
# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Softmax, Multiply, Lambda, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import os
import sys


# In[2]:


def make_dir(path): # Function to make adirectory to save files in
    cwd = os.getcwd()
    abs_path = cwd + path


    if not os.path.exists(abs_path):
        try: 
            os.makedirs(abs_path)
            print(abs_path)
        except OSError:return 0
        else: return 1
    else: return 2


# In[3]:


# Declaring Variables
num_epochs = 2000000
load_adam_maxval = 'SavedModels/IndVal/OptiStudies/model_Adam'
load_sgdnm_indval = 'SavedModels/IndVal/OptiStudies/model_SGD_NM'





# Filepaths for saving Model Checkpoints:
save_post_dir = 'PostProc/GridSearchCV'


# In[4]:


# Reading data from .csv file
data_dat = pd.read_csv (r'data.csv')
pred_dat = pd.read_csv (r'test.csv')


# In[5]:


# Extracting data into numpy arrays from pandas DataFrames

# Training data
X = pd.DataFrame(data_dat, columns = ['LD'])
X = X.to_numpy()

X_n = pd.DataFrame(data_dat, columns = ['XL'])
X_n = X_n.to_numpy()


y = pd.DataFrame(data_dat, columns = ['Ma'])
y = y.to_numpy()

# Prediction data
X_pred = pd.DataFrame(pred_dat, columns = ['LD'])
X_pred = X_pred.to_numpy()
X_pred_ns = X_pred

X_n_pred = pd.DataFrame(pred_dat, columns = ['XL'])
X_n_pred = X_n_pred.to_numpy()

y_pred = pd.DataFrame(pred_dat, columns = ['Ma'])
y_pred = y_pred.to_numpy()


# In[6]:


# Returns normalised valeus of Mach Number and Maximum Mach Number per L/D ratio as two arrays

def get_norm_y(y,c): 
    num = int(y.shape[0] / c)
    max_val = np.zeros(c*num)
    y_norm = np.zeros([num*c])
    for i in range(num):
        max_val[i*c:(i+1)*c] = max(y[i*c:(i+1)*c])
        for j in range(c):
            y_norm[i*c + j] = y[i*c+j]/max_val[i]
    return y_norm, max_val
    
y_norm, max_y = get_norm_y(y, 201)
y_pred_norm, y_pred_max = get_norm_y(y_pred, 201)

# Returns Input set fro training the Max Mach Numbers

def get_x(x,c):
    x_n = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        x_n[i] = x[i]
    
    return x_n

X_max = get_x(X,201)
X_pred_max = get_x(X_pred, 201)

X= np.hstack((X_n,X))
X_pred = np.hstack((X_n_pred,X_pred))


# In[7]:


y_comb = np.zeros([y_norm.shape[0],2])
for i in range(y_comb.shape[0]):
    y_comb[i,0]=y_norm[i]
    y_comb[i,1]=max_y[i]
print(y_comb)


# In[8]:


y_comb.shape


# In[9]:


print(X)


# In[ ]:


MaxVal_archi = [20, 20, 20, 1]
IndVal_archi = [20, 20, 20, 20, 20, 1]

def get_LD(x):
    x_new = x[0,1:]
    return x_new[:,np.newaxis]


def build_model(MaxVal_archi, IndVal_archi):

    Input_layer = Input(shape=[2,], name='Input')

    '''MaxVal_input = Lambda(get_LD, name='MaxVal_input', output_shape=(None, 1))(Input_layer)
    '''
    """    MaxVal_layer = []
    for i,node in enumerate(MaxVal_archi):
        if i==0:
            MaxVal_layer.append(Dense(node, name='MaxVal_layer%d' %(i+1), activation='relu')(MaxVal_input))
        elif i==len(MaxVal_archi)-1:
            MaxVal_layer.append(Dense(1, name='MaxVal_Final_layer', activation='relu')(MaxVal_layer[i-1])) 
        else:
            MaxVal_layer.append(Dense(node, name='MaxVal_layer%d' %(i+1), activation='relu')(MaxVal_layer[i-1]))
    """
    IndVal_layer = []
    for i, node in enumerate(IndVal_archi):
        if i==0:
            IndVal_layer.append(Dense(node, name='IndVal_layer%d' %(i+1), activation='relu')(Input_layer))
        elif i==len(IndVal_archi)-1:
            IndVal_layer.append(Dense(1, name='IndVal_Final_layer', activation='relu')(IndVal_layer[i-1])) 
        else:
            IndVal_layer.append(Dense(node, name='IndVal_layer%d' %(i+1), activation='relu')(IndVal_layer[i-1]))
            # IndVal_layer.append(Dropout(0.25)(IndVal_layer[i]))

            
#     Multiplication_layer = Multiply(name='Multiplication_layer')([IndVal_layer[len(IndVal_archi)-1], MaxVal_layer[len(MaxVal_archi)-1]])
    
#     PreFinal_layer1 = Dense(20, name='PreFinal_layer1', activation='relu')(Multiplication_layer)
#     PreFinal_layer2 = Dense(20, name='PreFinal_layer2', activation='relu')(PreFinal_layer1)
#     PreFinal_layer3 = Dense(20, name='PreFinal_layer3', activation='relu')(PreFinal_layer2)
    
#     Final_layer = Dense(1, name='Final_layer', activation='linear')(PreFinal_layer3)
    
    model = Model(inputs= [Input_layer], outputs= IndVal_layer[len(IndVal_archi)-1])
    return model

model = build_model(MaxVal_archi,IndVal_archi)


# In[ ]:


keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
# print(model.summary())


# In[ ]:


print(model.summary())


# In[ ]:


model.compile(loss='mse', optimizer='RMSprop', metrics=['mape'])


# In[ ]:


early_stopping_callback = EarlyStopping(monitor='val_mape', 
                                        patience=20,
                                        min_delta= 1e-3,
                                        restore_best_weights=True,
                                        mode='auto')


# In[ ]:


##### Using sklearn to split the data into train and test sets with shuffling ####
X_train, X_test, y_train_comb, y_test_comb = train_test_split(X, y_norm, test_size=0.2, random_state=42, shuffle=True)


# y_norm_train = y_train_comb[:,0]
# y_max_train = y_train_comb[:,1]
# y_norm_test = y_test_comb[:,0]
# y_max_test = y_test_comb[:,1]


# In[ ]:


history = model.fit(X_train,
                    y_train_comb,
                    batch_size = 2,
                    epochs=200000,
                    verbose=True,
                    validation_data=(X_test,y_test_comb),
                    callbacks=[early_stopping_callback]
                   )

model.save('temp_models/SiameseModel')


# In[ ]:


plt.plot(history.history['loss'])


# In[ ]:


pred = model.predict(X_pred)


# In[ ]:


print(pred)


# In[ ]:


for i in range(int(pred[1].shape[0]/201)):
    print(X_pred_max[i*201])


# In[ ]:


for i in range(int(pred[1].shape[0]/201)):
    
    print(abs(pred[1][i*201]-y_pred_max[i*201])/y_pred_max[i*201])


# In[ ]:


print(pred[1].shape[0]/201)


# In[ ]:


nrows=int(X_pred.shape[0]/201)
fig3 = plt.figure()
fig, axes = plt.subplots(nrows)
fig.set_size_inches(6,39)
for j in range(nrows):
    axes[j].plot(X_n_pred[:201],
                  pred[j*201:(j+1)*201],
                  'r',
                  label='Prediction')
    axes[j].plot(X_n_pred[j*201:(j+1)*201], y_pred_norm[j*201:(j+1)*201], 'g', label='Actual')
    axes[j].set(xlabel="Normalised x-coordinate", ylabel="Mach Number")
    axes[j].set_title('L/D Ratio = %.1f' %(X_pred_max[j]), fontsize=11)
    axes[j].legend(loc="upper right")
    axes[j].set_xlim((0,1))
    axes[j].set_ylim((0,5))
    axes[j].set_aspect(0.2)

fig = plt.gcf()
plt.tight_layout(pad=0.25, h_pad=1.25, w_pad=0.25, rect=None)
# plt.savefig(save_post_dir + 'cumilative_ind_%s_predictions.png' %(opti_name), dpi=500)
plt.show(fig3)
plt.close(fig3)


# In[ ]:


print(np.zeros([1,2]))


# In[ ]:


print(X_pred.shape)


# In[ ]:




