# Libraries to import
import pandas as pd
import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Softmax, Multiply, Lambda, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import L1L2
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
from math import sqrt
from matplotlib import pyplot as plt
import os
import sys

# Reading data from .csv file
data_dat = pd.read_csv (r'data.csv')
pred_dat = pd.read_csv (r'test.csv')

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

# Returns normalised valeus of Mach Number and Maximum Mach Number per L/D ratio as two arrays

def get_norm_y(y,c): 
    num = int(y.shape[0] / c)
    max_val = np.zeros(c*num)
    y_norm = np.zeros([num*c])
    for i in range(num):
        max_val[i*c:(i+1)*c] = max(y[i*c:(i+1)*c])
        for j in range(c):
            y_norm[i*c + j] = y[i*c+j]/max_val[i*c+j]
    return y_norm, max_val
    
y_norm, max_y = get_norm_y(y, 201)
y_pred_norm, max_y_pred = get_norm_y(y_pred, 201)

# Returns Input set fro training the Max Mach Numbers

def get_x(x,c):
    x_n = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        x_n[i] = x[i]
    
    return x_n

X_max = get_x(X,201)
X_pred_max = get_x(X_pred, 201)
# Concatenated Xcordinates and L/D Ratios
X= np.hstack((X_n,X))
X_pred = np.hstack((X_n_pred,X_pred))

y_comb = np.zeros([y_norm.shape[0],2])
for i in range(y_comb.shape[0]):
    y_comb[i,0]=y_norm[i]
    y_comb[i,1]=max_y[i]

y_pred_comb = np.zeros([y_pred_norm.shape[0],2])
for i in range(y_pred_comb.shape[0]):
    y_pred_comb[i,0]=y_pred_norm[i]
    y_pred_comb[i,1]=max_y_pred[i]

# Using sklearn to split the data into train and test sets with shuffling ####
X_train, X_test, y_train_comb, y_test_comb = train_test_split(X, y_comb, test_size=0.2, random_state=42, shuffle=True)

def get_LD(x): # Function to separate L/D Ratio for the Lambda Layer.
    x_new = x[:,1]
    return x_new[:,np.newaxis]

# Define a function to build a SN using KerasTuner syntax
def build_model(hp):
    Input_layer = Input(shape=[2,], name='Input')
    # Lambda layer to separately use it as only L/D ratio input for a fork in Network.
    MaxVal_input = Lambda(get_LD, name='MaxVal_input', output_shape=(1,))(Input_layer)
    # Defining the MaxVal fork of SN    
    MaxVal_layer = []
    # hl = hp.Int('hl', min_value =2, max_value=15, step=1)
    hl = 14
    # MaxVal_units = hp.Int('MaxVal_units' ,min_value=1,max_value=10,step=2)
    MaxVal_units = 7
    MaxVal_L1 = hp.Choice('MaxVal_L1', [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    MaxVal_L2 = hp.Choice('MaxVal_L2', [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    for i in range(hl):
        if i==0: # First layer in MaxVal fork
            MaxVal_layer.append(Dense(MaxVal_units,name='MaxVal_layer%d' %(i+1),
                                      kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2),
                                      activation='relu')(MaxVal_input))
        elif i==hl-1: # Last layer in MaxVal fork. *Activation must be linear*
            MaxVal_layer.append(Dense(1,
                                      kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2),
                                      name='MaxVal_Final_layer', activation='linear')(MaxVal_layer[i-1]))
        else: # For intermediate layers
            MaxVal_layer.append(Dense(MaxVal_units, name='MaxVal_layer%d' %(i+1),
                                      kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2),
                                      activation='relu')(MaxVal_layer[i-1]))
            
    # Defining the IndVal fork of Neural Network
    IndVal_L1 = hp.Choice('IndVal_L1' , [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    IndVal_L2 = hp.Choice('IndVal_L2' , [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    IndVal_layer = []
    # IndVal_units = hp.Int('IndVal_units', min_value=1, max_value=37, step=3)
    IndVal_units = 22
    for i in range(hl):
        if i==0: # First layer in MaxVal fork
            IndVal_layer.append(Dense(IndVal_units, name='IndVal_layer%d' %(i+1),
                                      kernel_regularizer=L1L2(l1=IndVal_L1, l2=IndVal_L2), 
                                      activation='relu')(Input_layer))
        elif i==hl-1: # Last layer in MaxVal fork. *Activation must be linear*
            IndVal_layer.append(Dense(1,
                                      kernel_regularizer=L1L2(l1=IndVal_L1, l2=IndVal_L2), 
                                      name='IndVal_Final_layer', activation='linear')(IndVal_layer[i-1])) 
        else:# For intermediate layers
            IndVal_layer.append(Dense(IndVal_units, name='IndVal_layer%d' %(i+1),
                                      kernel_regularizer=L1L2(l1=IndVal_L1, l2=IndVal_L2), 
                                      activation='relu')(IndVal_layer[i-1]))
    # Building the model with all connections
    model = Model(inputs= [Input_layer], outputs= [IndVal_layer[hl-1], MaxVal_layer[hl-1]])

    # Defining Optimizer
    Adam = tf.keras.optimizers.Adam(learning_rate=0.0005,
                                    # learning_rate=hp.Choice('Adam_lr', [1e-0,1e-1,1e-2,5e-3,1e-3,8e-4,5e-4,3e-4,1e-4,1e-5]),
                                    name="Adam")

    
    # Compiling the model
    model.compile(loss='mse', optimizer=Adam, metrics=['mape'])
       
    return model

hp = kt.HyperParameters()
model = build_model(hp)

tuner = Hyperband(
    build_model,
    objective=kt.Objective('val_IndVal_Final_layer_mape', direction='min'),
    max_epochs=400,
    hyperband_iterations=5,
    directory='HyperBandTrials',
    project_name='SN!Mult-valInd-Adam-KR-HPT'
    )

# Defining the Early Stopping Function
early_stopping_callback = EarlyStopping(monitor='val_IndVal_Final_layer_mape', 
                                        patience=500,
                                        min_delta= 1e-3,
                                        restore_best_weights=True,
                                        mode='auto',
                                        verbose=True)

tuner.search_space_summary()

print(model.summary())

tuner.search(X_train, (y_train_comb[:,0], y_train_comb[:,1]),
             epochs=5000,
             validation_data=(X_test, (y_test_comb[:,0], y_test_comb[:,1])),
             callbacks=[early_stopping_callback])

models = tuner.get_best_models(num_models=2)
models[0].save('/mnt/IMP/Work/Thesis/NeuralNetwork/DeepLearning-RarefiedFlows/SavedModels/HyperBand/Adam/Best_Model_1/')
models[1].save('/mnt/IMP/Work/Thesis/NeuralNetwork/DeepLearning-RarefiedFlows/SavedModels/HyperBand/Adam/Best_Model_2/')
tuner.results_summary()