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
X = pd.DataFrame(data_dat, columns = ['XL','LD'])
X = X.to_numpy()

X_n = pd.DataFrame(data_dat, columns = ['XL'])
X_n = X_n.to_numpy()


y = pd.DataFrame(data_dat, columns = ['P'])
y = y.to_numpy()

# Prediction data
X_pred = pd.DataFrame(pred_dat, columns = ['XL','LD'])
X_pred = X_pred.to_numpy()


y_pred = pd.DataFrame(pred_dat, columns = ['P'])
y_pred = y_pred.to_numpy()

# Using sklearn to split the data into train and test sets with shuffling ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Define a function to build a SN using KerasTuner syntax
def build_model(hp):
    Input_layer = Input(shape=[2,], name='Input')
    hl = hp.Int('hl', min_value =2, max_value=20, step=1)
    units = hp.Int('MaxVal_units' ,min_value=1,max_value=40,step=2)
    layers = []
    for i in range(hl):
        if i==0: # First layer in MaxVal fork
            layers.append(Dense(units,name='layer%d' %(i+1),
                                      # kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2),
                                      activation ='relu')(Input_layer))
        elif i==hl-1: # Last layer in MaxVal fork. *Activation must be linear*
            layers.append(Dense(1,
                                # kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2), 
                                name='Final_layer', activation='linear')(layers[i-1]))
        else: # For intermediate layers
            layers.append(Dense(units, name='layer%d' %(i+1),
                                # kernel_regularizer=L1L2(l1=MaxVal_L1, l2=MaxVal_L2),
                                activation='relu')(layers[i-1]))

    # Building the model with all connections
    model = Model(inputs= [Input_layer], outputs= [layers[hl-1]])

    # Defining Optimizer


    # Compiling the model
    Nadam = tf.keras.optimizers.Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    # Compile model 
    model.compile(loss='mse', optimizer=Nadam, metrics=['mape'])

    return model

hp = kt.HyperParameters()
model = build_model(hp)

tuner = Hyperband(
    build_model,
    objective=kt.Objective('val_mape', direction='min'),
    max_epochs=2000,
    hyperband_iterations=2,
    directory='HyperBandTrials',
    project_name='PressureOpti_hl'
    )

# Defining the Early Stopping Function
early_stopping_callback = EarlyStopping(monitor='val_mape',
                                        patience=500,
                                        min_delta= 1e-4,
                                        restore_best_weights=True,
                                        mode='auto',
                                        verbose=True)

tuner.search_space_summary()

print(model.summary())

tuner.search(X_train, y_train,
             epochs=2000,
             validation_data=(X_test,y_test),
             callbacks=[early_stopping_callback])

models = tuner.get_best_models(num_models=2)
models[0].save('/mnt/IMP/Work/Thesis/NeuralNetwork/DeepLearning-RarefiedFlows/SavedModels/Pressure/HyperBand_hl/Best_Model_1')
models[1].save('/mnt/IMP/Work/Thesis/NeuralNetwork/DeepLearning-RarefiedFlows/SavedModels/Pressure/HyperBand_hl/Best_Model_2')
tuner.results_summary()
