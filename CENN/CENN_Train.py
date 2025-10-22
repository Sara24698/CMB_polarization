#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jose Manuel Casas: casasjm@uniovi.es
CENN paper: https://arxiv.org/abs/2205.05623: 
"""

# This module defines the Cosmic microwave background
# extraction neural network (CENN) architecture and trains it
# For using it: python CENN_Execute.py <Number of GPU>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Activation, Input
import os
import argparse
from tensorflow.keras.callbacks import EarlyStopping


from CENN_Input import reading_the_data

print (tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


##############################################################################

# Original hyperparameters set

# learning_rate = 0.05
# batch_size = 32
# num_epochs = 500
# regularizer = keras.regularizers.l2(0.00001)
# activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)
# optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
# loss = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)

##############################################################################

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('GPU', type=str, help = 'Number of GPU you want to use')
    args = parser.parse_args()
    
    return args

def error(y_pred, y_true):
    
    # We use the MSE error along the paper

    return K.mean(K.square(y_pred - y_true), axis=-1)

def build_model_Conv(learning_rate):
    
  regularizer = keras.regularizers.l2(0.000001)
  activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)
  channels_order = 'channels_last'
  
  # Patches of 256x256 pixels and 3 frequency input channels
  
  inputs = Input(shape=(256, 256, 3)) 
  
  conv1 = Conv2D(filters = 8, kernel_size = 9, strides = 2, padding='same', activation=None, data_format=channels_order, 
                kernel_regularizer = regularizer)(inputs)
  
  conv1_activation_function = activation_function(conv1)

  conv2 = Conv2D(16, 9, 2, 'same', activation=None, data_format=channels_order, kernel_regularizer = regularizer)(conv1_activation_function)
  
  conv2_activation_function = activation_function(conv2)
  
  conv3 = Conv2D(64, 7, 2, 'same', activation=None, data_format=channels_order, kernel_regularizer = regularizer)(conv2_activation_function)
  
  conv3_activation_function = activation_function(conv3)
  
  conv4 = Conv2D(128, 7, 2, 'same', activation=None, data_format=channels_order, kernel_regularizer = regularizer)(conv3_activation_function)
  
  conv4_activation_function = activation_function(conv4)
  
  #conv5 = Conv2D(256, 5, 2, 'same', activation=None, data_format=channels_order, kernel_regularizer = regularizer)(conv4_activation_function)
  
  #conv5_activation_function = activation_function(conv5)
  
  #conv6 = Conv2D(512, 3, 2, 'same', activation=None, data_format=channels_order, kernel_regularizer = regularizer)(conv5_activation_function)

  #conv6_activation_function = activation_function(conv6)

  #deconv1 = Conv2DTranspose(256, 3, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(conv6_activation_function)
  
  #deconv1_activation_function = activation_function(deconv1)
  
  #add1 = tf.keras.layers.Concatenate(axis=3)([conv5, deconv1_activation_function])
  
  #deconv2 = Conv2DTranspose(128, 5, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(add1)
  
  #deconv2_activation_function = activation_function(deconv2)
  
  #add2 = tf.keras.layers.Concatenate(axis=3)([conv4, deconv2_activation_function])
  
  deconv3 = Conv2DTranspose(64, 7, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(conv4_activation_function)#(add2)
  
  deconv3_activation_function = activation_function(deconv3)
  
  add3 = tf.keras.layers.Concatenate(axis=3)([conv3, deconv3_activation_function])
  
  deconv4 = Conv2DTranspose(16, 7, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(add3)
  
  deconv4_activation_function = activation_function(deconv4)
  
  add4 = tf.keras.layers.Concatenate(axis=3)([conv2, deconv4_activation_function])
  
  deconv5 = Conv2DTranspose(8, 9, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(add4)
  
  deconv5_activation_function = activation_function(deconv5)
  
  add5 = tf.keras.layers.Concatenate(axis=3)([conv1, deconv5_activation_function])
  
  deconv6 = Conv2DTranspose(1, 9, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(add5)
  
  deconv6_activation_function = activation_function(deconv6)
  
  model = tf.keras.Model(inputs, deconv6_activation_function)
  
  optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
  loss = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
  
  model.compile(loss="mean_squared_error",
                optimizer=optimizer,
                metrics=[loss])

  return model

# Añadir en la sección de imports:
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Dentro de la función `train`, al final de la definición:
def train(learning_rate, batch_size, num_epochs, test_frequency, Patch_Size, Filtro):
    train_file_path = './Train_'+Filtro+'.h5'
    test_file_path = './Test_'+Filtro+'.h5'
 
    inputs_test, labels_test, inputs_train, labels_train = reading_the_data(train_file_path, test_file_path, Patch_Size)
    
    model = build_model_Conv(learning_rate)

    model.summary()

    Checkpoint = keras.callbacks.ModelCheckpoint('Models_'+Filtro+'/'+train_file_path[14:-3]+'_checkpoint-{val_loss:.5f}-{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
    Best = keras.callbacks.ModelCheckpoint('Models_'+ Filtro+'/'+train_file_path[14:-3]+'_Red_'+ Filtro +'.h5', monitor='val_loss', save_best_only=True)
    EarlyStop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Entrenamiento con callbacks
    history = model.fit(inputs_train, labels_train, batch_size = batch_size, shuffle=True,
              epochs = num_epochs, verbose = 1, validation_freq = test_frequency,
              validation_data = (inputs_test, labels_test), callbacks=[Checkpoint, Best, EarlyStop])

    # Evaluación
    results = model.predict(inputs_test)
    loss_error = error(results, labels_test)

    # Plot del historial de entrenamiento
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training history')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Models_' + Filtro + '/training_history.png')
    plt.close()

#if __name__ == "__main__":
    #args = parse_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU
    #train(batch_size, num_epochs, test_frequency, train_file_path, test_file_path, Patch_Size, Filtro)
