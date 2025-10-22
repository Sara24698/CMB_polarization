#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PEP 8 Style

"""
@author: Jose Manuel Casas: casasjm@uniovi.es
CENN paper: https://arxiv.org/abs/2205.05623: 
"""

# This module validates CENN. For using it, python CENN_Execute.py <Number of GPU>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import os
import h5py
import argparse

print (tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

import numpy as np

from CENN_Input import reading_the_data, reading_validation_set, reading_min_max_values, reading_model, reading_validation_set_no_normalized

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('GPU', type=str, help = 'Number of GPU you want to use')

    args = parser.parse_args()
    
    return args

def denormalize(results, maximo, minimo, Patch_Size):
    
    renormalize_labels_test = np.zeros(
        [len(results), Patch_Size, Patch_Size, 1])
           
    for i in range(len(results)):
        
        renormalize_labels_test[i,:,:,0] = (results[i,:,:,0]*(maximo[i] - minimo[i])) + minimo[i]       


    return renormalize_labels_test


def save_data(results_denorm, test_outputs_denorm, name_output_file):
    
    # Outputs_CENN is a file containing the outputs (net) and the true CMB maps (sim)
    # in the validation dataset 
    
    with h5py.File(name_output_file, 'w') as f:
        
        f['net'] = results_denorm
        f['sim'] = test_outputs_denorm

def error(y_pred, y_true):

    return K.mean(K.square(y_pred - y_true), axis=-1)

def quick_execute(Filtro, validation_file_path, Patch_Size, model, maximo, minimo, test_inputs=0, test_outputs=0, load_data_bool=False):
    
    labels_validation_no_normalized = reading_validation_set_no_normalized(validation_file_path)
    
    results = model.predict(test_inputs)


    results_denorm = denormalize(results, maximo, minimo, Patch_Size)

    save_data(results_denorm, labels_validation_no_normalized, name_output_file='Outputs_CENN_'+Filtro+'.h5')

def execute(Patch_Size, Filtro):
    
    # normalised validation dataset
    Model_file_path = './Models_'+Filtro+'/_Red_'+Filtro+'.h5'
    validation_file_path = './Validation_'+Filtro+'.h5'
    
    inputs_validation, labels_validation, maximo, minimo = reading_validation_set(validation_file_path, Patch_Size)

    model = keras.models.load_model(Model_file_path)
    model.summary()

    quick_execute(Filtro, validation_file_path, Patch_Size, model, maximo, minimo, test_inputs=inputs_validation, test_outputs=labels_validation, load_data_bool=False)
