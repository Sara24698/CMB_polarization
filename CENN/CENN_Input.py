#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PEP 8 Style

"""
@author: Jose Manuel Casas: casasjm@uniovi.es
CENN paper: https://arxiv.org/abs/2205.05623: 
"""

# This module reads train, test and validation files and prepares the train and  
# test images (sources with and without background) to train CENN

import h5py
import numpy as np

def reading_the_data(train_file_path, test_file_path, Patch_Size):
    
    # Train
    
    train_file = h5py.File(train_file_path, 'r')
    

    inputs_train = train_file["M"][:,:,:].astype(np.float32)
    labels_train = train_file["M0"][:,:,:].astype(np.float32)


    
    
    # Scaling train data between 0 and 1
    
    normalize_inputs_train = np.zeros(
            [len(inputs_train), Patch_Size, Patch_Size, 3])
       
    normalize_labels_train = np.zeros(
        [len(labels_train), Patch_Size, Patch_Size, 1])
    
    for i in range(len(inputs_train)):

        min_labels_train_217 = np.min(labels_train[i,:,:,0])
        max_labels_train_217 = np.max(labels_train[i,:,:,0])
    
        min_inputs_train_143 = np.min(inputs_train[i,:,:,0])
        max_inputs_train_143 = np.max(inputs_train[i,:,:,0])
        min_inputs_train_217 = np.min(inputs_train[i,:,:,1])
        max_inputs_train_217 = np.max(inputs_train[i,:,:,1])
        min_inputs_train_353 = np.min(inputs_train[i,:,:,2])
        max_inputs_train_353 = np.max(inputs_train[i,:,:,2])

        
        normalize_inputs_train[i,:,:,0] = (inputs_train[i,:,:,0] - min_inputs_train_143)/(max_inputs_train_143 - min_inputs_train_143)
        normalize_inputs_train[i,:,:,1] = (inputs_train[i,:,:,1] - min_inputs_train_217)/(max_inputs_train_217 - min_inputs_train_217)
        normalize_inputs_train[i,:,:,2] = (inputs_train[i,:,:,2] - min_inputs_train_353)/(max_inputs_train_353 - min_inputs_train_353)

        normalize_labels_train[i,:,:,0] = (labels_train[i,:,:,0] - min_inputs_train_217)/(max_inputs_train_217 - min_inputs_train_217)
        
        
    inputs_train = normalize_inputs_train
    labels_train = normalize_labels_train
  
    
    test_file = h5py.File(test_file_path, 'r')

    inputs_test = test_file["M"][:,:,:].astype(np.float32)
    labels_test = test_file["M0"][:,:,:].astype(np.float32)

    
    
    # Scaling test data between 0 and 1
    
    normalize_inputs_test = np.zeros(
            [len(inputs_test), Patch_Size, Patch_Size, 3])
       
    normalize_labels_test = np.zeros(
        [len(labels_test), Patch_Size, Patch_Size, 1])
    
    for i in range(len(inputs_test)):

        min_labels_test_217 = np.min(labels_test[i,:,:,0])
        max_labels_test_217 = np.max(labels_test[i,:,:,0])
    
        min_inputs_test_143 = np.min(inputs_test[i,:,:,0])
        max_inputs_test_143 = np.max(inputs_test[i,:,:,0])
        min_inputs_test_217 = np.min(inputs_test[i,:,:,1])
        max_inputs_test_217 = np.max(inputs_test[i,:,:,1])
        min_inputs_test_353 = np.min(inputs_test[i,:,:,2])
        max_inputs_test_353 = np.max(inputs_test[i,:,:,2])
        
        normalize_inputs_test[i,:,:,0] = (inputs_test[i,:,:,0] - min_inputs_test_143)/(max_inputs_test_143 - min_inputs_test_143)
        normalize_inputs_test[i,:,:,1] = (inputs_test[i,:,:,1] - min_inputs_test_217)/(max_inputs_test_217 - min_inputs_test_217)
        normalize_inputs_test[i,:,:,2] = (inputs_test[i,:,:,2] - min_inputs_test_353)/(max_inputs_test_353 - min_inputs_test_353)

        normalize_labels_test[i,:,:,0] = (labels_test[i,:,:,0] - min_inputs_test_217)/(max_inputs_test_217 - min_inputs_test_217)
        

        
    inputs_test = normalize_inputs_test
    labels_test = normalize_labels_test
    
    return inputs_test, labels_test, inputs_train, labels_train
    # return inputs_train, labels_train

def reading_validation_set(validation_file_path, Patch_Size):
    
    validation_file = h5py.File(validation_file_path, 'r')

    inputs_validation = validation_file["M"][:,:,:].astype(np.float32)
    labels_validation = validation_file["M0"][:,:,:].astype(np.float32)
    
    
    
    # Scaling validation data between 0 and 1
    
    normalize_inputs_validation = np.zeros(
            [len(inputs_validation), Patch_Size, Patch_Size, 3])
       
    normalize_labels_validation = np.zeros(
        [len(labels_validation), Patch_Size, Patch_Size, 1])

    maximo = np.zeros([len(inputs_validation), Patch_Size, Patch_Size])
    
    minimo = np.zeros([len(inputs_validation), Patch_Size, Patch_Size])
    
    for i in range(len(inputs_validation)):

        min_labels_validation_217 = np.min(labels_validation[i,:,:,0])
        max_labels_validation_217 = np.max(labels_validation[i,:,:,0])
    
        min_inputs_validation_143 = np.min(inputs_validation[i,:,:,0])
        max_inputs_validation_143 = np.max(inputs_validation[i,:,:,0])
        min_inputs_validation_217 = np.min(inputs_validation[i,:,:,1])
        max_inputs_validation_217 = np.max(inputs_validation[i,:,:,1])
        min_inputs_validation_353 = np.min(inputs_validation[i,:,:,2])
        max_inputs_validation_353 = np.max(inputs_validation[i,:,:,2])
        
        normalize_inputs_validation[i,:,:,0] = (inputs_validation[i,:,:,0] - min_inputs_validation_143)/(max_inputs_validation_143 - min_inputs_validation_143)
        normalize_inputs_validation[i,:,:,1] = (inputs_validation[i,:,:,1] - min_inputs_validation_217)/(max_inputs_validation_217 - min_inputs_validation_217)
        normalize_inputs_validation[i,:,:,2] = (inputs_validation[i,:,:,2] - min_inputs_validation_353)/(max_inputs_validation_353 - min_inputs_validation_353)

        normalize_labels_validation[i,:,:,0] = (labels_validation[i,:,:,0] - min_inputs_validation_217)/(max_inputs_validation_217 - min_inputs_validation_217)

        maximo[i,:,:] = max_inputs_validation_217
        minimo[i,:,:] = min_inputs_validation_217
        

    
    inputs_validation = normalize_inputs_validation
    labels_validation = normalize_labels_validation

    return inputs_validation, labels_validation, maximo, minimo


def reading_validation_set_no_normalized(validation_file_path):

    # For saving true CMB maps before normalisation
    
    validation_file = h5py.File(validation_file_path, 'r')

    inputs_validation = validation_file["M"][:,:,:].astype(np.float32)
    labels_validation = validation_file["M0"][:,:,:].astype(np.float32)

    return labels_validation

def reading_min_max_values():
    
    Min_Max_file_path = './CENN_Max_Min_U.h5' 
    
    Min_Max_file = h5py.File(Min_Max_file_path, 'r')

    maximo = Min_Max_file['max'][()].astype(float)
    minimo = Min_Max_file['min'][()].astype(float)

    return maximo, minimo

def reading_model(Model_file_path):
    
    Model_file = h5py.File(Model_file_path, 'r')

    Model = Model_file
    
    return Model