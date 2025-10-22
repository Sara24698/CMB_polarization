from CENN_Train import train
from CENN_Execute import execute
from Plot_CENN import plots

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf



learning_rate = 0.005
momentum = 0.9
batch_size = 32
num_epochs = 500
test_frequency = 1
Patch_Size=256
Filtro='B'
Num_imagenes=2

train_file_path = './Train_'+Filtro+'.h5'
test_file_path = './Test_'+Filtro+'.h5'
validation_file_path = './Validation_'+Filtro+'.h5'


def main():
    train(learning_rate, batch_size, num_epochs, test_frequency, Patch_Size, Filtro)
    execute(Patch_Size, Filtro)
    plots(Filtro, Num_imagenes, validation_file_path)



if __name__ == "__main__":
    main()
