#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:54:15 2023

@author: josemanuel
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py


def plots(Filtro, Num_imagenes, validation_file_path):

    data = h5py.File('./Outputs_CENN_'+Filtro+'.h5', 'r')
    validation = h5py.File(validation_file_path, 'r')

    inputs = data["sim"][:,:,:].astype(np.float32)
    input_total = validation["M"][:,:,:].astype(np.float32)
    outputs=data["net"][:,:,:].astype(np.float32)


    if Filtro == 'E':   
        for i in range(Num_imagenes):
            
            fig, ([ax0, ax1, ax2, ax3]) = plt.subplots(1, 4, figsize=(16,6))
            fig.tight_layout(pad=4.0)
            
            ax0.title.set_text('Input Total (E)')
            ax1.title.set_text('True CMB (E)')
            ax2.title.set_text('Output CMB (E)')
            ax3.title.set_text('Residual CMB (E)')

            fig0=ax0.imshow(input_total[i][:,:,1]*1e6)
            fig.colorbar(fig0, ax=ax0, fraction=0.046, pad=0.04)
            
            fig1=ax1.imshow(inputs[i][:,:,0]*1e6)
            fig.colorbar(fig1, ax=ax1, fraction=0.046, pad=0.04)
            
            fig2=ax2.imshow(outputs[i][7:249,7:249,0]*1e6)
            fig.colorbar(fig2, ax=ax2, fraction=0.046, pad=0.04)

            fig3=ax3.imshow(inputs[i][7:249,7:249,0]*1e6 - outputs[i][7:249,7:249,0]*1e6)
            fig.colorbar(fig3, ax=ax3, fraction=0.046, pad=0.04)
            
            plt.savefig('Validation_E_'+str(i)+'.pdf')
    

    else:
        for i in range(Num_imagenes):
            
            fig, ([ax0, ax4, ax5, ax6]) = plt.subplots(1, 4, figsize=(16,6))
            fig.tight_layout(pad=4.0)

            ax0.title.set_text('Input Total (B)')    
            ax4.title.set_text('True CMB (B)')
            ax5.title.set_text('Output CMB (B)')
            ax6.title.set_text('Residual CMB (B)')
            
            fig0=ax0.imshow(input_total[i][:,:,1]*1e6)
            fig.colorbar(fig0, ax=ax0, fraction=0.046, pad=0.04)
            
            fig4=ax4.imshow(inputs[i][:,:,0]*1e6)
            fig.colorbar(fig4, ax=ax4, fraction=0.046, pad=0.04)
            
            fig5=ax5.imshow(outputs[i][7:249,7:249,0]*1e6)
            fig.colorbar(fig5, ax=ax5, fraction=0.046, pad=0.04)

            fig6=ax6.imshow(inputs[i][7:249,7:249,0]*1e6 - outputs[i][7:249,7:249,0]*1e6)
            fig.colorbar(fig6, ax=ax6, fraction=0.046, pad=0.04)
            
            plt.savefig('Validation_B_'+str(i)+'.pdf')

