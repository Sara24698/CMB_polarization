#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# from InputSimulation import reading_the_data
# from Simulation import Simulation

import h5py

# Import the NaMaster python wrapper
import pymaster as nmt

# Simulation_Parameters = reading_the_data()

Images_Directory = './figures/draft/'


data_Q = ['./Outputs_0_2_B.h5', './Outputs_0_1_B.h5', './Outputs_0_05_B.h5', './Outputs_0_01_B.h5', './Outputs_0_004_B.h5', './Outputs_0_001_B.h5', './Outputs_Planck_B.h5']
data_U = ['./Outputs_0_2_B.h5', './Outputs_0_1_B.h5', './Outputs_0_05_B.h5', './Outputs_0_01_B.h5', './Outputs_0_004_B.h5', './Outputs_0_001_B.h5', './Outputs_Planck_B.h5']

color_Input = ['tab:red', 'tab:orange', 'tab:brown', 'tab:green', 'tab:blue', 'tab:purple', 'tab:gray']
color_Output = ['tab:red']
color_Error = ['black']
color_Residuals = ['black']

label_Input = ['r = 0.2', 'r = 0.1', 'r = 0.05', 'r = 0.01', 'r = 0.004', 'r = 0.001', 'Planck patches']
label_Output = ['Output']
label_Error = ['r = 0.1']
label_Residuals = ['R = 0.2']

lmax = [600]
aposcale = [0.8]

    
class Estimate_E_B_Power_Spectrum():
    
    def estimating_theoretical_power_spectrum_all_sky(Archivo_fits):
        
        CMB = hp.read_map(Archivo_fits, (0,1,2))
        
        # fwhm 0.5 deg in rad
      
        Cls = hp.sphtfunc.anafast(CMB, lmax = 2500)
        
        l = np.arange(0, 2501)
        
        # maps in muK^2
        
        EE = Cls[1]*1e12
        BB = Cls[2]*1e12
        
        # change to Cl l(l+1)/2pi
        
        EE = EE*(l*(l+1))/(2*np.pi)
        BB = BB*(l*(l+1))/(2*np.pi)
        
        return l, EE, BB

    def power_spectrum_from_E(mapE, mapB, lmax=2500):
        mapE = hp.read_map(mapE)
        mapB = hp.read_map(mapB)
       
        ClEE = hp.sphtfunc.anafast(mapE, lmax = 2500, pol=False)
        ClBB = hp.sphtfunc.anafast(mapB, lmax = 2500, pol=False)

        l = np.arange(0, 2501)  # l va de 0 a lmax
    
        EE = ClEE * 1e12 
        EE = EE * (l * (l + 1)) / (2 * np.pi)

        BB = ClBB * 1e12  
        BB = BB * (l * (l + 1)) / (2 * np.pi)

        return l, EE, BB

    def estimating_E_B_from_CENN(data_Q, data_U, flag):
        
        Number_Of_Initial_Pixels = 512
        Number_Of_Cutted_Pixels = 16
        Number_Of_Pixels = 480
        
        data_Q = h5py.File(data_Q, 'r')
        data_U = h5py.File(data_U, 'r')
        
        # Q
        
        inputs_Q = data_Q["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        
        # U
            
        inputs_U = data_U["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        
        # Residuals
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(inputs_Q)):
            
            # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
        
            mask = np.ones_like(inputs_Q[0,:,:]).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            

            apo_fac=16    
            mask[np.where(xarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(xarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask = mask.reshape([Nx, Ny])

            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")

            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [inputs_Q[i][:,:], inputs_U[i][:,:]], purify_b=True)            
            
            bins = np.arange(0,2525,35)

            b = nmt.NmtBinFlat(bins[:-1], bins[1:])
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q

            w22 = nmt.NmtWorkspaceFlat()
            w22.compute_coupling_matrix(f2_inputs, f2_inputs, b)

            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)

            Input_CMB = w22.decouple_cell(Input_CMB)

    
            Input_CMB_lista_ee.append(Input_CMB[0]*1e12)
            Input_CMB_lista_bb.append(Input_CMB[3]*1e12)
                        
        # estimating average power spectrum from all the patches    
            
        # Cls Input
        
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Input_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Input_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Input_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Input_CMB_lista_bb)
       

        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Input_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Input_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Input_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Input_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
                
        return mean_ee, mean_bb, std_ee, std_bb, ells_uncoupled
        

class Plot_E_B_Power_Spectrum():
        
    def reading_dust_spectra(resolution):
        
        # resolution 0 is 30 arcmin, 1 is 25 and 2 is 20
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(2):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_dust_power_spectra(resolution=resolution, channel=i)
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l
    
    def reading_the_spectra():
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(len(data_Q)):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_E_B_from_CENN(data_Q[i], data_U[i], flag=i)
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l    


    def plotting_E_B_Power_Spectrum_from_CENN():
        
        EE, BB, EE_Uncertainty, BB_Uncertainty, l = Plot_E_B_Power_Spectrum.reading_the_spectra()              
        l_th_all_sky_Planck, EE_th_all_sky_Planck, BB_th_all_sky_Planck = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./CMB_143GHZ_SimPlanck.fits')
             
                  
        # EE
        
                
                
        for i in range(len(data_Q)):
        

            if i ==0:
                plt.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Input[i], label=label_Input[i])
                plt.fill_between(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color=color_Input[i], alpha=0.2)
                
                                
            else:
                 plt.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color = color_Input[i], label=label_Input[i])
                 plt.fill_between(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color = color_Input[i], alpha=0.1)


        plt.plot(l_th_all_sky_Planck, EE_th_all_sky_Planck, color = 'black', label='Planck')
          
        plt.ylabel(r'$\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
        plt.tight_layout()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$\mathcal{l}$')
        plt.legend(loc='lower right', fontsize=10)
        plt.xlim(10, 2501)
        plt.ylim([0.00001,1201])        
        
        plt.savefig('EE_distinto_r_256_parches_E.png')
        
     
                       
# BB
                 
        for i in range(len(data_U)):
        
            if i ==0:
                plt.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Input[i], label=label_Input[i])
                plt.fill_between(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color=color_Input[i], alpha=0.2)
                
                                
            else:
                 plt.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color = color_Input[i], label=label_Input[i])
                 plt.fill_between(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color = color_Input[i], alpha=0.1)


        plt.plot(l_th_all_sky_Planck, BB_th_all_sky_Planck, color = 'black', label='Planck')

                        
        plt.ylabel(r'$\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
        plt.tight_layout()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$\mathcal{l}$')
        plt.xlim(10, 2001)
        plt.ylim(0.000009, 10)
        plt.legend(loc='lower right', fontsize=10)        
        plt.savefig('BB_distinto_r_512_parches_B.png')
        
               
Plot_E_B_Power_Spectrum.plotting_E_B_Power_Spectrum_from_CENN()

