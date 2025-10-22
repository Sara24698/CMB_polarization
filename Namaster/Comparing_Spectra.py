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


data_Q = ['./Outputs_CENN_B.h5']
data_U = ['./Outputs_CENN_B.h5']

color_Input = ['tab:green']
color_Output = ['tab:red']
color_Error = ['black']
color_Residuals = ['black']

label_Input = ['Input']
label_Output = ['Output']
label_Error = ['r = 0.1']
label_Residuals = ['R = 0.2']

lmax = [600]
aposcale = [0.8]


class Read():
    
    def reading_noise_patches():
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        
        data_Q = ['./Parches_ruido_Q.h5']
        data_U = ['./Parches_ruido_U.h5']
        
        data_Q = h5py.File(data_Q[0], 'r')
        data_U = h5py.File(data_U[0], 'r')
                
        noise_Q= data_Q['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U = data_U['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_Q = noise_Q[:,:,:, 1]
        noise_U = noise_U[:,:,:, 1]

        print(np.shape(noise_Q))

        return noise_Q, noise_U

    def reading_dust_patches():
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        
        data_Q = ['./Validation_Patches_D_Q_30arcmin.h5', './Validation_Patches_D_Q_25arcmin.h5', './Validation_Patches_D_Q_20arcmin.h5', './Validation_Patches_D_Q_5arcmin_d6s1.h5', './Validation_Patches_D_Q_5arcmin_d4s2.h5']
        data_U = ['./Validation_Patches_D_U_30arcmin.h5', './Validation_Patches_D_Q_25arcmin.h5', './Validation_Patches_D_U_20arcmin.h5', './Validation_Patches_D_U_5arcmin_d6s1.h5', './Validation_Patches_D_U_5arcmin_d4s2.h5']
        
        data_Q_30arcmin = h5py.File(data_Q[0], 'r')
        data_U_30arcmin = h5py.File(data_U[0], 'r')
        
        data_Q_25arcmin = h5py.File(data_Q[1], 'r')
        data_U_25arcmin = h5py.File(data_U[1], 'r')
        
        data_Q_20arcmin = h5py.File(data_Q[2], 'r')
        data_U_20arcmin = h5py.File(data_U[2], 'r')
        
        data_Q_5arcmin_d6s1 = h5py.File(data_Q[3], 'r')
        data_U_5arcmin_d6s1 = h5py.File(data_U[3], 'r')
        
        data_Q_5arcmin_d4s2 = h5py.File(data_Q[4], 'r')
        data_U_5arcmin_d4s2 = h5py.File(data_U[4], 'r')
        
        noise_Q_30arcmin = data_Q_30arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_30arcmin = data_U_30arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        noise_Q_25arcmin = data_Q_25arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_25arcmin = data_U_25arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        noise_Q_20arcmin = data_Q_20arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_20arcmin = data_U_20arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        dust_Q_5arcmin_d6s1 = data_Q_5arcmin_d6s1['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        dust_U_5arcmin_d6s1 = data_U_5arcmin_d6s1['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        dust_Q_5arcmin_d4s2 = data_Q_5arcmin_d4s2['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        dust_U_5arcmin_d4s2 = data_U_5arcmin_d4s2['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)


        return noise_Q_30arcmin, noise_U_30arcmin, noise_Q_25arcmin, noise_U_25arcmin, noise_Q_20arcmin, noise_U_20arcmin, dust_Q_5arcmin_d6s1, dust_U_5arcmin_d6s1, dust_Q_5arcmin_d4s2, dust_U_5arcmin_d4s2
    
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
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        
        data_Q = h5py.File(data_Q, 'r')
        data_U = h5py.File(data_U, 'r')
        
        # Q
        
        outputs_Q = data_Q["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        inputs_Q = data_Q["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        
        # U
            
        outputs_U = data_U["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        inputs_U = data_U["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        
        # Residuals
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        Output_CMB_lista_ee = []
        Output_CMB_lista_bb = []
        Residuals_CMB_lista_ee = []
        Residuals_CMB_lista_bb = []
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
        
            mask = np.ones_like(outputs_Q[0,:,:]).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            

            apo_fac=16    
            mask[np.where(xarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(xarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask = mask.reshape([Nx, Ny])

            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")

            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [inputs_Q[i][:,:,0], inputs_U[i][:,:,0]], purify_b=True)
            f2_outputs = nmt.NmtFieldFlat(Lx, Ly, mask, [outputs_Q[i][:,:,0], outputs_U[i][:,:,0]], purify_b=True)          
            f2_residuals = nmt.NmtFieldFlat(Lx, Ly, mask, [(inputs_Q[i][:,:,0]-outputs_Q[i][:,:,0]), (
                inputs_U[i][:,:,0]-outputs_U[i][:,:,0])], purify_b=True)
            
            
            bins = np.arange(0,2525,35)

            b = nmt.NmtBinFlat(bins[:-1], bins[1:])
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q

            w22 = nmt.NmtWorkspaceFlat()
            w22.compute_coupling_matrix(f2_inputs, f2_inputs, b)

            w33 = nmt.NmtWorkspaceFlat()
            w33.compute_coupling_matrix(f2_outputs, f2_outputs, b)


            w44 = nmt.NmtWorkspaceFlat()
            w44.compute_coupling_matrix(f2_residuals, f2_residuals, b)

            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)
            Output_CMB = nmt.compute_coupled_cell_flat(f2_outputs, f2_outputs, b)
            Residuals_CMB = nmt.compute_coupled_cell_flat(f2_residuals, f2_residuals, b)

            Input_CMB = w22.decouple_cell(Input_CMB)
            Output_CMB = w33.decouple_cell(Output_CMB)
            Residuals_CMB = w44.decouple_cell(Residuals_CMB)

    
            Input_CMB_lista_ee.append(Input_CMB[0]*1e12)
            Input_CMB_lista_bb.append(Input_CMB[3]*1e12)
            
            Output_CMB_lista_ee.append(Output_CMB[0]*1e12)
            Output_CMB_lista_bb.append(Output_CMB[3]*1e12)
            
            Residuals_CMB_lista_ee.append(Residuals_CMB[0]*1e12)
            Residuals_CMB_lista_bb.append(Residuals_CMB[3]*1e12)
            
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
        
        # Cls Output
             
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
        
           
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Output_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Output_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Output_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Output_CMB_lista_bb)

        for i in range(len(Output_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Output_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Output_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Output_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Output_CMB_lista_bb))


        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        # Cls Residuals
        
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Residuals_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Residuals_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Residuals_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Residuals_CMB_lista_bb)
        
        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Residuals_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Residuals_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Residuals_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Residuals_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        return mean_ee, mean_bb, std_ee, std_bb, ells_uncoupled
    
    def estimating_noise_power_spectra():
        
        # flag 0 = 30arcmin, flag 1 = 25 arcmin, flag 2 = 20 arcmin
        
        aposcale = [0.8]
        lmax = [800]
    
        Number_Of_Pixels = 240
            
        noise_Q, noise_U = Read.reading_noise_patches()
             
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(noise_Q)):
            
             # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
    
            mask = np.ones_like(noise_Q[0,:]).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            
            apo_fac=16    
            mask[np.where(xarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(xarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask = mask.reshape([Nx, Ny])

            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")

            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [noise_Q[i][:,:], noise_U[i][:,:]], purify_b=True)      
            
            bins = np.arange(0,2025,35)
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

    def estimating_dust_power_spectra(resolution, channel):
        
        # flag 0 = 30arcmin, flag 1 = 25 arcmin, flag 2 = 20 arcmin
        
        aposcale = [0.8, 1., 1.2, 1.5, 1.5]
        lmax = [600, 700, 800, 1500, 1500]
    
        Number_Of_Pixels = 240
            
        noise_Q_30arcmin, noise_U_30arcmin, noise_Q_25arcmin, noise_U_25arcmin, noise_Q_20arcmin, noise_U_20arcmin, dust_Q_5arcmin_d6s1, dust_U_5arcmin_d6s1, dust_Q_5arcmin_d4s2, dust_U_5arcmin_d4s2 = Read.reading_dust_patches()
    
        noise_Q = [noise_Q_30arcmin, noise_Q_25arcmin, noise_Q_20arcmin, dust_Q_5arcmin_d6s1, dust_Q_5arcmin_d4s2]
        noise_U = [noise_U_30arcmin, noise_U_25arcmin, noise_U_20arcmin, dust_U_5arcmin_d6s1, dust_U_5arcmin_d4s2]
        
        noise_Q = noise_Q[resolution]
        noise_U = noise_U[resolution]  
        
        noise_Q = noise_Q[:,:,:,channel]
        noise_U = noise_U[:,:,:,channel]  
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(noise_Q_30arcmin)):
            
            # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
    
            l, cl_tt, cl_ee_th, cl_bb_th, cl_te = np.loadtxt('cls.txt', unpack=True)
            beam = np.exp(-((0.5)* np.pi/180 * l)**2)
            cl_tt *= beam
            cl_ee_th *= beam
            cl_bb_th *= beam
            cl_te *= beam
            mpt, mpq, mpu = nmt.synfast_flat(Nx, Ny, Lx, Ly,
                                             np.array([cl_tt, cl_te, 0 * cl_tt,
                                                       cl_ee_th, 0 * cl_ee_th, cl_bb_th]),
                                             [0, 2])
    
            mask = np.ones_like(mpt).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            
            
            # First we dig a couple of holes
            def dig_hole(x, y, r):
                rad = (np.sqrt((xarr - x)**2 + (yarr - y)**2)).flatten()
                return np.where(rad < r)[0]
            
            
            mask[dig_hole(0.3 * Lx, 0.6 * Ly, 0.05 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.12 * Ly, 0.07 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.8 * Ly, 0.03 * np.sqrt(Lx * Ly))] = 0.
    
            mask[np.where(xarr.flatten() < Lx / 16.)] = 0
            mask[np.where(xarr.flatten() > 15 * Lx / 16.)] = 0
            mask[np.where(yarr.flatten() < Ly / 16.)] = 0
            mask[np.where(yarr.flatten() > 15 * Ly / 16.)] = 0
            mask = mask.reshape([Ny, Nx])
    
            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposcale[resolution], apotype="C1")
            # mask2 = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")
            # x 1e6 to muK
    
            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [noise_Q[i][:,:]*1e6, noise_U[i][:,:]*1e6], purify_b=True)
            
            # l0_bins = 10**np.arange(1.5, 3.5, 0.1)
            # lf_bins = 10**np.arange(1.5, 3.5, 0.1)
            
            l0_bins = np.arange(50, lmax[resolution], 25)
            lf_bins = np.arange(60, lmax[resolution], 25)
            
            
            b = nmt.NmtBinFlat(l0_bins, lf_bins)
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q
            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)
    
            Input_CMB_lista_ee.append(Input_CMB[0])
            Input_CMB_lista_bb.append(Input_CMB[3])
            
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
    
    def reading_noise_spectra():
        
        # resolution 0 is 30 arcmin, 1 is 25 and 2 is 20
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(3):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_noise_power_spectra()
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l
    
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
        

        # EE
        
             
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})        
        for i in range(len(data_Q)):
        
            

            if i ==0:
                #a0.plot(l_th_all_sky_Planck_ruido, EE_th_all_sky_Planck_ruido, linestyle='dashed', color='grey', label='Planck noise')
                a0.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='tab:blue', label=label_Input[i])
                #a0.plot(l_noise[i], ((EE_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), linestyle='dashed', color='grey', label='Patch noise')
                a0.fill_between(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color='tab:blue', alpha=0.2)
                
                a0.plot(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color = 'tab:red', label=label_Output[i])
                a0.fill_between(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][1]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color = 'tab:red', alpha=0.2)

                a0.plot(l[i], ((EE[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), color='tab_grey', label='Residuals')
                a0.fill_between(l[i], ((EE[i][2]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (EE_Uncertainty[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][2]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][2]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color='tab:grey', alpha=0.2)


                
                a1.plot(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])
                print(np.mean((((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi)))))
                print(np.mean(((EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi)))))
                a1.fill_between(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))) - ((
                    EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), (((EE[i][0]-EE[i][1])*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color=color_Error[i], alpha=0.2)
                
            else:
                 #err=(((EE[i][0][56]-EE[i][1][56])*(l[i][56]*(l[i][56]+1))/(2*np.pi)))
                 #a0.errorbar(2000, (EE[0][1][56]*(l[0][56]*(l[0][56]+1))/(2*np.pi)), yerr=err,  fmt='*',  color='red',  label='l = 2000', capsize=5, capthick=2)
                 a0.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='blue', label=label_Input[i])
                 a0.plot(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), label=label_Output[i])
                 a1.plot(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])

            a1.axhline(y = 0, color = 'tab:red', linestyle = 'dashed',linewidth=1)
        
            a0.set_ylabel(r'$\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
            a1.set_ylabel(r'$\Delta\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
            f.tight_layout()
            a0.set_yscale('log')
            a0.set_xscale('log')
            a1.set_xscale('log')
            a0.set_xlabel('$\mathcal{l}$')
            a0.legend(loc='lower right', fontsize=10)
            #a1.legend(loc='upper right', fontsize=6)
            a0.set_xlim(10, 2501)
            a1.set_xlim(10, 2501)
            a0.set_ylim([0.00001,1201])
            a1.set_ylim([-10,10])
        
        
        plt.savefig('EE_validation_Planck.png')
        
     
                       
# BB
        
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

        #a0.plot(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), color=color_Input[0], label='Input')
        #a0.fill_between(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))) - (
         #   (BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), ((BB[0][0]*(l[0]*(
          #      l[0]+1))/(2*np.pi))) + ((BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(
           #         2*np.pi))), color=color_Input[0], alpha=0.2)


        #plt.plot(l_th_all_sky_Planck, BB_th_all_sky_Planck, color='black', linestyle='dashed', label='BB Planck')
        #a0.plot(l_th_all_sky_Planck_ruido, BB_th_all_sky_Planck_ruido, linestyle='dashed', color='purple', label='Planck noise')
        #plt.plot(l_th_all_sky_0_2, BB_th_all_sky_0_2, label='r = 0.2')
        #plt.plot(l_th_all_sky_0_1, BB_th_all_sky_0_1, label='r = 0.1')
        #plt.plot(l_th_all_sky_0_05, BB_th_all_sky_0_05, label='r = 0.05')
        #plt.plot(l_th_all_sky_0_01, BB_th_all_sky_0_01,  label='r = 0.01')
        #plt.plot(l_th_all_sky_0_004, BB_th_all_sky_0_004, label='r = 0.004')
        #a0.plot(l_th_all_sky_143, BB_th_all_sky_143, color='orange', linestyle=':', label='Dust')


          
        for i in range(len(data_U)):
            if i ==0:
                a0.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='tab:blue', label=label_Input[i])
                #plt.plot(l[i], ((BB_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), color='grey', linestyle='dashed', label='Patch noise')
                a0.fill_between(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), color='tab:blue', alpha=0.2)
                  
                a0.plot(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color='tab:red', label=label_Output[i])   
                a0.fill_between(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][1]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color='tab:red', alpha=0.2)

                a0.plot(l[i], ((BB[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), color='black', label='Residuals')
                a0.fill_between(l[i], ((BB[i][2]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (BB_Uncertainty[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][2]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][2]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color='black', alpha=0.2)

                
                a1.plot(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])
                print(np.mean((((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi)))))
                print(np.mean(((BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi)))))
                a1.fill_between(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))) - ((
                    BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), (((BB[i][0]-BB[i][1])*(l[i]*(
                       l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                           2*np.pi))), color=color_Error[i], alpha=0.2)
               
            #else: 
                #a0.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='tab:blue', label=label_Input[i])
                #plt.plot(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), label=label_Output[i])
                #a1.plot(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])

           


                  
        a1.axhline(y = 0, color = 'tab:red', linestyle = 'dashed',linewidth=1)
                        
        a0.set_ylabel(r'$\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
        a1.set_ylabel(r'$\Delta\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
        f.tight_layout()
        a0.set_yscale('log')
        a0.set_xscale('log')
        a1.set_xscale('log')
        a0.set_xlabel('$\mathcal{l}$')
        a0.set_xlim(10, 2001)
        a1.set_xlim(10, 2001)
        a0.set_ylim(0.000009, 10)
        a0.legend(loc='lower right', fontsize=10)
        #a1.legend(loc='upper right', fontsize=6)
        a1.set_ylim([-1, 1])
        
        plt.savefig('BB_validation_Planck')
        
               
Plot_E_B_Power_Spectrum.plotting_E_B_Power_Spectrum_from_CENN()

