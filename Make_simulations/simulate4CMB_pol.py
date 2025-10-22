#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# simulate radio backgrouns PS
# add CMB
# add Syncrotron
# add Dust
# add noise

def make_parallel_simu4CMB(in_parfile, out_file, noise=False, QU=False):
    from multiprocessing import Pool
    from datetime import datetime
    import json
    from functools import partial
    from numpy import zeros, vstack, shape, max
    import matplotlib.pyplot as plt
    import numpy as np

    print(str(datetime.now()))

    with open(in_parfile) as par_file: 
        Simulation_Parameters = json.loads(par_file.read())

    Lon, Lat = CreateRandomCatalogue(nmcat = None,
                                     Ns = Simulation_Parameters['n_sims'], ex_rad = 6.,
                                     cut = Simulation_Parameters["gal_cut"])

    pos = [[x,y] for x, y in zip(Lon,Lat)]
    fun = partial(MakeSimu, nm_parfile = in_parfile)

    p = Pool()
    sims = p.map(fun, pos)

    # sims is a list of MakeSimu outputs.
    # sims = [total_map_Q_low, total_map_U_low, total_map_Q, total_map_U,  total_map_Q_high, total_map_U_high, label_Q, label_U]_1, ...

    total_map_list_Q = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 3))
    total_map_list_U = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 3))
    label_list_Q = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    label_list_U = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    label_list_Q_low = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    label_list_U_low = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    label_list_Q_high = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    label_list_U_high = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    
    for i in range(len(sims)):
        # lower freq
        total_map_list_Q[i,:,:,0] = sims[i][0]       
        total_map_list_U[i,:,:,0] = sims[i][1]
        # central freq
        total_map_list_Q[i,:,:,1] = sims[i][2]       
        total_map_list_U[i,:,:,1] = sims[i][3]

        label_list_Q[i,:,:,0] = sims[i][6]
        label_list_U[i,:,:,0] = sims[i][7]

        label_list_Q_low[i,:,:,0] = sims[i][8]
        label_list_U_low[i,:,:,0] = sims[i][9]

        label_list_Q_high[i,:,:,0] = sims[i][10]
        label_list_U_high[i,:,:,0] = sims[i][11]
        # higher freq
        total_map_list_Q[i,:,:,2] = sims[i][4]       
        total_map_list_U[i,:,:,2] = sims[i][5]

    if QU==True:       
        write2h5(total_map_list_Q, label_list_Q, label_list_Q_low, label_list_Q_high, out_file + '_Q.h5')
        write2h5(total_map_list_U, label_list_U, label_list_U_low, label_list_U_high, out_file + '_U.h5')

    else:
        write2h5(total_map_list_Q, label_list_Q, label_list_Q_low, label_list_Q_high, out_file + '_E.h5')
        write2h5(total_map_list_U, label_list_U, label_list_U_low, label_list_U_high, out_file + '_B.h5')
        
     
    p.terminate()
    print(str(datetime.now()))

    pass


def MakeSimu(pos, nm_parfile, noise = False):
    
    from corrsky_v0124 import corrsky_TQU_hp, fluxassoc_TQU_hp
    import astropy
    from astropy import units as u
    import json
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from numpy import sqrt, log, zeros, random, std, zeros_like, mean, std, pi
    import healpy as hp
    
    with open(nm_parfile) as par_file: 
        Simulation_Parameters = json.loads(par_file.read())

    # make patch with random radio background sources
    if 'PS' in Simulation_Parameters["contaminants"]:
        Rbk_map, Rbk_mapp, R_bkcat, Rbk_catp = corrsky_TQU_hp(nu = Simulation_Parameters['freq'],
                                                        npix = Simulation_Parameters['npix'],
                                                        pixsize = Simulation_Parameters['pixsize'],
                                                        sncmodel = 'tucci', pkmodel = 'lapi11',
                                                        Sn = Simulation_Parameters['Sn'],
                                                        Sx = Simulation_Parameters['Sx'],
                                                        Slim = None, zbin = None,
                                                        pol = [Simulation_Parameters['Pol_mu'],
                                                                Simulation_Parameters['Pol_sigma']],
                                                        powlaw = None, wpol = None)
        Rbk_mapp_low = zeros_like(Rbk_mapp)
        Rbk_mapp_high = zeros_like(Rbk_mapp)

        alpha = 0.
        alpha = random.normal(Simulation_Parameters['mean_alpha_radio_low'], 0.05, 1)
        for ii in range(2):
            Rbk_mapp_low[ii+2] =  Rbk_mapp[ii+2] * (Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

        alpha = random.normal(Simulation_Parameters['mean_alpha_radio_high'], 0.05, 1)
        for ii in range(2):
            Rbk_mapp_high[ii+2] =  Rbk_mapp[ii+2] * (Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
                

        # plt.figure(1)
        # plt.imshow(Rbk_map[2]), plt.colorbar()
        # plt.savefig('Rbk_map_143_Q.pdf')
        # plt.close()
        # make patch with IR background sources 
        
        IRbk_map, IRbk_mapp,IR_bkcat, IRbk_catp = corrsky_TQU_hp(nu = Simulation_Parameters['freq'],
                                                            npix = Simulation_Parameters['npix'],
                                                            pixsize = Simulation_Parameters['pixsize'],
                                                            sncmodel = 'lapi11', pkmodel = 'lapi11',
                                                            Sn = -2, Sx = min(Simulation_Parameters['Sx'],0.0),
                                                            Slim = None, zbin = None,
                                                            pol = [Simulation_Parameters['Pol_mu_IR'],
                                                                    Simulation_Parameters['Pol_sigma_IR']],
                                                            powlaw = None, wpol = None)
        
        IRbk_map_low = zeros_like(IRbk_map)
        IRbk_map_high = zeros_like(IRbk_map)
        
        alpha = 0.

        alpha = random.normal(Simulation_Parameters['mean_alpha_IR_low'], 0.05, 1)
        for ii in range(2):
            IRbk_map_low[ii+2] =  IRbk_map[ii+2] * (Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

        alpha = random.normal(Simulation_Parameters['mean_alpha_IR_high'], 0.05, 1)
        for ii in range(2):
            IRbk_map_high[ii+2] = IRbk_map[ii+2] * (Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
                

        # plt.figure(1)
        # plt.imshow(IRbk_map[2]), plt.colorbar()
        # plt.savefig('IRbk_map_143_Q.pdf')
        # plt.close()
        # make patch with IRLT sources
        
        IRLTbk_map, IRLTbk_mapp, IRLT_bkcat, IRLTbk_catp = corrsky_TQU_hp(nu = Simulation_Parameters['freq'],
                                                                    npix = Simulation_Parameters['npix'],
                                                                    pixsize = Simulation_Parameters['pixsize'],
                                                                    sncmodel = 'IRLT', pkmodel = 'lapi11',
                                                                    Sn = -2, Sx = Simulation_Parameters['Sx'],
                                                                    Slim = None, zbin = None,
                                                                    pol = [Simulation_Parameters['Pol_mu_IR'],
                                                                            Simulation_Parameters['Pol_sigma_IR']],
                                                                    powlaw = None, wpol = None)
        IRLTbk_map_low = zeros_like(IRLTbk_map)
        IRLTbk_map_high = zeros_like(IRLTbk_map)
        
        alpha = 0.

        alpha = random.normal(Simulation_Parameters['mean_alpha_IR_low'], 0.05, 1)
        for ii in range(2):
            IRLTbk_map_low[ii+2] =  IRLTbk_map[ii+2] * (Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

        alpha = random.normal(Simulation_Parameters['mean_alpha_IR_high'], 0.05, 1)
        for ii in range(2):
            IRLTbk_map_high[ii+2] = IRLTbk_map[ii+2] * (Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
                
        alpha = 0.

        # plt.figure(1)
        # plt.imshow(IRLTbk_map[2]), plt.colorbar()
        # plt.savefig('IRLTbk_map_143_Q.pdf')
        # plt.close()

        
        # Sum and smoothing
        sigma = Simulation_Parameters["fwhm"] / (2. * sqrt(2. * log(2.))) # fwhm in arcsec (4.9 X 60)?
        sigma = sigma / Simulation_Parameters["pixsize"]
        sigma_low = Simulation_Parameters["fwhm_low"] / (2. * sqrt(2. * log(2.))) # fwhm in arcsec (4.9 X 60)?
        sigma_low = sigma / Simulation_Parameters["pixsize"]
        sigma_high = Simulation_Parameters["fwhm_high"] / (2. * sqrt(2. * log(2.))) # fwhm in arcsec (4.9 X 60)?
        sigma_high = sigma / Simulation_Parameters["pixsize"]
        
        fac = conversion_factor_Jy_K(Simulation_Parameters["freq"],
                                                Simulation_Parameters["pixsize"]/60.)
        fac_low = conversion_factor_Jy_K(Simulation_Parameters["freq_low"],
                                    Simulation_Parameters["pixsize"]/60.)
        fac_high = conversion_factor_Jy_K(Simulation_Parameters["freq_high"],
                                    Simulation_Parameters["pixsize"]/60.)

        Qmap = Rbk_mapp[2] + IRbk_map[2] + IRLTbk_map[2]
        Qmap_low = Rbk_mapp_low[2] + IRbk_map_low[2] + IRLTbk_map_low[2]
        Qmap_high = Rbk_mapp_high[2] + IRbk_map_high[2] + IRLTbk_map_high[2]
        
        # Qmap = Rbk_mapp[2] 
        Qmap *= fac
        Qmap = gaussian_filter(Qmap, sigma)

        Qmap_low *= fac_low
        Qmap_low = gaussian_filter(Qmap_low, sigma_low)

        Qmap_high *= fac_high
        Qmap_high = gaussian_filter(Qmap_high, sigma_high)

        # plt.figure(1)
        # plt.imshow(Qmap), plt.colorbar()
        # plt.savefig('Qmap_143_Q.pdf')
        # plt.close()

        
        Umap = Rbk_mapp[3] + IRbk_map[3] + IRLTbk_map[3]
        Umap_low = Rbk_mapp_low[3] + IRbk_map_low[3] + IRLTbk_map_low[3]
        Umap_high = Rbk_mapp_high[3] + IRbk_map_high[3] + IRLTbk_map_high[3]
        # Umap = Rbk_mapp[3]
        Umap *= fac
        Umap = gaussian_filter(Umap, sigma)

        Umap_low *= fac_low
        Umap_low = gaussian_filter(Umap_low, sigma_low)

        Umap_high *= fac_high
        Umap_high = gaussian_filter(Umap_high, sigma_high)
    
    # add contaminant

    if not pos:
        Lon, Lat = CreateRandomCatalogue(nmcat = None, Ns = 1, ex_rad = 6.,
                                     cut = Simulation_Parameters["gal_cut"])
        pos = [Lon, Lat]
    
    patch_Q = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_Q_low = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_Q_high = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))

    label_Q = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    
    for comp in Simulation_Parameters["contaminants"]:

        # central freq       
        nm_map = Simulation_Parameters["channels"][0][comp]     
        # ForSE
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin,
                                                     npix= Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])

        if comp == "cmb":
            label_Q = omap_forse



        
        patch_Q += omap_forse

        # lower freq       
        nm_map = Simulation_Parameters["channels_low"][0][comp]
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin,
                                                     npix= Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])
        if comp == "cmb":
            label_Q_low = omap_forse

        
        patch_Q_low += omap_forse
  


        # higher freq       
        nm_map = Simulation_Parameters["channels_high"][0][comp]
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin, npix= Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])

        if comp == "cmb":
            label_Q_high = omap_forse
            #plt.figure(1)
            #plt.imshow(label_Q_high*1e6), plt.colorbar()
            #plt.savefig('217_E.png')
            #plt.close()

            
        patch_Q_high += omap_forse


    # plt.figure(1)
    # plt.imshow(patch_Q), plt.colorbar()
    # plt.savefig('patchQ_143_Q.pdf')
    # plt.close()

    patch_U = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_U_low = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_U_high = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))

    label_U = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    
    for comp in Simulation_Parameters["contaminants"]:

        # central freq
        nm_map = Simulation_Parameters["channels"][1][comp]
        # ForSE
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)  
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin,
                                                     npix=Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])

        if comp == "cmb":
            label_U = omap_forse




        patch_U += omap_forse

        # lower freq
        nm_map = Simulation_Parameters["channels_low"][1][comp]
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)  
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin,
                                                     npix=Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])
        
        if comp == "cmb":
            label_U_low = omap_forse



        patch_U_low += omap_forse


        # higher freq
        nm_map = Simulation_Parameters["channels_high"][1][comp]
        inputmap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)  
        omap_forse = cutting_patches_with_ForSE_code(imap=inputmap, pixelsize=(Simulation_Parameters['pixsize'] / 60.)*u.arcmin,
                                                     npix=Simulation_Parameters['npix'], overlap=5, longitude=pos[0], latitude=pos[1])
        
        if comp == "cmb":
            label_U_high = omap_forse
            #plt.figure(1)
            #plt.imshow(label_U_high*1e6), plt.colorbar()
            #plt.savefig('217_B.png')
            #plt.close()



        patch_U_high += omap_forse



    # add noise
    
    if noise == True:
        # central freq
        beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm"] / (2 * sqrt(2 * log(2.))))**2)        
        sigma_noise = Simulation_Parameters["spp"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])       
        Noise_Q = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_Q += Noise_Q
        Noise_U = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_U += Noise_U


        # lower freq
        beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm_low"] / (2 * sqrt(2 * log(2.))))**2)     
        sigma_noise = Simulation_Parameters["spp_low"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])  
        Noise_Q = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_Q_low += Noise_Q       
        Noise_U = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_U_low += Noise_U


        # higher freq
        beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm_high"] / (2 * sqrt(2 * log(2.))))**2) 
        sigma_noise = Simulation_Parameters["spp_high"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])
        Noise_Q = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_Q_high += Noise_Q          
        Noise_U = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
        patch_U_high += Noise_U
    
    if 'PS' in Simulation_Parameters["contaminants"]:
        total_map_Q = patch_Q + Qmap
        total_map_U = patch_U + Umap

        total_map_Q_low = patch_Q_low + Qmap_low
        total_map_U_low = patch_U_low + Umap_low
        
        total_map_Q_high = patch_Q_high + Qmap_high
        total_map_U_high = patch_U_high + Umap_high
    
    else:
        total_map_Q = patch_Q
        total_map_U = patch_U

        total_map_Q_low = patch_Q_low
        total_map_U_low = patch_U_low
        
        total_map_Q_high = patch_Q_high
        total_map_U_high = patch_U_high

    
    return total_map_Q_low, total_map_U_low, total_map_Q, total_map_U, total_map_Q_high, total_map_U_high, label_Q, label_U, label_Q_low, label_U_low, label_Q_high, label_U_high

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def write2h5(total_maps_list, label_list, labels_list_low, labels_list_high, out_file):
    import h5py
    
    f = h5py.File(out_file,'w')
    
    f.create_dataset('M', data = total_maps_list)
    f.create_dataset('M0', data = label_list)
    #f.create_dataset('ML', data = labels_list_low)
    #f.create_dataset('MH', data = labels_list_high)
        
    f.close()
        
    pass

def conversion_factor_Jy_K(freq, pix_size):
    # freq: Planck frequency in GHz
    # pix_size: in arcmin
        
    from astropy.cosmology import Planck15
    from astropy import units as u
    from numpy import pi

    freq = freq * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
    fac = (1. * u.Jy / u.sr).to(u.K, equivalencies = equiv)
    fac = fac.value * (1/((pix_size / 60 / 180 * pi)**2))

    return fac

def CreateRandomCatalogue(nmcat=None, Ns=1e3, ex_rad=6., cut=15.):
    """ Create a random catalogue avoiding source positions
        from the PCCS input catalogue.
        Ns: Number of random positions
        ex_rad: Exclusion radius in arcmin
        cut = galactic cut in degxs
        output glon, glat in degrees
    """
    from astropy.io import fits
    from numpy import random, pi, where, cos, sin, vstack, array, ones, delete, arcsin, ones, pi
    from scipy import spatial as sp

    # Generate random all-sky catalogue
    glon = random.uniform(0, 360, Ns*3)
    glat = arcsin(random.uniform(sin(cut*pi/180.), 1, size=Ns*3)) * 180. / pi
    segno = ones(len(glat), dtype=int)
    segno[random.random(Ns*3) < 0.5] = -1
    glat *= segno # lat in degrees

    # # Remove auto close pairs
    # xyz = lonlat_to_xyz(vstack((glon, glat)).swapaxes(0, 1))
    # T = sp.cKDTree(xyz)
    # chord = 2. * sin(ex_rad / 60. / 180. * pi / 2.)
    # pairs = array(list(T.query_pairs(chord)))
    # if len(pairs) > 0:
    #     glon = delete(glon, pairs[:, 1])
    #     glat = delete(glat, pairs[:, 1])

    if nmcat:
        # Read input catalogue
        # "../../../../ancillary_data/catalogues/COM_PCCS_030_R2.04.fits"
        hdu = fits.open(nmcat)
        iglon = hdu[1].data.field("GLON")
        iglat = hdu[1].data.field("GLAT")

        xyz = lonlat_to_xyz(vstack((iglon, iglat)).swapaxes(0, 1))
        iT = sp.cKDTree(xyz)
        pairs = T.query_ball_tree(iT, chord)

        mask = ones(glon.size, dtype=bool)

        for ii in xrange(len(pairs)):
            if pairs[ii]:
                mask[pairs[ii]] = False

        glon = glon[mask]
        glat = glat[mask]

    return glon[:Ns], glat[:Ns]

def lonlat_to_xyz(pos_lonlat):
    """
    Transform angular position (lon;lat) into cartessian ones (xyz)
    """
    from numpy import hstack, array
    from astropy.coordinates import SkyCoord

    c = SkyCoord(pos_lonlat[:, 0], pos_lonlat[:, 1], frame='icrs',
                 unit='deg')
    x = array(c.cartesian.x, ndmin=2).swapaxes(0, 1)
    y = array(c.cartesian.y, ndmin=2).swapaxes(0, 1)
    z = array(c.cartesian.z, ndmin=2).swapaxes(0, 1)

    return hstack((x, y, z))

def cutting_patches_with_ForSE_code(imap, pixelsize, npix, overlap, longitude, latitude):

    # This function cuts patches with ForSE (Krachmalnicoff&Puglisi21) code
    # Avoiding E-to-B leakage in patches below npix=256

    import astropy
    from astropy import units as u
    import numpy as np
    import healpy as hp
    import warnings
    warnings.filterwarnings("ignore")
    from projection_tools import make_mosaic_from_healpix_with_random_positions

    patches = make_mosaic_from_healpix_with_random_positions(imap, pixelsize.to(u.deg) , npix, overlap= overlap, lon=longitude, lat=latitude)

    Map = np.zeros([npix,npix])

    Map[:,:] = patches[:,:]
    
    return Map


    # apply final smoothing (in arcmin) if needed

def apply_smoothing(infile, outfile, in_fwhm, out_fwhm, pixsize):
    import h5py
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    inp_file = h5py.File(infile, 'r')
    inputs = inp_file["M"][:,:,:,:]
    labels = inp_file["M0"][:,:,:,:]
    """

    plt.figure(1)
    plt.imshow(inputs[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_100.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(inputs[0,:,:,1]), plt.colorbar()
    plt.savefig('Map_U_143.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(inputs[0,:,:,2]), plt.colorbar()
    plt.savefig('Map_U_217.pdf')
    plt.close()


    plt.figure(1)
    plt.imshow(labels[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_label.pdf')
    plt.close()
    """
    sigma = np.zeros(np.size(in_fwhm))
    smo_inputs = inputs
    smo_labels = labels
    
    for i in range(len(in_fwhm)):
        sigma[i] = np.sqrt((out_fwhm**2.) - (in_fwhm[i]**2.))# fwhm in and out in arcsec
        sigma[i] = sigma[i] * 60. / (2. * np.sqrt(2. * np.log(2.))) # fwhm in arcsec (30 X 60)?
        sigma[i] = sigma[i] / pixsize # pixsize in arcsec

    print(sigma)

    for i in range(len(inputs)):
        smo_inputs[i,:,:,0] = gaussian_filter(inputs[i,:,:,0], sigma[0])
        smo_inputs[i,:,:,1] = gaussian_filter(inputs[i,:,:,1], sigma[1])
        smo_inputs[i,:,:,2] = gaussian_filter(inputs[i,:,:,2], sigma[2])
        smo_labels[i,:,:,0] = gaussian_filter(labels[i,:,:,0], sigma[1])
    """ 
    plt.figure(1)
    plt.imshow(smo_inputs[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_100_smo.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(smo_inputs[0,:,:,1]), plt.colorbar()
    plt.savefig('Map_U_143_smo.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(smo_inputs[0,:,:,2]), plt.colorbar()
    plt.savefig('Map_U_217_smo.pdf')
    plt.close()


    plt.figure(1)
    plt.imshow(smo_labels[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_label_smo.pdf')
    plt.close()
    """  
    write2h5(smo_inputs, smo_labels, outfile)

    pass


def main():

    make_parallel_simu4CMB('./sim_cfreq143.par', './Validation', noise = False, QU=False)
    
if __name__ == "__main__":
        main()
