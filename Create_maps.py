# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import healpy as hl
import pandas as pd


def create_maps_camb(r, frecuencia, Map_Q_U=True):

    total_cls = pd.read_csv('./Cl_files/r_'+str(r)+'_prueba.csv', sep = ";")
    total_cls=np.array(total_cls)


    #Extrae los multipolos y los Cls y los escala
    l=total_cls[:, 0]
    tt=total_cls[:, 1]*2*np.pi/(l*(l+1))*7.43
    ee=total_cls[:, 2]*2*np.pi/(l*(l+1))*7.43
    bb=total_cls[:, 3]*2*np.pi/(l*(l+1))*7.43
    te=total_cls[:, 4]*2*np.pi/(l*(l+1))*7.43

    cls=[tt, ee, bb, te]

    #Define las frecuencias de los mapas
    if frecuencia == 100:
        frecuencia_arcmin = 9.66/60/180*np.pi

    if frecuencia == 143:
        frecuencia_arcmin= 7.22/60/180*np.pi

    if frecuencia == 217:
        frecuencia_arcmin= 4.90/60/180*np.pi

    else:
        print('Esta frecuencia no se usa en este estudio')


    #Saca un mapa total, un mapa en Q y otro en U para cada frecuencia	
    map = hl.sphtfunc.synfast(cls, nside=2048, pol=True, fwhm=frecuencia_arcmin, new=True)
    hl.write_map('Mapa_total_'+str(frecuencia)+'_GHz_'+str(r)+'.fits',map)

    if Map_Q_U==True:
        hl.write_map('Mapa_I_'+str(frecuencia)+'_GHz_'+str(r)+'.fits',map[0])
        hl.write_map('Mapa_Q_'+str(frecuencia)+'_GHz_'+str(r)+'.fits',map[1])
        hl.write_map('Mapa_U_'+str(frecuencia)+'_GHz_'+str(r)+'.fits',map[2])

    else:
        cls, als= hl.sphtfunc.anafast(map, alm=True, pol=True)
        EE = als[1]
        BB = als[2]

        mapE = hl.sphtfunc.alm2map(EE, nside=2048, pol = False, inplace=False)
        mapB = hl.sphtfunc.alm2map(BB, nside=2048, pol = False, inplace=False)

        # Guardar los mapas en FITS
        hl.write_map('Mapa_E_'+str(frecuencia)+'_GHz_'+str(r)+'.fits', mapE, overwrite=True)
        hl.write_map('Mapa_B_'+str(frecuencia)+'_GHz_'+str(r)+'.fits', mapB, overwrite=True)


def create_EB_maps_Planck(ruta):
    mapa_total = hl.read_map(ruta, (0,1,2))
    cls, als= hl.sphtfunc.anafast(mapa_total, alm=True, pol=True)

    EE = als[1]
    BB = als[2]

    mapE = hl.sphtfunc.alm2map(EE, nside=2048, pol = False, inplace=False)
    mapB = hl.sphtfunc.alm2map(BB, nside=2048, pol = False, inplace=False)

    # Guardar los mapas en FITS
    hl.write_map("Map"+"E.fits", mapE, overwrite=True)
    hl.write_map("Map"+"B.fits", mapB, overwrite=True)


create_maps_camb(0.1, 143, Map_Q_U=False)
