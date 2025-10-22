import numpy as np
import healpy as hp
import matplotlib.pyplot as plt



def Smooth_FullSkyMap(in_map, out_map, in_fwhm, out_fwhm):
   # Smooth_FullSkyMap("../CMB_maps/Mapa_Q_r0_2_217_GHz.fits","../CMB_maps/Mapa_Q_r0_2_100_GHz", 294.0,579.6)
   
   # fwhm in and out in arcsec

   sigma_smo = np.sqrt((out_fwhm**2.) - (in_fwhm**2.))# fwhm in and out in arcsec
   sigma_smo = (sigma_smo / (3600. * 180.)) * np.pi


   mappa = hp.read_map(in_map)
   smo_map = hp.smoothing(mappa, fwhm = sigma_smo, nest = False)
   hp.write_map(out_map+'.fits', smo_map, nest = False )


Smooth_FullSkyMap("./Mapa_Q_217_GHz.fits","Mapa_Q_143_GHz", 294.0, 433.2)

