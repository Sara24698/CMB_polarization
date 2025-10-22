import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def estimating_theoretical_power_spectrum_all_sky(Archivo_fits):
        
    CMB = hp.read_map(Archivo_fits, (0,1,2))
    
    # fwhm 0.5 deg in rad
    
    Cls = hp.sphtfunc.anafast(CMB, lmax = 3000)
    
    l = np.arange(0, 3001)
    
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


l_th_all_sky_Planck, EE_th_all_sky_Planck, BB_th_all_sky_Planck = estimating_theoretical_power_spectrum_all_sky('./CMB_143GHZ_SimPlanck.fits')


plt.plot(l_th_all_sky_Planck, BB_th_all_sky_Planck, label='Planck')
plt.legend()
plt.ylabel(r'$\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
plt.xlabel(r'$\mathcal{l}$')
plt.yscale('log')
plt.xscale('log')
plt.xlim(10, 3000)
plt.ylim(0.00001, 0.1)
plt.savefig('Espectro_B_143_teorico')


       