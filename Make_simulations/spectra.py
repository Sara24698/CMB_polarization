#! /usr/bin/env python
def get_spectrum_lapi11(spec_path, nu=857, zbin=None, wpol=None):
    from numpy import loadtxt
    freq = [30, 44, 70, 100, 143, 217, 353, 545, 857, 1200]
    idx = abs(freq - nu).argmin()
    namefile = spec_path + "cl_lapi_{0:}GHz.dat".format(freq[idx])
    k, Pk = loadtxt(namefile, comments='%', unpack=True)
    return k, 10 ** Pk


def get_spectrum_zlapi(spec_path, nu=None, zbin=[1.2, 1.6], wpol=None):
    from numpy import loadtxt, array
    zn = array([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])
    zx = array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
    zbin = array(zbin)
    idn = abs(zbin[0] - zn).argmin()
    idx = abs(zbin[1] - zx).argmin()
    namefile = "Hermes_857_350_{0:3.1f}_{1:3.1f}_PS.dat".format(
        zn[idn], zx[idx])
    namefile = spec_path + namefile
    k, Pknorm = get_spectrum_lapi11(857, spec_path)
    k, Pk = loadtxt(namefile, unpack=True, usecols=(0, 3))
    return k * 180 * 60, Pk / max(Pk) * max(Pknorm)


def get_spectrum_polynomial(spec_path, nu=None, zbin=None, wpol=[1., 0.8]):
    from numpy import pi, arange
    from scipy.special import gamma
    e = wpol[1]  # w(theta)= (theta/theta0) ** - e
    theta0 = wpol[0] ** (1. / e) * pi / 180.
    Pk0 = (2 * pi * theta0 ** e * 2. ** (1 - e) * gamma(1 - e / 2.) /
           gamma(e / 2.))

    K = 10. ** arange(0, 5, 0.1)
    return K, Pk0 * K ** (e - 2.)


def spectra(spectrum, nu=857, zbin=[1.2, 1.6], Area=1.0,
            spec_path="./spectra/", wpol=None):
    """
    Call the correct power spectrum files
    and homogenize the output.
    INPUT : spectrum: type
    OUTPUT: multipole and Pk
    """
    from numpy import log10, arange, interp, append, int_
    import numpy.polynomial.polynomial as poly
    models = {"lapi11": get_spectrum_lapi11,
              "zlapi": get_spectrum_zlapi,
              "polynomial": get_spectrum_polynomial}
    k, Pk = models[spectrum](spec_path, nu=int_(nu), zbin=zbin, wpol=wpol)
    # Extension to lower/higher values for all-sky/higher resolution
    p = poly.polyfit(log10(k[:10]), log10(Pk[:10]), 1)
    xx_low = arange(-1, log10(min(k)), 0.1)
    yy_low = poly.polyval(xx_low, p)
    xx0 = append(xx_low, log10(k))
    yy0 = append(yy_low, log10(Pk))
    #
    p = poly.polyfit(xx0[-10:], yy0[-10:], 1)
    xx_high = arange(max(xx0), 6.1, 0.1)
    yy_high = poly.polyval(xx_high, p)
    xx0 = append(xx0, xx_high)
    yy0 = append(yy0, yy_high)

    xx = arange(-1, 6.1, 0.1)
    yy = 10 ** interp(xx, xx0, yy0)
    return 10 ** xx, yy / Area
