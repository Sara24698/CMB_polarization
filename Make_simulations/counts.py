#! /usr/bin/env python
def counts_interp_freq(nu, freqn, freqx, ccn, ccx):
    """Interpolate two counts models """
    from numpy import interp, transpose, vstack, array
    if len(ccn) == len(ccx):
        hor = ccn[:, 0]
        vern = ccn[:, 1]
        verx = ccx[:, 1]
    elif len(ccn) > len(ccx):
        hor = ccn[:, 0]
        vern = ccn[:, 1]
        verx = interp(hor, ccx[:, 0], ccx[:, 1])
    else:
        hor = ccx[:, 0]
        verx = ccx[:, 1]
        vern = interp(hor, ccn[:, 0], ccn[:, 1])
    ver = array([interp(nu, array([freqn, freqx]), x) for x in
                transpose(vstack((vern, verx)))])
    return hor, 10 ** ver


def poiss_err(H, sigma=1.):
    """ poiss_err estimate the poissonian errors
        Based on Gehrels (1986)
    """
    from numpy import sqrt, where, NaN

    # Upper limit
    Errup = lambda H, sigma: (H + 1.) * (
        1. - 1. / (9. * (H + 1.)) + sigma / (3. * sqrt(H + 1.))) ** 3.
    # Lower limit
    Errlow = lambda H, sigma: H * (
        1. - 1. / (9. * H) - sigma / (3. * sqrt(H))) ** 3.

    Err_up = where(H > 0, Errup(H, sigma), NaN)
    Err_low = where(H > 0, Errlow(H, sigma), NaN)
    return Err_up, Err_low


def diff_counts(S, Sn=-5., Sx=2.0, lbin=0.1, integral=False, c_area=1.):
    """ DIFF_COUNTS estimate the differential counts of the flux density
        list provided as input. dN/dlogS [sr^-1]
        INPUT:
            S: list of flux densities in Jy
        OPTIONS:
            Sn: minimum flux density in log10(Jy)
            Sx: maximum flux density in log10(Jy)
            lbin: step in logaritmic scale
            integral: flag to provide the integral counts instead
            c_area: area in sr
    """
    from numpy import arange, histogram, log10

    if integral:
        n, h = histogram(log10(S), bins=arange(Sn, Sx + lbin, lbin))
        n = n[::-1].cumsum()[::-1]
        h = h[:-1] + lbin / 2.
        e_nu, e_nl = poiss_err(n)
        n = n.astype('float') / c_area
    else:
        n, h = histogram(log10(S), bins=arange(Sn, Sx + lbin, lbin))
        n = n
        h = h[:-1] + lbin / 2.
        e_nu, e_nl = poiss_err(n)
        n = n.astype('float')
        #n = n / (lbin * log(10.) * power(power(10., h), -3. / 2.))
        #e_nu = e_nu / (lbin * log(10.) * power(power(10., h), -3. / 2.))
        #e_nl = e_nl / (lbin * log(10.) * power(power(10., h), -3. / 2.))
        n = n / lbin / c_area
        e_nu = e_nu / lbin / c_area
        e_nl = e_nl / lbin / c_area

    return h, n, e_nu, e_nl


def counts_dezotti(nu, subpop=4, zbin=None, counts_path='./counts/', *args):
    """Load de Zotti et al. (2005) source number counts """
    from numpy import loadtxt, log, argsort
    # Allowed frequencies for each model
    freq0 = [1.4, 5, 10, 11, 15, 20, 23, 30, 33, 41, 44, 61, 70, 94,
             100, 143, 217, 353, 545, 857]
    if subpop:
        col = subpop - 1
    else:
        col = 3
    if nu in freq0:
        namefile = counts_path + "cr{0:}dezotti.dat".format(nu)
        hor, ver = loadtxt(namefile, comments='%',
                           usecols=(0, col), unpack=True)
        ver = 10 ** ver
    else:  # When the input frequency is between two of the allowed ones
        freqx = next((x for x in freq0 if x > nu), None)
        freqn = freq0[freq0.index(freqx) - 1]
        namefile = counts_path + "cr{0:}dezotti.dat".format(freqn)
        ccn = loadtxt(namefile, comments='%', usecols=(0, col))
        namefile = counts_path + "cr{0:}dezotti.dat".format(freqx)
        ccx = loadtxt(namefile, comments='%', usecols=(0, col))
        hor, ver = counts_interp_freq(nu, freqn, freqx, ccn, ccx)
    ver = ver / (10 ** hor) ** 1.5 * log(10.)
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_tucci_old(nu, subpop=2, zbin=None, counts_path='./counts/', *args):
    """Load de Tucci et al. (2011) source number counts """
    from numpy import loadtxt, argsort
    # Allowed frequencies for each model
    freq0 = [70, 100, 143, 217]
    if subpop:
        col = subpop - 1
    else:
        col = 1
    if nu in freq0:
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(nu)
        hor, ver = loadtxt(namefile, comments='%',
                           usecols=(0, col), unpack=True)
        ver = 10 ** ver
    else:  # When the input frequency is between two of the allowed ones
        freqx = next((x for x in freq0 if x > nu), None)
        freqn = freq0[freq0.index(freqx) - 1]
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(freqn)
        ccn = loadtxt(namefile, comments='%', usecols=(0, col))
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(freqx)
        ccx = loadtxt(namefile, comments='%', usecols=(0, col))
        hor, ver = counts_interp_freq(nu, freqn, freqx, ccn, ccx)
    hor -= 3
    # ver = ver * (10 ** hor) ** (-1.5) * log(10.)
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_tucci(nu, subpop=2, zbin=None, counts_path='./counts/', *args):
    """Load de Tucci et al. (2011) source number counts """
    from numpy import loadtxt, argsort, log10, log
    # Allowed frequencies for each model
    freq0 = [60, 69, 150, 220, 353, 550, 900]
    if nu < min(freq0):
        nu = min(freq0)

    if subpop:
        col = subpop - 1
    else:
        col = 1

    if nu in freq0:
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(nu)
        hor, ver = loadtxt(namefile, comments='%',
                           usecols=(0, col), unpack=True)
        hor = log10(hor)
    else:  # When the input frequency is between two of the allowed ones
        freqx = next((x for x in freq0 if x > nu), None)
        freqn = freq0[freq0.index(freqx) - 1]
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(freqn)
        ccn = loadtxt(namefile, comments='%', usecols=(0, col))
        namefile = counts_path + "tucci11/ns{0:}_C2Ex.dat".format(freqx)
        ccx = loadtxt(namefile, comments='%', usecols=(0, col))
        hor, ver = counts_interp_freq(nu, freqn, freqx, log10(ccn), log10(ccx))
    ver = ver * (10 ** hor) * log(10.)
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_IRLT(nu, subpop=2, zbin=None, counts_path='./counts/', *args):
    """Load de Lapi+11 IR Late Type source number counts """
    from numpy import loadtxt, log10, argsort, around, pi, array
    # Allowed frequencies for each model
    lambda0 = [100, 160, 250, 350, 500, 850, 1100, 1400, 2000, 3000]
    ilambda = around(3e8 / float(nu) * 1e6 / 1e9)
    # Normalization factor from Negrello et al. 2013
    NN = array([1., 1., 1., 1.6, 2.1, 2.7, 3.3, 4., 5., .5])
    if subpop:
        col = (0, subpop - 1)
    else:
        col = (0, 1, 2)
    namefile = "latetype/counts{0:}_spirals_starbursts.dat"
    if ilambda.astype(int) in lambda0:
        idx = lambda0.index(ilambda)
        namefile = counts_path + namefile.format(lambda0[idx])
        cc = loadtxt(namefile, comments='%', usecols=col)
        hor = cc[:, 0]
        ver = (cc[:, 1] + cc[:, 2]) * (180. / pi) ** 2 * NN[idx]
    else:  # When the input frequency is between two of the allowed ones
        lambdax = next((x for x in lambda0 if x > ilambda), None)
        lambdan = lambda0[lambda0.index(lambdax) - 1]
        namefile = "latetype/counts{0:}_spirals_starbursts.dat"
        namefile = (counts_path + namefile.format(lambdan))
        ccn = loadtxt(namefile, comments='%', usecols=col)
        ccn = ccn[:, :3]
        ccn[:, 1] = log10((ccn[:, 1] + ccn[:, 2]) * (180. / pi) ** 2 *
                          NN[lambda0.index(lambdax) - 1])
        ccn = ccn[:, :2]
        namefile = "latetype/counts{0:}_spirals_starbursts.dat"
        namefile = (counts_path + namefile.format(lambdax))
        ccx = loadtxt(namefile, comments='%', usecols=col)
        ccx = ccx[:, :3]
        ccx[:, 1] = log10((ccx[:, 1] + ccx[:, 2]) * (180. / pi) ** 2 *
                          NN[lambda0.index(lambdax)])
        ccx = ccx[:, :2]
        hor, ver = counts_interp_freq(ilambda, lambdan, lambdax, ccn, ccx)
    hor -= 3
    #ver = ver * (10 ** hor) ** (-1.5) * log(10.)
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_lapi11(nu, subpop=2, zbin=None, counts_path='./counts/', *args):
    """Load Lapi et al. (2011) source number counts """
    from numpy import loadtxt, log, argsort
    # Allowed frequencies for each model
    freq0 = [30, 44, 70, 100, 143, 217, 353, 600, 857, 1200]
    xia = [0.55, 0.55, 0.55, 0.55, 0.55, 0.71, 0.81, 1, 1, 1]
    col_read = [1, 1, 1, 1, 1, 1, 1, 2, 2, 1]
    if nu in freq0:
        idx = freq0.index(nu)
        if subpop:
            col = subpop - 1
        else:
            col = col_read[idx]
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(nu)
        hor, ver = loadtxt(namefile, comments='%',
                           usecols=(0, col), unpack=True)
        ver = 10 ** ver
    else:  # When the input frequency is between two of the allowed ones
        freqx = next((x for x in freq0 if x > nu), None)
        idn = freq0.index(freqx) - 1
        freqn = freq0[idn]
        col = col_read[idn]
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(freqn)
        ccn = loadtxt(namefile, comments='%', usecols=(0, col))

        idx = freq0.index(freqx)
        col = col_read[idx]
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(freqx)
        ccx = loadtxt(namefile, comments='%', usecols=(0, col))
        hor, ver = counts_interp_freq(nu, freqn, freqx, ccn, ccx)
    hor -= 3
    ver = ver * (10 ** hor) ** (-1.5) * log(10.) * xia[idx]
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_lapilen11(nu, subpop=2, zbin=None, counts_path='./counts/', *args):
    """Load Lapi et al. (2011) source number counts """
    from numpy import loadtxt, log, log10, argsort, nan
    # Allowed frequencies for each model
    freq0 = [30, 44, 70, 100, 143, 217, 353, 600, 857, 1200]
    xia = [0.55, 0.55, 0.55, 0.55, 0.55, 0.71, 0.81, 1, 1, 1]
    if nu in freq0:
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(nu)
        cc = loadtxt(namefile, comments='%')
        hor = cc[:, 0]
        if len(cc[0, :]) > 3:
            ver = 10 ** cc[:, 3]
        else:
            ver = 10 ** cc[:, 2] - 10 ** cc[:, 1]
            ver[ver < 0] = nan
        idx = freq0.index(nu)
    else:  # When the input frequency is between two of the allowed ones
        freqx = next((x for x in freq0 if x > nu), None)
        freqn = freq0[freq0.index(freqx) - 1]
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(freqn)
        ccn = loadtxt(namefile, comments='%')
        namefile = counts_path + "/lapi2011/counts{0:}GHz.txt".format(freqx)
        ccx = loadtxt(namefile, comments='%')
        if len(ccn[0, :]) < 4:
            lensed = 10 ** ccn[:, 2] - 10 ** ccn[:, 1]
            lensed[lensed < 0] = nan
            ccn[:, 1] = log10(lensed)
        else:
            ccn[:, 1] = ccn[:, 3]
        if len(ccx[0, :]) < 4:
            lensed = 10 ** ccx[:, 2] - 10 ** ccx[:, 1]
            lensed[lensed < 0] = nan
            ccx[:, 1] = log10(lensed)
        else:
            ccx[:, 1] = ccx[:, 3]
        ccn = ccn[:, :2]
        ccx = ccx[:, :2]
        hor, ver = counts_interp_freq(nu, freqn, freqx, ccn, ccx)
        idx = freq0.index(freqx)
    hor -= 3
    # ver = ver * (10 ** hor) ** (-1.5) * log(10.) * xia[idx]
    ver = ver * (10 ** hor) ** (-2.5) * xia[idx]
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_zlapi(nu, subpop=2, zbin=[1.2, 4.0], counts_path='./counts/', *args):
    """
    Load Lapi et al. (2011) source number counts at 857 GHz
    at different redshift bins
    """
    from numpy import loadtxt, argsort, array, pi
    zn = array([1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
    zx = array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
    zbin = array(zbin)
    idn = abs(zbin[0] - zn).argmin()
    idx = abs(zbin[1] - zx).argmin()
    for i in range(idn, idx + 1):
        namefile = "zlapi/perquincho_counts_{0:2.0f}z{1:2.0f}.txt".format(
            zn[i] * 10, zx[i] * 10)
        hor, ver0 = loadtxt(counts_path + namefile, comments='%',
                            usecols=(0, 1), unpack=True)
        if i == idn:
            ver = 10 ** ver0
        else:
            ver += 10 ** ver0
    hbin = abs(hor[0] - hor[1])
    hor -= hbin / 4.
    hor -= 3
    ver *= (180. / pi) ** 2
    I = argsort(hor)[::-1]
    hor = hor[I]
    ver = ver[I]
    return hor, ver


def counts_powlaw(nu, subpop=None, zbin=None, counts_path="./counts/",
                  powlaw=[2., 2.1], *args):
    from numpy import arange, log

    step = 0.1
    hor = arange(-5., 2., step)
    ver = powlaw[0] * (10 ** hor) ** (1-powlaw[1]) * log(10.) / step
    return hor[::-1], ver[::-1]


def counts(nu, population, subpop=None,
           zbin=[1.2, 4.0], counts_path='./counts/', powlaw=[1., 2.5]):
    """
    Call the correct source number counts files
    and homogenize the output.
    INPUT nu:frequency
          population: model
    OUTPUT: logS [Jy]  and dN/dlogS
    """
    from numpy import int_
    # Dict: (namefile, def_col, modifications)
    models = {"dezotti": counts_dezotti,
              "lapi11": counts_lapi11,
              "lapi_len11": counts_lapilen11,
              "tucci": counts_tucci,
              "tucci_old": counts_tucci_old,
              "IRLT": counts_IRLT,
              "zlapi": counts_zlapi,
              "powlaw": counts_powlaw}

    #
    hor, ver = models[population](int_(nu), subpop, zbin, counts_path, powlaw)
    # for x in range(10):
    #     print '{0:4.2f} {1:7.4f}'.format(hor[x], log10(ver[x]))
    return hor, ver
