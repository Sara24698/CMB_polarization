#! /usr/bin/env python
def Nsources(hor, ver, Sn=-5., Sx=2.0):
    """Total number of sources in the area
    INPUT:  hor[log10(S); Jy]
            ver[dN/dlogS; sr^-1]
            Sn[log10(Smin); Jy]
            Sx[log10(Smax); Jy]
            Area[sr]
    OUTPUT: Ntotal sources in the area
    """
    from numpy import sum, around, where, isnan

    w = where((hor >= Sn) & (hor <= Sx) & ~(isnan(ver)))
    hbin = abs(hor[0] - hor[1])
    Ntotal = around(sum(ver[w]) * hbin)
    return Ntotal


def pkestimator(Imap, Area=1):
    from numpy import fft, indices, hypot, array, size, sqrt
    from numpy import argsort, mean, append, zeros, unique, pi

    # Imap = Imap / mean(Imap.flat) - 1
    Pmap = abs(fft.fftshift(fft.ifft2(Imap))) ** 2
    x, y = indices(Pmap.shape)
    center = array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = hypot(x - center[0], y - center[1])
    del x, y

    ind = argsort(r.flat)
    r = r.flat[ind]
    Pmap = Pmap.flat[ind]

    u, ind = unique(r, return_index=True)
    ind = append(ind, size(Pmap))
    Pk = zeros(u.shape, dtype=float)
    for i in range(size(ind) - 1):
        Pk[i] = mean(Pmap[ind[i]:ind[i + 1]])

    return u * 2.0 * pi / sqrt(Area), Pk


def corrmap(npix, Ns, ik0, ipk0, Area=1.0):
    from numpy import random, fft, indices, mean, sqrt, floor
    from numpy import array, hypot, pi, interp, log10, where

    random.RandomState()
    Rmap = random.poisson(Ns / npix ** 2, (npix, npix))
    Rmap = Rmap / mean(Rmap.flat) - 1
    rk0, rpk0 = pkestimator(Rmap, Area)

    Fmap = fft.fftshift(fft.ifft2(Rmap))
    x, y = indices(Rmap.shape)
    center = array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    k = hypot(x - center[0], y - center[1]) * 2.0 * pi / sqrt(Area)
    del x, y, Rmap

    rpk = sqrt(10 ** interp(log10(k), log10(rk0), log10(rpk0)))
    ipk = sqrt(10 ** interp(log10(k), log10(ik0), log10(ipk0)))
    Fmap = fft.ifftshift(Fmap / (rpk + 0j) * (ipk + 0j))
    Dmap = (fft.fft2(Fmap).real + 1) * Ns / npix ** 2
    del Fmap
    Cmap = floor(Dmap)
    r = random.rand(Dmap.shape[0], Dmap.shape[1])
    Cmap = where(Dmap - Cmap > r, Cmap + 1, Cmap)
    return Cmap, Dmap


def rand_flux(Ns, hor, ver, Sn=-5., Sx=2.):
    from numpy import random, interp, hstack

    Sx = min(Sx, max(hor[ver * Ns > 0.01]))
    ver = ver[(hor >= Sn) & (hor < Sx)]
    hor = hor[(hor >= Sn) & (hor < Sx)]
    Nr = int(max((Ns * 3, 1e6)))
    x = Sn + (Sx - Sn) * random.rand(Nr)
    y = min(ver) + (max(ver) - min(ver)) * random.rand(Nr)
    y0 = interp(x, hor[::-1], ver[::-1])
    S = x[y < y0]
    if (S.size < Ns):
        while (S.size < Ns):
            x = Sn + (Sx - Sn) * random.rand(Nr)
            y = min(ver) + (max(ver) - min(ver)) * random.rand(Nr)
            y0 = interp(x, hor[::-1], ver[::-1])
            S = hstack((S, x[y < y0]))
        S = S[:Ns]
    else:
        S = S[:Ns]
    return 10 ** S


def fluxassoc(Dmap, hor, ver, Sn=-5., Sx=2., Slim=None):
    """ Associate flux densitiy values from model counts to
        a density map.
    """
    from numpy import where, empty, unravel_index, column_stack, transpose
    from numpy import random, floor, hstack, vstack

    sh = Dmap.shape
    Dmap = Dmap.reshape(-1)
    Dmap = where(Dmap > 0, Dmap, 0)
    Smap = Dmap.copy() * 0.
    Smapp = Dmap.copy() * 0.

    Sx = min(Sx, max(hor[ver > 1e-6]))
    Scat = None
    Scatp = None

    Nstot = Dmap.sum().astype('int64')
    # print("Simulated sources: {0:}".format(Nstot))
    Ns = 0
    if Slim:
        Scat = empty([2, 0])
        Scatp = empty([2, 0])
        Nsc = 0
        while Ns < Nstot:
            w = where(Dmap > 0)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            Dmap[w] -= 1
            wlim = where(S < Slim)[0]
            Smap[w[wlim]] += S[wlim]
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Smapp[rpos] += S[wlim]

            wlim = where(S > Slim)[0]
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Scat = hstack((Scat, vstack((w[wlim], S[wlim]))))
            Scatp = hstack((Scatp, vstack((rpos, S[wlim]))))
            Nsc += wlim.size
            Ns += S.size
        # print("# sources in cat: {0:}".format(Nsc))
        Scat = Scat.swapaxes(1, 0)
        idx = unravel_index(Scat[:, 0].astype(int), sh)
        Scat = column_stack((transpose(idx), Scat[:, 1]))
        Scatp = Scatp.swapaxes(1, 0)
        idx = unravel_index(Scatp[:, 0].astype(int), sh)
        Scatp = column_stack((transpose(idx), Scatp[:, 1]))
    else:
        while Ns < Nstot:
            w = where(Dmap > 0.)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            # print(S.size, w.size)
            Dmap[w] -= 1
            Smap[w] += S
            rpos = floor(random.rand(w.size) * Smapp.size).astype('int64')
            Smapp[rpos] += S
            Ns += S.size
    return Smap.reshape(sh), Smapp.reshape(sh), Scat, Scatp


def fluxassoc_pol(Dmap, hor, ver, Sn=-5., Sx=2., Slim=None, pol=[1., 0.5]):
    """ Associate flux densitiy values from model counts to
        a density map.
    """
    from numpy import where, empty, unravel_index, column_stack, transpose
    from numpy import random, hstack, vstack, floor

    sh = Dmap.shape
    Dmap = Dmap.reshape(-1)
    Dmap = where(Dmap > 0, Dmap, 0)
    Smap = Dmap.copy() * 0.
    Smapp = Dmap.copy() * 0.
    Pmap = Dmap.copy() * 0.
    Pmapp = Dmap.copy() * 0.

    Sx = min(Sx, max(hor[ver > 1e-6]))
    Scat = None
    Scatp = None

    Nstot = Dmap.sum().astype('int64')
    # print("Simulated sources: {0:}".format(Nstot))
    Ns = 0
    if Slim:
        Scat = empty([3, 0])
        Scatp = empty([3, 0])
        Nsc = 0
        while Ns < Nstot:
            w = where(Dmap > 0)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            P = random.lognormal(pol[0], pol[1], w.size) / 100.
            Dmap[w] -= 1
            wlim = where(S < Slim)[0]
            Smap[w[wlim]] += S[wlim]
            Pmap[w[wlim]] += S[wlim] * P[wlim]
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Smapp[rpos] += S[wlim]
            Pmapp[rpos] += S[wlim] * P[wlim]

            wlim = where(S > Slim)[0]
            Scat = hstack((Scat, vstack((w[wlim], S[wlim], P[wlim]))))
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Scatp = hstack((Scatp, vstack((rpos, S[wlim], P[wlim]))))
            Nsc += wlim.size
            Ns += S.size
        # print("# sources in cat: {0:}".format(Nsc))
        Scat = Scat.swapaxes(1, 0)
        idx = unravel_index(Scat[:, 0].astype(int), sh)
        Scat = column_stack((transpose(idx), Scat[:, 1:]))
        Scatp = Scatp.swapaxes(1, 0)
        idx = unravel_index(Scatp[:, 0].astype(int), sh)
        Scatp = column_stack((transpose(idx), Scatp[:, 1:]))
    else:
        while Ns < Nstot:
            w = where(Dmap > 0.)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            P = random.lognormal(pol[0], pol[1], w.size) / 100.
            Dmap[w] -= 1
            Smap[w] += S
            Pmap[w] += S * P
            rpos = floor(random.rand(w.size) * Smapp.size).astype('int64')
            Smapp[rpos] += S
            Pmapp[rpos] += S * P
            Ns += S.size
    return ((Smap.reshape(sh), Pmap.reshape(sh)),
            (Smapp.reshape(sh), Pmapp.reshape(sh)), Scat, Scatp)


def fluxassoc_TQU(Dmap, hor, ver, Sn=-5., Sx=2., Slim=None, pol=[1., 0.5]):
    """ Associate flux densitiy values from model counts to
        a density map.
    """
    from numpy import where, empty, unravel_index, column_stack, transpose
    from numpy import random, hstack, vstack, floor, pi, cos, sin

    sh = Dmap.shape
    Dmap = Dmap.reshape(-1)
    Dmap = where(Dmap > 0, Dmap, 0)
    Smap = Dmap.copy() * 0.
    Smapp = Dmap.copy() * 0.
    Pmap = Dmap.copy() * 0.
    Pmapp = Dmap.copy() * 0.

    Qmap = Dmap.copy() * 0.
    Qmapp = Dmap.copy() * 0.
    Umap = Dmap.copy() * 0.
    Umapp = Dmap.copy() * 0.

    Sx = min(Sx, max(hor[ver > 1e-6]))
    Scat = None
    Scatp = None

    Nstot = Dmap.sum().astype('int64')
    # print("Simulated sources: {0:}".format(Nstot))
    Ns = 0
    if Slim:
        Scat = empty([6, 0])
        Scatp = empty([6, 0])
        Nsc = 0
        while Ns < Nstot:
            w = where(Dmap > 0)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            P = random.lognormal(pol[0], pol[1], w.size) / 100.
            # P = random.normal(3., 0.75, w.size) / 100. # test to validate with gaussian simulations
            phi = random.rand(w.size) * 2 * pi  
            Dmap[w] -= 1
            wlim = where(S < Slim)[0]
            Smap[w[wlim]] += S[wlim]
            Pmap[w[wlim]] += S[wlim] * P[wlim]
            Qmap[w[wlim]] += S[wlim] * P[wlim] * cos(phi[wlim])  # phi has to be in radians
            Umap[w[wlim]] += S[wlim] * P[wlim] * sin(phi[wlim])
            
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Smapp[rpos] += S[wlim]
            Pmapp[rpos] += S[wlim] * P[wlim]
            Qmapp[rpos] += S[wlim] * P[wlim] * cos(phi[wlim])  # phi has to be in radians
            Umapp[rpos] += S[wlim] * P[wlim] * sin(phi[wlim])

            wlim = where(S > Slim)[0]
            Scat = hstack((Scat, vstack((w[wlim], S[wlim], P[wlim], P[wlim] * cos(phi[wlim]),
                                         P[wlim] * sin(phi[wlim]), phi[wlim]))))
            rpos = floor(random.rand(wlim.size) *
                         Smapp.size).astype('int64')
            Scatp = hstack((Scatp, vstack((rpos, S[wlim], P[wlim], P[wlim] * cos(phi[wlim]),
                                           P[wlim] * sin(phi[wlim]), phi[wlim]))))
            Nsc += wlim.size
            Ns += S.size
        # print("# sources in cat: {0:}".format(Nsc))
        Scat = Scat.swapaxes(1, 0)
        idx = unravel_index(Scat[:, 0].astype(int), sh)
        Scat = column_stack((transpose(idx), Scat[:, 1:]))
        Scatp = Scatp.swapaxes(1, 0)
        idx = unravel_index(Scatp[:, 0].astype(int), sh)
        Scatp = column_stack((transpose(idx), Scatp[:, 1:]))
    else:
        while Ns < Nstot:
            w = where(Dmap > 0.)[0]
            S = rand_flux(w.size, hor, ver, Sn, Sx)
            P = random.lognormal(pol[0], pol[1], w.size) / 100.
            # P = random.normal(3., 0.75, w.size) / 100. # test to validate with gaussian simulations        
            phi = random.rand(w.size) * 2 * pi 
            Dmap[w] -= 1
            Smap[w] += S
            Pmap[w] += S * P
            Qmap[w] += S * P * cos(phi)
            Umap[w] += S * P * sin(phi)
            rpos = floor(random.rand(w.size) * Smapp.size).astype('int64')
            Smapp[rpos] += S
            Pmapp[rpos] += S * P
            Qmapp[rpos] += S * P * cos(phi)
            Umapp[rpos] += S * P * sin(phi)
            
            Ns += S.size
    return ((Smap.reshape(sh), Pmap.reshape(sh), Qmap.reshape(sh), Umap.reshape(sh)),
            (Smapp.reshape(sh), Pmapp.reshape(sh), Qmapp.reshape(sh), Umapp.reshape(sh)), Scat, Scatp)

def corrsky(nu=857, npix=256., pixsize=10., sncmodel="lapi11",
            pkmodel="lapi11", Sn=-5., Sx=0., Slim=1e6,
            zbin=None, pol=None, powlaw=None, wpol=None):
    """ Generate a simulated maps with correlated sources
        INPUT:
            - nu: frequency [GHz]
            - npix: patch size in pixel
            - pixelsize: pixel size [arcsec]
            - sncmodel: source number count model
            - pkmodel: power spectrum model
            - Sn: minimum flux density allowed in log [log10 Jy]
            - Sx: maximum flux density allowed in log [log10 Jy]
            - Slim: Source brigther than Slim goes to a catalogue [Jy]
            - zbin: [zmin,zmax] redhsift range to be simulated
            - pol: polarization info; [mu, sigma] for lognormal pdf
        OUTPUT:
            - Smap: Flux density simulated maps TPQU [npix x npix; Jy]
            - Smapp: Flux density simulated map TPQU(Poisson distributed)
                [npix x npix; Jy]
            - Scat, Scatp: Output catalogues of brithest sources.
    """
    from counts import counts
    from spectra import spectra
    from numpy import pi, random, isnan

    random.seed()

    Area = (pixsize / 3600. * npix / 180. * pi) ** 2
    hor, ver = counts(nu, sncmodel, zbin=zbin, powlaw=powlaw)
    ver[isnan(ver)] = 1e-20
    Ns = Nsources(hor, ver, Sn=Sn, Sx=Sx) * Area
    #print("Expected sources: {0:}".format(Ns.astype('int64')))

    k0, Pk0 = spectra(pkmodel, zbin=zbin, nu=nu, Area=Area, wpol=wpol)
    Cmap = corrmap(npix, Ns, k0, Pk0, Area)[0]
    # if pol:
    #     Smap, Smapp, Scat, Scatp = fluxassoc_pol(Cmap, hor, ver * Area, Sn=Sn,
    #                                               Sx=Sx, Slim=Slim, pol=pol)
    if pol:
        Smap, Smapp, Scat, Scatp = fluxassoc_pol(Cmap, hor, ver * Area, Sn=Sn,
                                                  Sx=Sx, Slim=Slim, pol=pol)
    else:
        Smap, Smapp, Scat, Scatp = fluxassoc(Cmap, hor, ver * Area, Sn=Sn,
                                             Sx=Sx, Slim=Slim)

    if (Scat is not None):
        Scat[:, :2] += random.rand(Scat.shape[0], 2)
        Scatp[:, :2] += random.rand(Scatp.shape[0], 2)

    return Smap, Smapp, Scat, Scatp

def corrsky_TQU(nu=857, npix=256., pixsize=10., sncmodel="lapi11",
            pkmodel="lapi11", Sn=-5., Sx=0., Slim=1e6,
            zbin=None, pol=[1., 0.5], powlaw=None, wpol=None):
    """ Generate a simulated maps with correlated sources
        INPUT:
            - nu: frequency [GHz]
            - npix: patch size in pixel
            - pixelsize: pixel size [arcsec]
            - sncmodel: source number count model
            - pkmodel: power spectrum model
            - Sn: minimum flux density allowed in log [log10 Jy]
            - Sx: maximum flux density allowed in log [log10 Jy]
            - Slim: Source brigther than Slim goes to a catalogue [Jy]
            - zbin: [zmin,zmax] redhsift range to be simulated
            - pol: polarization info; [mu, sigma] for lognormal pdf
        OUTPUT:
            - Smap: Flux density simulated map [npix x npix; Jy]
            - Smapp: Flux density simulated map (Poisson distributed)
                [npix x npix; Jy]
            - Scat, Scatp: Output catalogues of brithest sources.
    """
    from counts import counts
    from spectra import spectra
    from numpy import pi, random, isnan

    random.seed()

    Area = (pixsize / 3600. * npix / 180. * pi) ** 2
    hor, ver = counts(nu, sncmodel, zbin=zbin, powlaw=powlaw)
    ver[isnan(ver)] = 1e-20
    Ns = Nsources(hor, ver, Sn=Sn, Sx=Sx) * Area
    #print("Expected sources: {0:}".format(Ns.astype('int64')))

    k0, Pk0 = spectra(pkmodel, zbin=zbin, nu=nu, Area=Area, wpol=wpol)
    Cmap = corrmap(npix, Ns, k0, Pk0, Area)[0]
    # if pol:
    #     Smap, Smapp, Scat, Scatp = fluxassoc_pol(Cmap, hor, ver * Area, Sn=Sn,
    #                                               Sx=Sx, Slim=Slim, pol=pol)
    if pol:
        Smap, Smapp, Scat, Scatp = fluxassoc_TQU(Cmap, hor, ver * Area, Sn=Sn,
                                                  Sx=Sx, Slim=Slim, pol=pol)
    else:
        Smap, Smapp, Scat, Scatp = fluxassoc(Cmap, hor, ver * Area, Sn=Sn,
                                             Sx=Sx, Slim=Slim)

    if (Scat is not None):
        Scat[:, :2] += random.rand(Scat.shape[0], 2)
        Scatp[:, :2] += random.rand(Scatp.shape[0], 2)

    return Smap, Smapp, Scat, Scatp

def main():
    import matplotlib.pyplot as plt
    from numpy import log10

    Smap, Smapp = corrsky_jm(nu=217, npix=128)
    # plt.imshow(log10(Smapp + 1e-6), interpolation='none')
    # plt.colorbar()
    # plt.figure()
    # plt.plot(Scat[:, 0], Scat[:, 1], 'r.')
    # plt.plot(Scatp[:, 0], Scatp[:, 1], '.')
    # plt.show()

if __name__ == "__main__":
    main()
