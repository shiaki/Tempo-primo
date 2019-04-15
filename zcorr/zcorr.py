#

'''
    Peculiar velocity correction for nearby galaxies using Carrick+15.

    TODO:
        disclaimer,
        warning for multiple values,
        fine-tuning root finding.
        (YQ, 04/14)

        different measurements of redshift. (v3k, heliocentric, etc.)
        (YQ, 04/15)
'''

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import newton
import astropy.coordinates as crd

def galactic_Rn(ra, dec):
    ''' Convert RA/Dec in degrees to a unit vector in Galactic frame. '''
    g = crd.SkyCoord(ra=ra,
                     dec=dec,
                     unit=('deg', 'deg')).transform_to(crd.Galactic)
    l, b = g.l.radian, g.b.radian # Galactic lon, lat
    return np.cos(l) * np.cos(b), \
           np.sin(l) * np.cos(b), \
           np.sin(b)

class ZcorrClass(object):

    ''' Class for peculiar velocity correction '''

    def __init__(self, h=0.6736):

        # read velocity file.
        vpec_arr = np.load(os.path.join(os.path.dirname( \
                os.path.abspath(__file__)), 'twompp_velocity.npy'))

        # calculate radial velocity.
        arr_dim = (257, 257, 257)
        u_gr, a_gr = np.ones(257), np.linspace(-200., 200., 257)
        X_gr = np.reshape(np.outer(u_gr, np.outer(u_gr, a_gr)), arr_dim)
        Y_gr = np.reshape(np.outer(u_gr, np.outer(a_gr, u_gr)), arr_dim)
        Z_gr = np.reshape(np.outer(a_gr, np.outer(u_gr, u_gr)), arr_dim)
        R_gr = np.sqrt(X_gr ** 2 + Y_gr ** 2 + Z_gr ** 2) # radius.
        Vr_pec = (vpec_arr[0] * X_gr \
                + vpec_arr[1] * Y_gr \
                + vpec_arr[2] * Z_gr) / R_gr

        # create 3-d interpolator for Vr_pec
        self._Vr_pec_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), Vr_pec)

        # write other parameters
        self._h = h
        self._c = 2.99792458e5 # in km/s
        self._hsq = h ** 2
        self._R_ax = np.linspace(0., 200., 512)

    def _dc(ra, dec, z_obs,):

        # otherwise: calculate V_pec corrected redshift.
        cx, cy, cz = galactic_Rn(ra, dec)
        xi_cart = np.outer(self._R_ax, [cx, cy, cz]) # GC, cartesian, Mpc/h
        Vp_R = self._Vr_pec_intp(xi_cart) # Vr_pec along xi_cart (LOS)

        # root-finding.
        v_obs = self._c * z_obs
        vrp_intp = interp1d(self._R_ax, Vp_R, fill_value='extrapolate')
        rfc = lambda d: v_obs - (1.e2 * self._hsq) * d - vrp_intp(d)
        d_opt = newton(rfc, z_obs * self._c / (100. * (h ** 2)),)

        return d_opt

    def zcorr(self, ra, dec, z_obs,): # ** assuming z_obs in cmb frame.

        ''' Peculiar velocity-corrected redshift. '''

        # beyond the range: return original z.
        if z_obs > 2.e2 * (1.e2 * self._hsq) / self._c:
            return z_obs # about 0.03

        d_opt = self._dc(ra, dec, z_obs)
        return d_opt * (1.e2 * self._hsq / self._c)

    def dist(self, ra, dec, z_obs,):

        ''' Peculiar velocity corrected distance. '''

        # beyond the range: return original z.
        if z_obs > 2.e2 * (1.e2 * self._hsq) / self._c:
            return z_obs * self._c / (1.e2 * self._h)
            # ** only works for local universe

        d_opt = self._dc(ra, dec, z_obs)
        return d_opt * self._h

Zcorr = ZcorrClass()
