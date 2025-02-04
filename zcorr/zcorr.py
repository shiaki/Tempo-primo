#!/usr/bin/env python

'''
    Peculiar velocity correction for nearby galaxies using Carrick+15.

    210113: Major bug fix
'''

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import newton, brentq
import astropy.coordinates as crd

import matplotlib.pyplot as plt

def galactic_Rn(ra, dec):
    ''' Convert RA/Dec in degrees to a unit vector in Galactic frame. '''
    g = crd.SkyCoord(ra=ra,
                     dec=dec,
                     unit=('deg', 'deg')).transform_to(crd.Galactic)
    l, b = g.l.radian, g.b.radian # Galactic lon, lat
    return np.cos(l) * np.cos(b), \
           np.sin(l) * np.cos(b), \
           np.sin(b)

def _helio_to_CMB(cx, cy, cz):
    ''' Correct heliocentric radial velocity to CMB frame
        Ref: ApJ 473, 576, 1996 '''
    cx_ap = -0.0679719430363560
    cy_ap = -0.6622724518847442
    cz_ap =  0.7461735819730095
    dv_ap =  3.71e2
    return dv_ap * (cx * cx_ap + cy * cy_ap + cz * cz_ap)

def helio_to_CMB(ra, dec):
    ''' Correct heliocentric radial velocity to CMB frame
        Ref: ApJ 473, 576, 1996 '''
    return _helio_to_CMB(*galactic_Rn(ra, dec))

class ZcorrClass(object):

    ''' Class for peculiar velocity correction '''

    def __init__(self,):

        # read velocity file.
        vpec_arr = np.load(os.path.join(os.path.dirname( \
                os.path.abspath(__file__)), 'twompp_velocity.npy'))

        # calculate radial velocity.
        a_gr = np.linspace(-200., 200., 257)

        # interpolator for individual components.
        cart_grid_t = (a_gr, a_gr, a_gr)
        self._Vp_x = RegularGridInterpolator(cart_grid_t, vpec_arr[0])
        self._Vp_y = RegularGridInterpolator(cart_grid_t, vpec_arr[1])
        self._Vp_z = RegularGridInterpolator(cart_grid_t, vpec_arr[2])

        # write other parameters
        self._c = 2.99792458e5 # [km / s]
        self._R_ax = np.linspace(0., 200., 1024) # [Mpc / h]

    def _dc(self, ra, dec, z_obs, z_type='helio',):

        ''' Calculate V_pec corrected distance in [Mpc / h] '''

        # input values.
        cx, cy, cz = galactic_Rn(ra, dec)
        v_obs = self._c * z_obs # v: [km / s]

        # convert to v3k if necessary,
        if z_type == 'helio': v_obs += _helio_to_CMB(cx, cy, cz)
        elif z_type == 'cmb': pass
        else: raise RuntimeError('"z_type" not recognized.')

        # if v3k beyond the box,
        if v_obs > 2.e2 * 1.e2: # box size +/- 200 Mpc / h
            return np.array([v_obs / 1.e2])
            # return z_CMB in Mpc / h without correction.

        # find velocity components.
        xi = np.outer(self._R_ax, [cx, cy, cz]) # GC, cartesian, Mpc/h
        Vp_x, Vp_y, Vp_z = self._Vp_x(xi), self._Vp_y(xi), self._Vp_z(xi)
        Vp_R = cx * Vp_x + cy * Vp_y + cz * Vp_z # Vr_pec along xi (LOS)

        # define target function for root-finding
        vrp_intp = interp1d(self._R_ax, Vp_R,
                            fill_value='extrapolate',
                            kind='linear')
        rfc = lambda d: 1.e2 * d + vrp_intp(d) - v_obs
        # d: Mpc / h

        # find intervals for root-finding.
        V_rf = rfc(self._R_ax) # [km / s]
        intv_cond = (V_rf[:-1] * V_rf[1:] <= 0.) * (V_rf[1:] != 0.)
        id_intv = np.arange(1023)[intv_cond]

        # in each interval: find root.
        dopt_vals = [brentq(rfc, \
                            self._R_ax[i_zero], \
                            self._R_ax[i_zero + 1]) \
                     for i_zero in id_intv]

        return np.array(dopt_vals) # in [Mpc / h]

    def zcorr(self, ra, dec, z_obs, z_type='helio'):

        ''' Peculiar velocity-corrected redshift. '''

        d_opt = self._dc(ra, dec, z_obs, z_type=z_type) # [Mpc / h]
        return d_opt * 1.e2 / self._c # redshift.

    def dist(self, ra, dec, z_obs, h=None, z_type='helio',):

        ''' Peculiar velocity corrected distance in Mpc '''

        d_opt = self._dc(ra, dec, z_obs, z_type=z_type)
        return d_opt / (1. if not h else h)

Zcorr = ZcorrClass()
