#

'''
    Peculiar velocity correction for nearby galaxies using Carrick+15.

    TODO: (YQ, 04/14)
        disclaimer,
        warning for multiple values,
        fine-tuning root finding.
'''

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
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
        self._hsq = h ** 2

    def zcorr(self, ra, dec, z_obs):
        pass # TODO

    def dist(self, ra, dec, z_obs):
        pass # TODO

Zcorr = ZcorrClass()
