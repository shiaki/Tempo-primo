#

'''
    Read reconstructed velocity field.
'''

import sys, os
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    # read configuration.
    with open('cfg.json', 'r') as f:
        cfg = json.load(f)

    # read density and velocity grids.
    density_arr = np.load(cfg['density_arr_file'])
    velocity_arr = np.load(cfg['velocity_arr_file'])

    # calculate radial velocity.
    arr_dim = (257, 257, 257)
    u_gr, a_gr = np.ones(257), np.linspace(-200., 200., 257)
    X_gr = np.reshape(np.outer(u_gr, np.outer(u_gr, a_gr)), arr_dim)
    Y_gr = np.reshape(np.outer(u_gr, np.outer(a_gr, u_gr)), arr_dim)
    Z_gr = np.reshape(np.outer(a_gr, np.outer(u_gr, u_gr)), arr_dim)
    R_gr = np.sqrt(X_gr ** 2 + Y_gr ** 2 + Z_gr ** 2) # radius.
    V_r = velocity_arr[0] * X_gr / R_gr \
        + velocity_arr[1] * Y_gr / R_gr \
        + velocity_arr[2] * Z_gr / R_gr

    # convert galactocentric to cartesian
    Vr_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), V_r,)
    d_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), density_arr,)

    cx, cy, cz = galactic_Rn(138., 33.)
    R = np.linspace(0., 200., 256)
    Vr_R = Vr_intp(np.dstack((cx * R, cy * R, cz * R)))
    d_R = d_intp(np.dstack((cx * R, cy * R, cz * R)))
    plt.plot(R * 72., Vr_R[0])
    plt.plot(R * 72., d_R[0] * 100.), plt.show()
