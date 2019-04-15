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

    # interpolator
    Vr_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), V_r,)
    d_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), density_arr,)

    H = 72. # or 69 if you like.

    # make a figure
    if 1:

        # convert galactocentric to cartesian
        cx, cy, cz = galactic_Rn(138., 33.)
        R = np.linspace(0., 200., 256)
        R_vec = np.dstack((cx * R, cy * R, cz * R))
        Vr_R, d_R = Vr_intp(R_vec), d_intp(R_vec)

        fig = plt.figure(figsize=(8., 4.))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(R, Vr_R[0])
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(R, d_R[0])
        ax2.axhline(y=0., ls='dashed', c='0.5')

        ax1.set_ylabel('$\Delta$ V')
        ax2.set_ylabel('$\delta$')
        ax2.set_xlabel('Distance [Mpc/h]')
        plt.savefig('fig1.pdf')
        plt.show()

    # search for triple value cases
    if 1:
        R = np.linspace(0., 200., 512)
        delta_R = np.median(np.diff(R))
        R_mp = (R[1:] + R[:-1]) / 2.
        Rc_mv, i_pt = np.empty((8192, 4)), 0
        while True:
            c = np.random.randn(3)
            c = c / np.sqrt(np.sum(c ** 2))
            Vr_R = Vr_intp(np.outer(R, c)) + R * H
            V_diff = np.diff(Vr_R) / delta_R
            if np.sum(V_diff < 0) > 0:
                d_med = np.random.choice(R_mp[V_diff < 0.])
                Rc_mv[i_pt, :] = (c[0], c[1], c[2], d_med)
                i_pt += 1
            if i_pt == Rc_mv.shape[0]:
                break
        Rc_mv = np.array(Rc_mv)
        b = np.arcsin(Rc_mv[:, 2])
        l = np.arctan2(Rc_mv[:, 1], Rc_mv[:, 0])
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
        ax.scatter(l, b, s=1., c=Rc_mv[:, 3],
                   vmin=0., vmax=200., cmap='jet', alpha=0.75)
        ax.set_xlim(np.pi, -np.pi)
        ax.set_ylim(-np.pi / 2., np.pi / 2.)
        plt.savefig('fig2.pdf')
        plt.show()
