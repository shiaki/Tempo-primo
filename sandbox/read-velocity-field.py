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

def _heliocentric_to_CMB(cx, cy, cz):
    cx_ap = -0.0679719430363560
    cy_ap = -0.6622724518847442
    cz_ap =  0.7461735819730095
    dv_ap = 371.
    return dv_ap * (cx * cx_ap + cy * cy_ap + cz * cz_ap)

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

    R_gr = np.sqrt( a_gr[None, None, :] ** 2 \
                  + a_gr[None, :, None] ** 2 \
                  + a_gr[:, None, None] ** 2)
    V_r = ( velocity_arr[0] * a_gr[:, None, None] \
          + velocity_arr[1] * a_gr[None, :, None] \
          + velocity_arr[2] * a_gr[None, None, :] ) / R_gr

    # interpolator
    Vr_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), V_r,)
    d_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), density_arr,)

    Vx_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), velocity_arr[0],)
    Vy_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), velocity_arr[1],)
    Vz_intp = RegularGridInterpolator((a_gr, a_gr, a_gr), velocity_arr[2],)

    H = 72. # or 69 if you like.

    fig_name = 'Coma'
    cx, cy, cz = galactic_Rn(194.9529167, 27.9805556) # Coma

    # fig_name = 'Perseus'
    # cx, cy, cz = galactic_Rn(49.9505, 41.5117) # Perseus

    # fig_name = 'Shapley'
    # cx, cy, cz = galactic_Rn(201.9783333, -31.4922222) # Shapley

    # fig_name = 'Hercules'
    # cx, cy, cz = galactic_Rn(241.1490, 17.7216) # Hercules

    # make a figure
    if 1:

        # convert galactocentric to cartesian
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
        plt.savefig('fig1-{:}.pdf'.format(fig_name))
        plt.savefig('fig1-{:}.png'.format(fig_name))
        plt.show()

    # Hubble flow + pec vel.
    if 1:

        # convert galactocentric to cartesian
        R = np.linspace(0., 200., 256)
        R_vec = np.dstack((cx * R, cy * R, cz * R))

        Vr_R, d_R = Vr_intp(R_vec), d_intp(R_vec)
        Vx, Vy, Vz = Vx_intp(R_vec), Vy_intp(R_vec), Vz_intp(R_vec)
        Vr_c = Vx * cx + Vy * cy + Vz * cz

        fig = plt.figure(figsize=(8., 4.))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(R * 100. / H,           H * R, c='C0')
        ax1.plot(R * 100. / H, Vr_c[0] + H * R, c='C4')

        ax1.set_ylabel('$V_{\mathrm{CMB}}$')
        ax1.set_xlabel('Distance [Mpc] (H={:})'.format(H))
        plt.savefig('fig2-{:}.pdf'.format(fig_name))
        plt.savefig('fig2-{:}.png'.format(fig_name))
        plt.show()

    # search for triple value cases
    if 0:
        R = np.linspace(0., 200., 512)
        delta_R = np.median(np.diff(R))
        R_mp = (R[1:] + R[:-1]) / 2.
        Rc_mv, i_pt = np.empty((16384, 4)), 0
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
        b = np.arcsin(Rc_mv[:, 2]) * 180. / np.pi
        l = np.arctan2(Rc_mv[:, 1], Rc_mv[:, 0])  * 180. / np.pi
        sin_b = Rc_mv[:, 2]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1, aspect='auto')
        ax.scatter(l, sin_b, s=0.1, c=Rc_mv[:, 3],
                   vmin=0., vmax=200., cmap='jet',
                   alpha=0.35)
        # alpha=0.25 + 0.75 * np.clip(Rc_mv[:, 3] / 200., 0., 1.)
        y_ticks = [-90, -60, -30, 0, 30, 60, 90]
        ax.set_yticks(np.sin(np.array(y_ticks) * np.pi / 180.))
        ax.set_yticklabels([str(w) for w in y_ticks])
        ax.set_xlim(180., -180)
        ax.set_ylim(-1., 1.)
        ax.set_xlabel('l [deg]'), ax.set_ylabel('b [deg]')
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.95)
        plt.savefig('fig3.pdf')
        plt.show()

    if 0:

        # convert galactocentric to cartesian
        cx, cy, cz = galactic_Rn(313.43581749999896, 0.4198550000002115)
        print(_heliocentric_to_CMB(cx, cy, cz))