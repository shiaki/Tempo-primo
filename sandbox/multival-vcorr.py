#!/usr/bin/python

'''
    Test radial velocoty correction for multi-valued cases.
'''

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, newton
import matplotlib.pyplot as plt

if __name__ == '__main__':

    h = 0.68
    c = 2.99792458e5
    R = np.linspace(0., 200., 512) # Mpc/h

    # make a simple pec vel model.
    Rc, s = 100., 10. # Mpc/h
    Gc = -np.exp(-0.5 * (R - Rc) ** 2 / (s ** 2)) * (R - Rc) / (s ** 2)
    Vp = Gc * 2.5e4 # peculiar velocity, km/s
    Vr = 1.e2 * R * (h ** 2) + Vp # observed velocity, km/s
    Vp_intp = interp1d(R, Vp, fill_value='extrapolate')

    # root finding
    rfc = lambda D, Vt: Vt - (1.e2 * (h ** 2)) * D - Vp_intp(D)

    rec_pos = list()
    for v_i in np.random.choice(Vr, size=25):
        d_i = newton(rfc, v_i / (100. * (h ** 2)), args=(v_i,))
        rec_pos.append((d_i, v_i))
    rec_pos = np.array(rec_pos)

    plt.plot(R, Vr, label='Hubble flow + Peculiar motion')
    plt.scatter(rec_pos[:, 0], rec_pos[:, 1], s=18, c='C1', label='Reconstructed')
    plt.xlabel('Distance [Mpc/h]')
    plt.ylabel('Observed velocity')
    plt.legend()
    plt.savefig('fig3.pdf')
    plt.show()
