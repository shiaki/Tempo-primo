# - - - - - - - - - - - - - - -
if None:
    
    # From: zcorr.py. YJ, 191211

    # check: triple-valued regions?
    Vr = 1.e2 * self._hsq * self._R_ax + vrp_intp(self._R_ax) # v + pec motion
    d_Vr = np.gradient(Vr) # marks triple-value zones.

    # split into monotonic regions.
    if np.nansum(d_Vr < 0.) == 0: # monotonic
        Mrg = [(0, 512)] # just a single monotonic region.
    else: # having multiple monotonic regions:
        dVrz = np.r_[0, np.arange(511)[d_Vr[:-1] * d_Vr[1:] <= 0.], 512]
        Mrg = list(zip(dVrz[:-1], dVrz[1:]))

    # for each monotonic interval: find possible 'd_opt'
    dopt_vals = list()
    for i_zero, j_zero in Mrg:
        i_zero, j_zero = max(0, i_zero), min(511, j_zero + 1)
        if j_zero - i_zero < 2: # just a tiny spike
            continue
        Vr_sct = Vr[i_zero: j_zero + 1]
        print(i_zero, j_zero, Vr_sct, v_obs)
        Vr_max, Vr_min = Vr_sct.max(), Vr_sct.min()
        if (v_obs < Vr_min) or (v_obs > Vr_max): # beyond range: skip
            continue
        d_opt = brentq(rfc, self._R_ax[i_zero], self._R_ax[j_zero])
        dopt_vals.append(d_opt)

    # return a list of possible d.
    return np.array(dopt_vals)
