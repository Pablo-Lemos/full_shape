'''
This module calculates the anisotropic BAO damping for a given power spectrum

Based on Lado's Julia script.
'''

import numpy as np
import scipy.optimize as optimize

def get_smooth_pk(k, pars):
    keq, A, a0, a2, a4 = pars
    q = k/keq
    L = np.log(2*np.exp(1) + 1.8*q)
    C = 14.2 + 731.0/(1 + 62.5*q)
    T = L/(L + C*q**2)
    return A*(T**2 *k + a0 + a2*k**2 + a4*k**4)

def get_chisq_smooth_pk(k, pk, pars):
    smooth_p = get_smooth_pk(k, pars)
    chi2 = np.sum(k**2*(pk - smooth_p)**2)
    return chi2

def minimize_smooth_pk(k, pk):
    chi2 = lambda params: get_chisq_smooth_pk(k, pk, params) 
    initial_guess = [0.6, 60000, 0.1, 0., 0.]
    result = optimize.minimize(chi2, initial_guess)#, method='Nelder-Mead', tol=1e-8)
    best_pars = result.x
    smooth_pk = get_smooth_pk(k, best_pars)
    return smooth_pk