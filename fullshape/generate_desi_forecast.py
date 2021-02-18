# -*- coding: utf-8 -*-
''' Generate a noisy DESI forecast, and store it '''

import numpy as np
import os
from fullshape.pk_full_shape import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def generate_pk_for_tracer(zs, name, bias_factor, sigma_v, path, noisy = False):
    ''' For a given tracer (LRG, ELG or QSO), generate pk and a covariance

    For the linear bias, we use a fiducial model b = bias_factor/D where
    D is the growth factor and bias factor is 0.84 for ELG, 1.7 for LRG and 
    1.2 for QSO. 
    '''
    # Initiate anisotropic pk calculator
    pk_calc = PK_Calculator(zs = zs, mink=1e-4, maxk = 0.1, num_k = 20, hunits = False)   

    # Set a cosmology
    pk_calc.set_cosmology(As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                                omch2=0.1188, mnu=0.06, omk=0, tau=0.06)    

    D = pk_calc.growth_factor()
    bias = bias_factor/D

    # Calculate anisotropic pk
    pk_calc.get_anisotropic_pk(bias, sigma_v,  bao_damping = True)

    # Generate a noisy realisation
    pk_noisy, cov = pk_calc.generate_noisy(nave = 1e-4)

    np.save(path + 'k_bins', pk_calc.k)
    np.save(path + 'cov_' + name, cov)

    if noisy: 
        np.save(path + 'pk_' + name, pk_noisy)
    else:
        pk_noiseless = np.stack([pk_calc.p0, pk_calc.p2, pk_calc.p4])
        np.save(path + 'pk_' + name, pk_noiseless)
    

if __name__ == '__main__':
    # Set up redshifts
    z_lrg = np.arange(0.05, 1, 0.1)
    z_elg = np.arange(1.05, 1.6, 0.1)
    z_qso = np.arange(1.65, 2.1, 0.1)

    path = '/Users/Pablo/Code/full_shape_external/fullshape/simulated_data/desi_forecast/'

    generate_pk_for_tracer(z_lrg, name = 'lrg', bias_factor= 0.84, sigma_v = 5., 
                            path=path, noisy = False)
    generate_pk_for_tracer(z_elg, name = 'elg', bias_factor= 1.7, sigma_v = 4., 
                            path=path, noisy = False)
    generate_pk_for_tracer(z_qso, name = 'qso', bias_factor= 1.2, sigma_v = 3., 
                            path=path, noisy = False)