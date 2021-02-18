# -*- coding: utf-8 -*-
''' Generate a noisy power spectrum, and store it '''

import numpy as np
import os
from fullshape.pk_full_shape import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Define value of nuisance parameters
    b1 = [1., 2.]
    sigma_v = [5., 4.]

    # Initiate anisotropic pk calculator
    pk_calc = PK_Calculator(zs = [0.1, 0.5], mink=1e-4, maxk = 0.1, num_k = 20, hunits = False)

    # Set a cosmology
    pk_calc.set_cosmology(As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                                omch2=0.1188, mnu=0.06, omk=0, tau=0.06)

    # Calculate anisotropic pk
    pk_calc.get_anisotropic_pk(b1, sigma_v,  bao_damping = False)

    # Generate a noisy realisation
    pk_noisy, cov = pk_calc.generate_noisy(nave = 1e-4)

    cwd = os.path.realpath(__file__)[:-25]
    np.save(cwd + '/simulated_data/k_bins', pk_calc.k)
    np.save(cwd + '/simulated_data/pk_noisy', pk_noisy)
    np.save(cwd + './simulated_data/cov', cov)
