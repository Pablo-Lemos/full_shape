# -*- coding: utf-8 -*-
''' Generate a noisy power spectrum, and store it '''

import numpy as np
import os
from pk_full_shape import *

if __name__ == '__main__':
    # Define value of nuisance parameters
    b1 = 2.
    sigma_v = 0.
    sigma_per = 10.
    sigma_par = 10.

    # Initiate anisotropic pk calculator
    pk_calc = PK_Calculator(minkh=1e-5, maxkh = 0.5, num_k = 50)

    # Calculate anisotropic pk
    pk_calc.get_anisotropic_pk(sigma_per, sigma_par, b1, sigma_v)

    # Generate a noisy realisation
    pk_noisy, cov = pk_calc.generate_noisy(nave = 1e-4, vol = 1e9)

    cwd = os.path.realpath(__file__)[:-25]
    np.save(cwd + '/simulated_data/p0_noisy', pk_noisy)
    np.save(cwd + './simulated_data/cov', cov)
