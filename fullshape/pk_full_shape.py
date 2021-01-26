''' This module calculates the anisotropic power spectrum

To do:
- add BAO damping
- calculate covariance matrix (diff module)
- add noise (diff module)
- improve integration (w convergence criterion)
'''


import numpy as np
from scipy.integrate import simps 
import camb

def get_growth_factor(results):
    '''returns the growth factor from a CAMB results object'''
    return results.get_fsigma8()/results.get_sigma8()[0]

def generate_power_spectrum(zs = [0.], minkh=5e-5, maxkh = 1, npoints = 200, 
                            As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                            omch2=0.1188, mnu=0.06, omk=0, tau=0.06):

    ''' 
    Given redshift, a min and max kh, a number of k bins and cosmological 
    parameters, returns a kh array and the matter power spectrum from CAMB. 
    It also returns the growth factor f and the redshifts (as they might be
    resorted).
    '''

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, 
                       tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)

    pars.set_matter_power(redshifts=zs, kmax=2.0)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    f = get_growth_factor(results)
    kh, zs, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, 
                                                  npoints = npoints)

    return kh, zs, pk, f

def get_kaiser_factor(mu,b1,f):
    ''' Calculate the Kaiser factor'''
    kaiser = (b1+np.outer(f,mu**2))**2.
    return kaiser[:,np.newaxis, :]

def get_fog_factor(kh, mu, sigma_v, f):
    ''' Calculate the Fingers of God factor'''
    temp = sigma_v*np.outer(kh, mu)
    logf = np.einsum('i, jk -> ijk', f, temp)
    return np.exp(-(logf)**2.)

def get_legendre_2(x):
    ''' Returns the second order Legendre polynomial'''
    return 0.5*(3.*x**2-1)

def get_legendre_4(x):
    ''' Returns the fourth order Legendre polynomial'''
    return (1/8.)*(35.*x**4-30*x**2.+3)

def get_anisotropic_pk(kh, pk, f, b1, sigma_v):
    ''' Returns anisotropic power spectra, with dimensions [redshift, k]'''
    mu_arr = np.linspace(0,1,1000)
    kaiser_arr = get_kaiser_factor(mu_arr, b1, f)
    fog_arr = get_fog_factor(kh, mu_arr, sigma_v, f)
    Pg0 = fog_arr*kaiser_arr*pk[:,:,np.newaxis]
    Pg2 = Pg0*get_legendre_2(mu_arr)
    Pg4 = Pg0*get_legendre_4(mu_arr)
    P_theory_0 = simps(y = Pg0, x = mu_arr, axis = -1)
    P_theory_2 = simps(y = Pg2, x = mu_arr, axis = -1)
    P_theory_4 = simps(y = Pg4, x = mu_arr, axis = -1)
 
    #P_theory_0 = np.trapz(Pg0, mu_arr, axis = -1)
    #P_theory_2 = np.trapz(Pg2, mu_arr, axis = -1)
    #P_theory_4 = np.trapz(Pg4, mu_arr, axis = -1)
    return P_theory_0, P_theory_2, P_theory_4