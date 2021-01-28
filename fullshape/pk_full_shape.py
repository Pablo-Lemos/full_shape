# -*- coding: utf-8 -*-
''' 
This module performs full shape power spectrum calculations. 

Written by Pablo Lemos 
28-01-2020
'''

import numpy as np
import camb
from scipy.integrate import simps 
from fullshape.smooth_pk import minimize_smooth_pk

def get_legendre_2(x):
    ''' Returns the second order Legendre polynomial'''
    return 0.5*(3.*x**2-1)

def get_legendre_4(x):
    ''' Returns the fourth order Legendre polynomial'''
    return (1/8.)*(35.*x**4-30*x**2.+3)

class PK_Calculator:
    """
    A class to perform full shape calculations and generate noisy realizations
    """

    def __init__(self, zs = [0.], minkh=1e-4, maxkh = 1, num_k = 200, num_mu = 1000, 
                            As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                            omch2=0.1188, mnu=0.06, omk=0, tau=0.06):
        """
        The initial function. Given a list of redshifts, a k range, and cosmological
        parameters, it does the following:
        
        - Calculate the growth factor f
        - Generate linearly separated arrays for kh and mu
        - Generate a power spectrum from CAMB and evaluate it at kh
        
        Arrays in this class are aranged to take the shape [z, k, mu]
        """

        self.zs = np.sort(zs)
        if self.zs != zs:
            print('Redshifts have been sorted in increasing order')

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, 
                        tau=tau)
        pars.InitPower.set_params(As=As, ns=ns, r=0)

        pars.set_matter_power(redshifts=self.zs, kmax=2.0)
        #pars.NonLinear = camb.model.NonLinear_none
        self.results = camb.get_results(pars)
        self.f = self.results.get_fsigma8()/self.results.get_sigma8()[0]
        #self.kh, self.zs, pk = self.results.get_matter_power_spectrum(
        #                     minkh=minkh, maxkh=maxkh, npoints = num_k)

        PK = camb.get_matter_power_interpolator(pars, nonlinear=False, 
            kmax=maxkh, zmax=max([0.01,self.zs[-1]]))

        self.kh = np.linspace(minkh, maxkh, num_k)        
        self.mu = np.linspace(0, 1, num_mu)
        pk = PK.P(self.zs, self.kh)

        # Reshape pk into [z,k,mu] shape
        self.pk_camb = pk[:,:,np.newaxis]


    def kaiser_factor(self, b1):
        ''' Calculate the Kaiser factor'''
        kaiser = (b1+np.outer(self.f,self.mu**2))**2.
        return kaiser[:,np.newaxis, :]

    def fog_factor(self, sigma_v):
        ''' Calculate the Fingers of God factor'''
        temp = sigma_v*np.outer(self.kh, self.mu)
        logfog = np.einsum('i, jk -> ijk', self.f, temp)
        return np.exp(-(logfog)**2.)

    def calculate_BAO_damping(self, sigma_per, sigma_par, b1):
        ''' Calculate the BAO damping factor'''
        mu = np.reshape(self.mu, [1,1,-1])
        k = np.reshape(self.kh, [1,-1,1])
        logdamp = k**2/2*(mu**2*sigma_par**2 + (1 - mu**2)*sigma_per**2)
        return np.exp(-logdamp)

    def add_BAO_damping(self, sigma_per, sigma_par, b1):
        ''' Add BAO damping to a CAMB power sectrum'''
        ps = np.empty([len(self.zs), len(self.kh), 1])
        for (i,pk) in enumerate(self.pk_camb[:,:,0]):
            ps[i, :, 0] = minimize_smooth_pk(self.kh, pk)
        pnl = self.pk_camb - ps
        bao_damp_factor = self.calculate_BAO_damping(sigma_per, sigma_par, b1)
        pnl = np.einsum('ijk, ijk -> ijk', pnl, bao_damp_factor)
        pnl *= bao_damp_factor
        return pnl + ps

    def get_anisotropic_pk(self, sigma_per, sigma_par, b1, sigma_v, integration = 'Simps'):
        ''' Returns anisotropic power spectra, with dimensions [redshift, k]'''

        kaiser = self.kaiser_factor(b1)
        fog = self.fog_factor(sigma_v)
        
        self.Pmu = self.add_BAO_damping(sigma_per, sigma_par, b1)
        self.Pmu = fog*kaiser*self.Pmu
        Pmu0 = self.Pmu
        Pmu2 = self.Pmu*get_legendre_2(self.mu)
        Pmu4 = self.Pmu*get_legendre_4(self.mu)

        if integration == 'Simps':
            self.p0 = simps(y = Pmu0, x = self.mu, axis = -1)
            self.p2 = simps(y = Pmu2, x = self.mu, axis = -1)
            self.p4 = simps(y = Pmu4, x = self.mu, axis = -1)
        
        elif integration == 'Trapz':
            self.p0 = np.trapz(y = Pmu0, x = self.mu, axis = -1)
            self.p2 = np.trapz(y = Pmu2, x = self.mu, axis = -1)
            self.p4 = np.trapz(y = Pmu4, x = self.mu, axis = -1)

        else:
            print("Integration must be either 'Simps' or 'Trapz' ")
            raise

    def generate_noisy(self, nave, vol, integration = 'Simps'):
        ''' Generate a noisy realization of the anisotropic power spectrum. 
        Returns a power spectrum and covariance. '''

        if self.Pmu is None:
            print('You must generate an anisotropic power spectrum first')
            print("Use 'get_anisotropic_pk'")

        num_k = len(self.kh)
        num_z = len(self.zs)
        num_mu = len(self.mu)        

        # Calculate k_min and k_max, the edges of the k bins
        dk = self.kh[1] - self.kh[0]
        k_edges = np.concatenate([[self.kh[0] - dk/2.], self.kh + dk/2.])
        k_max = k_edges[1:]
        k_min = k_edges[:-1]

        l0 = self.mu
        l2 = get_legendre_2(l0)
        l4 = get_legendre_4(l0)
        leg = np.stack([l0, l2, l4])

        pk_noisy = np.stack([self.p0, self.p2, self.p4])
        cov = np.empty([num_z, 3*num_k, 3*num_k])

        for z in range(num_z):
            for i in range(3):
                for j in range(3):
                    f = (self.Pmu[z] + 1/nave)**2*leg[i]*leg[j]

                    if integration == 'Simps':
                        res = simps(y = f, x = self.mu, axis = -1)
                    elif integration == 'Trapz':
                        res = np.trapz(y = f, x = self.mu, axis = -1)
                    else:
                        print("Integration must be either 'Simps' or 'Trapz' ")
                        raise

                    k_vol = 2*np.pi/3.*(k_max**3. - k_min**3)*vol/(2*np.pi)**3.
                    cov[z, i*num_k:(i+1)*num_k,j*num_k:(j+1)*num_k] = res/k_vol*np.identity(num_k)

                    if i == j:
                        pk_noisy[i, z] += np.random.multivariate_normal(np.zeros(num_k), res/k_vol*np.identity(num_k))

        return pk_noisy, cov

    