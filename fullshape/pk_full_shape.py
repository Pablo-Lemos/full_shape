# -*- coding: utf-8 -*-
''' 
This module performs full shape power spectrum calculations. 

Written by Pablo Lemos 
28-01-2020
'''

import numpy as np
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

    def __init__(self, zs = [0.], mink=1e-4, maxk = 1, num_k = 200, num_mu = 1000, hunits = False):
        """
        The initial function. Given a list of redshifts, a k range, and cosmological
        parameters, it does the following:
        
        - Calculate the growth factor f
        - Generate linearly separated arrays for kh and mu
        - Generate a power spectrum from CAMB and evaluate it at kh
        
        Arrays in this class are aranged to take the shape [z, k, mu]
        """

        self.zs = np.sort(zs)
        #if self.zs != zs:
        #    print('Redshifts have been sorted in increasing order')

        self.k = np.linspace(mink, maxk, num_k)        
        self.mu = np.linspace(0, 1, num_mu)
        self.hunits = hunits

    def set_cosmology(self, As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                            omch2=0.1188, mnu=0.06, omk=0, tau=0.06):
        ''' Generate a CAMB instance, and use it to calculate the growth 
        parameter f, and a power spectrum interpolator '''
        import camb

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, 
                        tau=tau)
        pars.InitPower.set_params(As=As, ns=ns, r=0)

        pars.set_matter_power(redshifts=self.zs, kmax=2.0)
        self.results = camb.get_results(pars)
        #print(results.get_fsigma8(),results.get_sigma8())
        self.f = self.results.get_fsigma8()/self.results.get_sigma8()[0]
        self.sigma8 = self.results.get_sigma8_0()

        if self.hunits:
            self.PK = camb.get_matter_power_interpolator(pars, nonlinear=False, 
                kmax=self.k[-1], zmax=max([0.,self.zs[-1]]), hubble_units = True, 
                k_hunit = True)
        else:
            self.PK = camb.get_matter_power_interpolator(pars, nonlinear=False, 
                kmax=self.k[-1], zmax=max([0.,self.zs[-1]]), hubble_units = False, 
                k_hunit = False)

    def growth_factor(self):
        ''' Calculate the growth factor D(a) '''
        #TODO If I ever add nonlinear corrections in Planck, make sure they are not used here!
        try:
            pk = self.PK.P(self.zs, self.k)
        except:
            print('You must generate a CAMB power spectrum interpolator first. ')
            print("Use 'set cosmology'")
            raise

        pk0 = self.PK.P(0, self.k)       
        D2 = np.mean(pk/pk0, axis = -1)
        return D2**0.5


    def kaiser_factor(self, bias):
        ''' Calculate the Kaiser factor'''
        if isinstance(bias, float) or isinstance(bias, int):
            # Using a single bias for all redshift bins
            kaiser = (bias+np.outer(self.f,self.mu**2))**2.
        else:
            # Using a bias parameter for each redshift
            kaiser = (bias[:,np.newaxis]+np.outer(self.f,self.mu**2))**2.
        return kaiser[:,np.newaxis, :]

    def fog_factor(self, sigma_v):
        ''' Calculate the Fingers of God factor'''
        temp = sigma_v*np.outer(self.k, self.mu)
        logfog = np.einsum('i, jk -> ijk', self.f, temp)
        return np.exp(-(logfog)**2.)

    def calculate_BAO_damping(self, sigma_perp, sigma_par):
        ''' Calculate the BAO damping factor'''
        #mu = np.reshape(self.mu, [1,1,-1])
        k = np.reshape(self.k, [1,-1,1])
        # Reshape mu^2*sigma_par^2 and (1-mu^2)*sigma_perp into shape [z,k,mu]
        spar_factor = np.outer(sigma_par**2, self.mu**2)[:,np.newaxis,:]
        sperp_factor = np.outer(sigma_perp, (1 - self.mu**2))[:,np.newaxis,:]
        logdamp = k**2/2*(spar_factor + sperp_factor)
        return np.exp(-logdamp)

    def add_BAO_damping(self, sigma_perp, sigma_par):
        ''' Add BAO damping to a CAMB power sectrum'''
        ps = np.empty([len(self.zs), len(self.k), 1])
        for (i,pk) in enumerate(self.pk_camb[:,:,0]):
            ps[i, :, 0] = minimize_smooth_pk(self.k, pk)
        pnl = self.pk_camb - ps
        bao_damp_factor = self.calculate_BAO_damping(sigma_perp, sigma_par)
        pnl = np.einsum('ijk, ijk -> ijk', pnl, bao_damp_factor)
        pnl *= bao_damp_factor
        return pnl + ps

    def get_anisotropic_pk(self, bias, sigma_v, bao_damping = True, integration = 'Simps'):
        ''' Returns anisotropic power spectra, with dimensions [redshift, k]'''

        # Generate a CAMB pk, and reshape it into [z,k,mu] shape
        try:
            pk = self.PK.P(self.zs, self.k)
            self.pk_camb = pk[:,:,np.newaxis]
        except:
            print('You must generate a CAMB power spectrum interpolator first. ')
            print("Use 'set cosmology'")
            raise

        kaiser = self.kaiser_factor(bias)
        fog = self.fog_factor(sigma_v)
        
        if bao_damping:
            ''' Formulas for sigma_perp and sigma_par from Lado'''
            D = self.growth_factor()
            sigma_perp = 9.4*self.sigma8/0.9*D
            sigma_par = (1 + self.f)*sigma_perp
            self.Pmu = self.add_BAO_damping(sigma_perp, sigma_par)
        else:
            self.Pmu = self.pk_camb
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

    def generate_noisy(self, nave, integration = 'Simps'):
        ''' Generate a noisy realization of the anisotropic power spectrum. 
        Returns a power spectrum and covariance. '''

        try: 
            self.Pmu
        except:
            print('You must generate an anisotropic power spectrum first')
            print("Use 'get_anisotropic_pk'")
            raise

        num_k = len(self.k)
        num_z = len(self.zs)
        num_mu = len(self.mu)        

        # Calculate k_min and k_max, the edges of the k bins
        dk = self.k[1] - self.k[0]
        k_edges = np.concatenate([[self.k[0] - dk/2.], self.k + dk/2.])
        k_max = k_edges[1:]
        k_min = k_edges[:-1]

        # Same for the redshift
        if num_z > 1:
            dk = self.z[1] - self.z[0]
            z_edges = np.concatenate([[self.z[0] - dz/2.], self.z + dz/2.])
            chi_edges = self.results.comoving_radial_distance(z_edges)
            vol = 4*np.pi/3*(chi_edges[1:]**3 - chi_edges[:-1]**3)
            z_max = z_edges[1:]
            z_min = z_edges[:-1]
        else:
            print('Because there is only one redshift, I am considering an infinite volume for the covariance calculation')
            vol = 4*np.pi/3*self.results.comoving_radial_distance(np.infty)

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

                    k_vol = 2*np.pi/3.*(k_max**3. - k_min**3)*vol[z]/(2*np.pi)**3.
                    cov[z, i*num_k:(i+1)*num_k,j*num_k:(j+1)*num_k] = res/k_vol*np.identity(num_k)

                    if i == j:
                        pk_noisy[i, z] += np.random.multivariate_normal(np.zeros(num_k), res/k_vol*np.identity(num_k))

        return pk_noisy, cov

    