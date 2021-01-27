import numpy as np
import camb
from scipy.integrate import simps 
import sys
from smooth_pk import minimize_smooth_pk

def get_legendre_2(x):
    ''' Returns the second order Legendre polynomial'''
    return 0.5*(3.*x**2-1)

def get_legendre_4(x):
    ''' Returns the fourth order Legendre polynomial'''
    return (1/8.)*(35.*x**4-30*x**2.+3)

class PK_Calculator:
    def __init__(self, zs = [0.], minkh=5e-5, maxkh = 1, num_k = 200, num_mu = 1000, 
                            As=2.142e-9, ns=0.9667, H0=67.36, ombh2=0.02230, 
                            omch2=0.1188, mnu=0.06, omk=0, tau=0.06):
        ''' The initial function generates the camb pk
        
        The arrays have dimensions [z, k, mu]
        '''

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, 
                        tau=tau)
        pars.InitPower.set_params(As=As, ns=ns, r=0)

        pars.set_matter_power(redshifts=zs, kmax=2.0)
        pars.NonLinear = camb.model.NonLinear_none
        self.results = camb.get_results(pars)
        self.f = self.results.get_fsigma8()/self.results.get_sigma8()[0]
        self.kh, self.zs, pk = self.results.get_matter_power_spectrum(
                             minkh=minkh, maxkh=maxkh, npoints = num_k)
        
        self.mu = np.linspace(0, 1, num_mu)
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
        mu = np.reshape(self.mu, [1,1,-1])
        k = np.reshape(self.kh, [1,-1,1])
        logdamp = k**2/2*(mu**2*sigma_par**2 + (1 - mu**2)*sigma_per**2)
        return np.exp(-logdamp)

    def add_BAO_damping(self, sigma_per, sigma_par, b1):
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
        
        Pmu0 = self.add_BAO_damping(sigma_per, sigma_par, b1)
        Pmu0 = fog*kaiser*Pmu0
        Pmu2 = Pmu0*get_legendre_2(self.mu)
        Pmu4 = Pmu0*get_legendre_4(self.mu)

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

        