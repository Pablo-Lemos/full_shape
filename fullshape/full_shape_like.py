import numpy as np
from cobaya.likelihood import Likelihood
from fullshape.pk_full_shape import *

class FullShapeLikelihood(Likelihood):

    def initialize(self):

        self.zs = self.zs
        self.k_bins = np.load(self.k_bins_file)
        p0, p2, p4 = np.load(self.pk_data_file)
        pk_cov = np.load(self.pk_cov_file)
        self.invcov = np.linalg.inv(pk_cov)

        # Create a datavector with dimensions [z, k], where the k has 
        self.pk_data = np.concatenate([p0, p2, p4], axis = -1)

        self.pk_calc = PK_Calculator(zs = self.zs, mink=self.k_bins[0], 
                            maxk = self.k_bins[-1], num_k = len(self.k_bins), hunits = False)
    
    def get_requirements(self):

        # Create array of redshifts for interpolation
        zmax = max(0.1, max(self.zs))
        zarr = np.linspace(0, zmax, 50)

        return {
            "Pk_interpolator": {
                "z": zarr, "k_max": self.k_bins[-1], "nonlinear": False,
                "vars_pairs": ([("delta_tot", "delta_tot")])},
                "fsigma8": {"z": self.zs},
                "sigma8z": {"z": self.zs},
                #'H0': None
               #"sigma_R": {"z": self.zs, "R": 8},
        }

    def logp(self, **params_values):
        #H0_theory = self.provider.get_param("H0")
        #h_theory = H0_theory/100.
        #print(self.provider.get_fsigma8(self.zs), self.provider.get_sigma8z(self.zs))
        self.pk_calc.f = self.provider.get_fsigma8(self.zs)/self.provider.get_sigma8z(self.zs)
        self.pk_calc.PK = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear = False)      
        b1 = params_values['b1']
        sigma_v = params_values['sigma_v']

        #print(self.provider.get_fsigma8(self.zs), self.provider.get_sigma_R(), self.provider.get_sigma8z(self.zs))
        self.pk_calc.get_anisotropic_pk(b1, sigma_v, bao_damping = True)
        pk_theory = np.concatenate(
            [self.pk_calc.p0, self.pk_calc.p2, self.pk_calc.p4], axis = -1)
        
        #pk_theory*=h_theory**3.

        chi2 = 0.0
        for z in range(len(self.zs)):
            #print(self.pk_data)
            #print(pk_theory)
            delta = self.pk_data[z] - pk_theory[z]
            chi2 += self.invcov[z].dot(delta).dot(delta)

        return -0.5*chi2     

