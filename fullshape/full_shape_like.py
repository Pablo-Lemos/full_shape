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
        # a concatenation of p0, p2, p4
        self.pk_data = np.concatenate([p0, p2, p4], axis = -1)

        

        self.pk_calc = PK_Calculator(zs = self.zs, minkh=self.k_bins[0], 
                            maxkh = self.k_bins[-1], num_k = len(self.k_bins))
    
    def get_requirements(self):

        return {"Pk_interpolator": {
                "z": self.zs, "k_max": self.k_bins[-1], "nonlinear": False},
                "fsigma8": {"z": self.zs},
                "sigma_R": {"z": self.zs, "R": 8}
        }

    def logp(self, **params_values):
        self.pk_calc.PK = self.provider.get_PK_interpolator()
        self.pk_calc.f = self.provider.get_fsigma8()/self.provider.get_sigma8()[0]

        sigma_per = params_values['sigma_per']
        sigma_par = params_values['sigma_par']
        b1 = params_values['b1']
        sigma_v = params_values['sigma_v']

        self.pk_calc.get_anisotropic_pk(sigma_per, sigma_par, b1, sigma_v)

        pk_theory = np.concatenate(
            [self.pk_calc.p0, self.pk_calc.p2, self.pk_calc.p4], axis = -1)

        chisq = 0.0
        for z in range(len(self.zs)):
            delta = self.pk_data[z] - pk_theory[z]
            chi2 += self.invcov[z].dot(delta).dot(delta)

        return -0.5*chisq        

