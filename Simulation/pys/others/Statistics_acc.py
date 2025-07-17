import healpy as hp
import numpy as np
from functools import cached_property


class sim_data:
    def __init__(self, hist, mask, print_ = False):
        assert np.shape(hist)[1] == len(mask)
        self.data = hist
        self.mask = mask
        if print_ == True:
            print("{} objects sampled!".format(np.sum(hist, axis=(0,1,2))))

    @cached_property
    def photoz_shell_Nmaps(self):
        return np.sum(self.data, axis = 0).T

    @cached_property
    def photoz_shell_Dmaps(self):
        Nmaps = self.photoz_shell_Nmaps
        masked = self.mask
        means = np.mean(Nmaps[:, masked], axis=1)  # vectorized over shells
        densities = (Nmaps.T / means).T - 1  # shape (n_shells, n_pix)
        # Mask the map
        densities[:, ~masked] = hp.UNSEEN
        return densities

    def photoz_shell_nzs(self):
        return np.sum(self.data, axis = 1).T


    def cl_photoz_shell(self, index, lmax, de_Poisson=True):
        Nmaps = self.photoz_shell_Nmaps
        Dmaps = self.photoz_shell_Dmaps
        mean_mask = np.mean(self.mask)
        Cl_list = []

        for i in index:
            Nmap = Nmaps[i]
            Dmap = Dmaps[i]
            p = 0
            if de_Poisson:
                p = 4 * np.pi * mean_mask / np.sum(Nmap)
            Cl = hp.anafast(Dmap, lmax=lmax) / mean_mask - p
            Cl_list.append(Cl)
        return Cl_list

