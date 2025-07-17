import numpy as np
import camb
#import para as Cp
import matplotlib.pyplot as plt
import glass.ext.camb


"""redshift window"""

def trapezoid(start, stop, upper_wide_f):
    lower_start = start
    lower_stop = stop
    middle_point = (lower_stop + lower_start) / 2
    lower_width = lower_stop - lower_start
    upper_width = lower_width * upper_wide_f
    upper_start = middle_point - upper_width / 2
    upper_stop = middle_point + upper_width / 2

    def f(z):
        if z <= lower_start:
            return 0
        elif (z > lower_start) & (z < upper_start):
            f1 = 1 / (upper_start - lower_start)
            return (z - lower_start) * f1 + 0
        elif (z > upper_start) & (z < upper_stop):
            return 1
        elif (z > upper_stop) & (z < lower_stop):
            f2 = -1 / (lower_stop - upper_stop)
            return (z - upper_stop) * f2 + 1
        elif z >= lower_stop:
            return 0

    return f


def Compare_Cls(Cl1, Cl2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), gridspec_kw={'height_ratios': [3, 1]})
    ax1.loglog(Cl1, label = r"Cl$_{1}$")
    ax1.loglog(Cl2, label = r"Cl$_{2}$")
    ax1.legend(fontsize=9, loc='lower left')
    ax1.tick_params(labelbottom=False)

    # Lower subplot (shorter)
    ax2.plot((Cl1 - Cl2) / Cl1, color='red', label = r"1 - Cl$_{2}$/Cl$_{1}$")
    ax2.set_ylim([-0.05, 0.05])
    ax2.axhline(y=0, linestyle='--', color='k')
    ax2.legend(fontsize=9, loc='lower left')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=1, hspace=0.0)
    plt.show()



class Shell_Cls:
    def __init__(self, cosmology_para, zbin_in, upper_f, plot=False):
        self.H0 = cosmology_para['H0']
        self.ombh2 = cosmology_para['ombh2']
        self.omch2 = cosmology_para['omch2']
        self.ns = cosmology_para['ns']
        self.s8 = cosmology_para['s8']

        As = 2e-9  # fiducial amplitude guess to start with
        pars = camb.set_params(H0=self.H0, ombh2= self.ombh2, omch2=self.omch2, ns=self.ns, As=As)
        pars.set_matter_power(redshifts=[0.], kmax=2.0)
        results = camb.get_results(pars)
        s8_fid = results.get_sigma8_0()
        # now set correct As using As \propto sigma8**2.
        pars.InitPower.set_params(As=As * self.s8 ** 2 / s8_fid ** 2, ns=self.ns)

        self.pars = pars
        self.zbin_in = zbin_in
        self.upper_f = upper_f
        shell_z = []
        shell_nz = []
        for bin_i in range(len(self.zbin_in) - 1):
            start_i = self.zbin_in[bin_i]
            stop_i = self.zbin_in[bin_i + 1]
            z_i_line = np.linspace(start_i, stop_i, 10000)
            z_i_f = trapezoid(start_i, stop_i, self.upper_f)
            nz_f = [z_i_f(zi) for zi in z_i_line]
            shell_z.append(z_i_line)
            shell_nz.append(nz_f)
        self.shell_z = shell_z
        self.shell_nz = shell_nz
        self.shells = [[self.shell_z[i],self.shell_nz[i],0] for i in range(len(self.shell_z))]
        if plot:
            for shell in self.shells:
                plt.plot(shell[0], shell[1])
                plt.xlabel("redshift z")
                plt.ylabel("n(z)")


    def cls_shell_theory(self, lmax):
        #print("calculating shell cls")
        return glass.ext.camb.matter_cls(self.pars, lmax, self.shells)

    def cls_total_theory_withwin(self, lmax, Ncount, plot=False):
        z_total = np.linspace(min(self.zbin_in), max(self.zbin_in), 10000)
        nz_total = np.zeros_like(z_total)
        Ncount_hist_normalized = Ncount / np.sum(Ncount)
        for bin_i in range(len(self.zbin_in) - 1):
            start_i = self.zbin_in[bin_i]
            stop_i = self.zbin_in[bin_i + 1]
            z_i_f = trapezoid(start_i, stop_i, self.upper_f)
            nz_total += np.array([z_i_f(zi) for zi in z_total]) * Ncount_hist_normalized[bin_i]

        if plot:
            plt.plot(z_total, nz_total)
            plt.xlabel("redshift z")
            plt.ylabel("n(z)")
        return glass.ext.camb.matter_cls(self.pars, lmax, [[z_total, nz_total, 0]])[0]

    def cls_total_theory_nowin(self, lmax, Ncount, plot=False):
        z_total = (self.zbin_in[:-1] + self.zbin_in[1:])/2
        Ncount_hist_normalized = Ncount / np.sum(Ncount)
        if plot:
            plt.plot(z_total, Ncount_hist_normalized)
            plt.xlabel("redshift z")
            plt.ylabel("n(z)")
        return glass.ext.camb.matter_cls(self.pars, lmax, [[z_total, Ncount_hist_normalized, 0]])[0]


    def cls_total_linear_combination(self, cls_shell, label, Ncount):
        cls_total_flat_FromShellTheory_cross = 0
        for j in range(len(label)):
            if label[j][0] == label[j][1]:
                cls_total_flat_FromShellTheory_cross += (
                            cls_shell[j] * Ncount[label[j][0] - 1] *
                            Ncount[label[j][1] - 1])
            if label[j][0] != label[j][1]:
                cls_total_flat_FromShellTheory_cross += 2 * (
                            cls_shell[j] * Ncount[label[j][0] - 1] *
                            Ncount[label[j][1] - 1])
        cls_total_flat_FromShellTheory_cross = cls_total_flat_FromShellTheory_cross / (
                    np.sum(Ncount) ** 2)
        return cls_total_flat_FromShellTheory_cross



def cls_total_direct(cls_shell, label, Ncount):
    cls_total_flat_FromShellTheory_cross = 0
    for j in range(len(label)):
        if label[j][0] == label[j][1]:
            cls_total_flat_FromShellTheory_cross += (
                            cls_shell[j] * Ncount[label[j][0] - 1] *
                            Ncount[label[j][1] - 1])
        if label[j][0] != label[j][1]:
            cls_total_flat_FromShellTheory_cross += 2 * (
                            cls_shell[j] * Ncount[label[j][0] - 1] *
                            Ncount[label[j][1] - 1])
    cls_total_flat_FromShellTheory_cross = cls_total_flat_FromShellTheory_cross / (
                np.sum(Ncount) ** 2)
    return cls_total_flat_FromShellTheory_cross


def cls_covariance(Cl_list, Cl_label_list, i, j, m, n):
    """ For coveriance between auto or cross-correlations
    eg: Cl_list = [Cl_11, Cl_22, Cl_21,...]
        Cl_label_list = [[1, 1], [2, 2], [2, 1], ,....]
        i,j,m,n = 1, 2, 3, ...
    l = np.array(range(1024))

    https://arxiv.org/pdf/2302.04507 equation 53
    """
    dim = len(Cl_list[0])

    title_im = [i, m]
    title_im.sort(reverse=True)
    Cl1_im = Cl_list[Cl_label_list.index(title_im)]

    title_jn = [j, n]
    title_jn.sort(reverse=True)
    Cl2_jn = Cl_list[Cl_label_list.index(title_jn)]

    title_in = [i, n]
    title_in.sort(reverse=True)
    Cl1_in = Cl_list[Cl_label_list.index(title_in)]

    title_jm = [j, m]
    title_jm.sort(reverse=True)
    Cl2_jm = Cl_list[Cl_label_list.index(title_jm)]

    fsky = 1

    result_matrix = np.zeros([dim, dim])
    for l1 in range(dim):
        for l2 in range(dim):
            f1 = Cl1_im[l1]
            f2 = Cl2_jn[l2]
            f3 = Cl1_in[l1]
            f4 = Cl2_jm[l2]
            delta = int(l1==l2)
            d_l = 1
            result = ((f1 * f2 + f3 * f4) * delta) / ((2 * l1 + 1) * fsky * d_l)
            result_matrix[l1][l2] = result
    return result_matrix

def poisson_covariance(ell, C_ell, total_count, mask):
    """nbar = total_count/(4pi*mask)
    fsky = np.mean(mask)
    C_ell cl without p noise
    """
    fsky = np.mean(mask)
    nbar = total_count/(4*np.pi*np.mean(mask))
    N_ell = 1.0 / nbar
    prefac = 2 / ((2 * ell + 1) )
    cov_noise = prefac * (N_ell**2 + 2 * C_ell * N_ell)
    return cov_noise