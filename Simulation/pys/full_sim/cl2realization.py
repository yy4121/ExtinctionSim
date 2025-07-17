from itertools import accumulate
import numpy as np
import healpy as hp
import glass.fields
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import tqdm
cool_cmap = cm.cool
cool_cmap.set_under("w") # sets background to white
def triangle(num):
    return list(accumulate(range(1, num + 1)))[-1]


class GenerateShell:
    def __init__(self, num_shell, cls_jax, lmax, nside, recorder = False):
        """
        :param num_shell:
        :param cls_jax:
        :param lmax:
        :param nside:the nside if healpix
        :param density_distribution: density of each shell

        Generate a GenerateShell object used for following
        """
        self.num_shell = num_shell
        self.cls_jax = np.array(cls_jax)
        self.lmax = lmax
        self.nside = nside
        self.pixel_num = hp.nside2npix(self.nside)
        self.recorder = recorder

        self.cls_label_jax = []
        self.cls_label_glass = []
        self.cls_glass = []


        assert triangle(self.num_shell) == np.shape(self.cls_jax)[
            0], "the number of shell does not match with the number if cls"
        assert np.shape(self.cls_jax)[1] == self.lmax+1, "lmax not match with cls"
        assert self.nside in [2 ** p for p in range(11)], "number of nisde error or too large"

        self.reorder_jaxcosmo2glass()
        self.gls_tag = False
        self.gls = None

    def realization_density(self, lognormal = True):
        """generate healpix log-normal density maps for number of shells"""
        if not self.gls_tag:
            #print("generating gls for number of shells")
            self.gls = self.cls_glass
            if lognormal == True:
                self.gls = glass.fields.lognormal_gls(self.cls_glass)
            self.gls_tag = True
        else:
            pass
        #print("generating log-normal density maps for number of shells")
        if lognormal == True:
            matter = glass.fields.generate_lognormal(self.gls, nside=self.nside, ncorr=20)
        else:
            matter = glass.fields.generate_gaussian(self.gls, nside=self.nside, ncorr=20)
        realization_shells = []
        for i, delta_i in enumerate(matter):
            realization_shells.append(delta_i)
        assert len(realization_shells) == self.num_shell, "realization number not correct"
        return np.array(realization_shells)

    # def calculate_gls(self):
    #     if not self.gls_tag:
    #         self.gls = glass.fields.lognormal_gls(self.cls_glass, nside=self.nside, lmax=self.lmax, ncorr=20)
    #         self.gls_tag = True
    #     else:
    #         pass
    #
    def calculate_gls(self, lognormal = True):
        if lognormal == False:
            return None
        else:
            return glass.fields.lognormal_gls(self.cls_glass)

    def reorder_jaxcosmo2glass(self):
        """
        reorder cls from jax_cosmo to glass
        https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/angular_cl.html?highlight=_get_cl_ordering(tracer_jax_n)#
        https://github.com/glass-dev/glass.ext.camb/blob/main/glass/ext/camb.py
        """
        cl_label_jax = []
        for i in range(self.num_shell):
            for j in range(i, self.num_shell):
                cl_label_jax.append([j+1, i+1])
        self.cls_label_jax = cl_label_jax


        if self.recorder:
            self.cls_label_glass = [[i,j] for i in range(1, self.num_shell + 1) for j in range(i, 0, -1)]
            self.cls_glass = np.array([self.cls_jax[self.cls_label_jax.index(i)] for i in self.cls_label_glass])
        else:
            self.cls_label_jax = [[i, j] for i in range(1, self.num_shell + 1) for j in range(i, 0, -1)]
            self.cls_label_glass = self.cls_label_jax
            self.cls_glass = self.cls_jax

        assert len(self.cls_glass) == len(self.cls_jax)


    def auto_index_jax(self):
        """return the index of auto correlation in jax order cls"""
        auto_index = []
        for i in range(len(self.cls_label_jax)):
            if self.cls_label_jax[i][0] == self.cls_label_jax[i][1]:
                auto_index.append(i)
        return auto_index

    def auto_index_glass(self):
        """return the index of auto correlation in glass order cls"""
        auto_index = []
        for i in range(len(self.cls_label_glass)):
            if self.cls_label_glass[i][0] == self.cls_label_glass[i][1]:
                auto_index.append(i)
        return auto_index


    def auto_cls_jax(self):
        """return auto cls from glass"""
        index = self.auto_index_jax()
        return np.array([self.cls_jax[i] for i in index])


    def auto_cls_glass(self):
        """return auto cls from jax"""
        index = self.auto_index_glass()
        return np.array([self.cls_glass[i] for i in index])


class Realization_i:
    def __init__(self, hp_density_list, GenerateShell_object):
        """operation on certain density map.
        Input density maps(hp_density_list, with randomness), and generate related plot and cls"""
        assert type(GenerateShell_object) == GenerateShell, "type(GenerateShell_object) not correct"
        self.hp_density_list = hp_density_list
        self.cls_glass_list = GenerateShell_object.cls_glass
        self.cls_label_glass = GenerateShell_object.cls_label_glass
        self.source = GenerateShell_object

        self.nside = hp.pixelfunc.npix2nside(np.shape(hp_density_list)[1])
        self.num_shell = np.shape(hp_density_list[0])
        self.pix_win_f = hp.pixwin(self.nside)

        assert len(self.cls_glass_list) == triangle(len(hp_density_list)), "!!!number of hps and cls not match"

    def plot_healpix(self, list_of_index):
        """
        :param list_of_index: the index of REAL redshift bin STARTED with 1
        """
        fontsize = 50
        list_of_index = np.array(list_of_index)-1
        list_of_index = list(list_of_index)
        list_of_index.sort()
        list_of_hp = self.hp_density_list

        if type(list_of_index)==int:
            list_of_index = [list_of_index]
        else:
            pass
        num = len(list_of_index)
        fig, axe = plt.subplots(num,1)
        fig.set_size_inches(36, 30)
        map_to_plot = [list_of_hp[i] for i in list_of_index]
        if num == 1:
            plt.axes(axe)
            hp.mollview(map_to_plot[0], title=r"Density map $\delta_{}$".format(list_of_index[0]), hold=True, cmap='viridis')
        else:
            for j in range(len(list_of_index)):
                plt.axes(axe[j])
                hp.mollview((map_to_plot[j]/np.mean(map_to_plot[j]))-1, r"Density map $\delta_{}$".format(list_of_index[j]), hold= True,cmap=cool_cmap)
        #matplotlib.rcParams.update({'font.size': fontsize})
        matplotlib.pyplot.show()
        #matplotlib.rcParams.update({'font.size': 10})

    def calculate_cls_all(self, list_of_index, l_max, plot = True, Poisson = False, ref_N = 500):
        """
        do not use this, not completed
        calculate auto and cross correlation for the list of index,
        :param list_of_index: the index of REAL redshift bin STARTED with 1
        :return:
        """
        list_of_index = list(list_of_index)
        list_of_index.sort()
        num = len(list_of_index)
        cls_index_sublist = [[list_of_index[i], list_of_index[j]] for i in range(0, num)
                        for j in range(i, -1, -1)]

        if Poisson == False:
            plot_list = self.hp_density_list
        else:
            Poisson_map, Poisson_density = self.add_Poisson(ref_N)
            plot_list = Poisson_density


        if plot==True:
            fig, axs = plt.subplots(num, num, figsize=(4*len(list_of_index), 2.5*len(list_of_index)))
            fig.tight_layout()

            for i in range(num):
                map1_healpix = plot_list[list_of_index[i]-1]
                map1_index = list_of_index[i]
                for j in range(i, -1, -1):
                    map2_healpix = plot_list[list_of_index[j]-1]
                    map2_index = list_of_index[j]
                    plot_name = 'W{0}xW{1}'.format(map1_index, map2_index)

                    density1 = map1_healpix
                    density2 = map2_healpix
                    result = hp.sphtfunc.anafast(density1, map2=density2, lmax=l_max, iter=1)
                    #result = result/(self.pix_win_f[:len(result)]**2)
                    theory_index = list(self.cls_label_glass).index([map1_index, map2_index])
                    theory = self.cls_glass_list[theory_index]

                    if num==1:
                        axs.plot(result[1:l_max], label="Simulation", color="blue")
                        axs.plot(theory[1:l_max], label="theory", color="red")
                        axs.set_title(plot_name)
                        axs.set_yscale('log')
                    else:
                        axs[i, j].plot(result[1:l_max], label="Simulation", color="blue")
                        axs[i, j].plot(theory[1:l_max], label="Theory", color="red")
                        if i == j:
                            axs[i, j].set_title(plot_name)

                        axs[i, j].set_yscale('log')
                        axs[i, j].set_xscale('log')
                        for k in range(i + 1, num):
                            axs[i, k].axis('off')

            plt.subplots_adjust(wspace = 0.2, hspace = 0.0)
            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)


        return cls_index_sublist

    def calculate_cls_auto(self, list_of_index, l_max, plot = True, Poisson=False, denoise_P = False, ref_N = 1000, tqdm_bar = True):
        """
        calculate auto correlation with for the list of index
        :param list_of_index: the index of REAL redshift bin STARTED with 1
        :return:
        """
        list_of_index = list(list_of_index)
        list_of_index.sort()
        num = len(list_of_index)

        if Poisson == False:
            plot_list = self.hp_density_list

        else:
            Poisson_map, Poisson_density = self.add_Poisson(ref_N, tqdm_bar = tqdm_bar)
            plot_list = Poisson_density

        noise_list = [0] * num
        cls_result = [None] * num
        l = np.array(range(1, l_max))
        for i in range(num):
            map1_healpix = plot_list[list_of_index[i] - 1]

            if Poisson and denoise_P:
                noise = 4 * np.pi/np.sum(Poisson_map[list_of_index[i] - 1])
                noise_list[i] = noise

            result = hp.sphtfunc.anafast(map1_healpix, map2=map1_healpix, lmax=l_max, iter=0)
            result = result - noise_list[i]
            #result = result/(self.pix_win_f[:len(result)]**2)
            cls_result[i] = result[:l_max]
            noise_list = np.array(noise_list)

        if plot:
            fig, axs = plt.subplots(1, num, figsize=(4*len(list_of_index), 2.5))
            for i in range(num):
                map1_index = list_of_index[i]
                plot_name = 'W{0}xW{1}'.format(map1_index, map1_index)
                theory_index = list(self.cls_label_glass).index([map1_index, map1_index])
                theory = self.cls_glass_list[theory_index]
                estimator = cls_result[i]
                noise_here = noise_list[i]

                if num == 1:
                    if Poisson and denoise_P:
                        axs.plot(l, l * (l + 1) * (estimator[1:l_max]+noise_here), label="with shot noise", color="green")
                    axs.plot(l, l*(l+1)*estimator[1:l_max], label="Simulation", color="blue")
                    axs.plot(l, l*(l+1)*theory[1:l_max], label="Theory", color="red")
                    axs.set_title(plot_name)
                    axs.set_xlabel(r'$l$')
                    axs.set_ylabel(r'$l(l+1)$')

                    axs.set_yscale('log')
                    axs.set_xscale("log")

                else:
                    if Poisson and denoise_P:
                        axs[i].plot(l, l * (l + 1) * (estimator[1:l_max]+noise_here), label="with shot noise", color="green")
                    axs[i].plot(l, l*(l+1)*estimator[1:l_max], label="Simulation", color="blue")
                    axs[i].plot(l, l*(l+1)*theory[1:l_max], label="Theory", color="red")
                    axs[i].set_title(plot_name)
                    axs[i].set_xlabel(r'$l$')
                    axs[i].set_ylabel(r'$l(l+1)$C$_{l}$')

                    axs[i].set_yscale('log')
                    axs[i].set_xscale('log')
                    if i == 0:
                        axs[i].legend(loc='lower right')
                #plt.tight_layout()
            plt.subplots_adjust(wspace=0.3, hspace=0.5)
                    #for k in range(i + 1, num):
                        #axs[i, k].axis('off')
                    #for k in range(0, i):
                        #axs[i, k].axis('off')
            #fig.legend(loc='upper right')

        return np.array(cls_result)

    def add_Poisson(self, ref_N, tqdm_bar = True):
        expectation = np.array((self.hp_density_list+1)*ref_N)
        Poissoned = []
        Poissoned_density = []

        if tqdm_bar:
            for i in tqdm.tqdm(range(np.shape(self.hp_density_list)[0])):
                expectation_i = expectation[i]
                Poissoned_i = np.random.poisson(expectation_i)
                Poissoned.append(Poissoned_i)
                Poissoned_density_i = Poissoned_i / np.mean(Poissoned_i) - 1
                Poissoned_density.append(Poissoned_density_i)

        else:
            for i in range(np.shape(self.hp_density_list)[0]):
                expectation_i = expectation[i]
                Poissoned_i = np.random.poisson(expectation_i)
                Poissoned.append(Poissoned_i)
                Poissoned_density_i = Poissoned_i / np.mean(Poissoned_i) - 1
                Poissoned_density.append(Poissoned_density_i)

        Poissoned = np.array(Poissoned)
        Poissoned_density = np.array(Poissoned_density)
        return Poissoned, Poissoned_density


