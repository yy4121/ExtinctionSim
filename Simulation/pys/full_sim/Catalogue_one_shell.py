import numpy as np
import tqdm
import pandas as pd
from itertools import chain
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)


def ExpNcount_map_shells(all_density_map, all_Ncount_hist, expectation_total_sample = 2e8):
    Ncount_shell_map_unnormalized = np.array([(all_density_map[i]+1)  * all_Ncount_hist[i] for i in range(len(all_Ncount_hist))])
    Ncount_shell = Ncount_shell_map_unnormalized / (np.sum(Ncount_shell_map_unnormalized, axis=(0, 1)) / expectation_total_sample)
    return Ncount_shell

class Catalogue_one_shell:
    def __init__(self, ExpNcount_map_shell, template_cata_shell, Obs_parameter):
        self.ExpNcount_map_shell = ExpNcount_map_shell
        self.template_cata_shell = template_cata_shell
        self.nside = Obs_parameter["nside"]
        self.EBV_in = Obs_parameter["EBV_in"]
        self.EBV_out = Obs_parameter["EBV_out"]
        self.Band_coefficient_in = Obs_parameter["Band_coefficient_in"]
        self.Band_coefficient_out = Obs_parameter["Band_coefficient_out"]
        self.ErrorModel_clean = Obs_parameter["ErrorModel_clean"]
        self.ErrorModel_obs = Obs_parameter["ErrorModel_obs"]
        self.Rv_map = Obs_parameter["Rv_map"]
        self.mask = Obs_parameter["mask"]
        self.photoz_est = Obs_parameter["photoz_est"]
        assert (len(self.ExpNcount_map_shell) == len(self.EBV_in))
        assert (len(self.ExpNcount_map_shell) == len(self.EBV_out))
        assert 12*(self.nside**2) == len(self.ExpNcount_map_shell)
        self.sampling_flag = False
        self.extinction_flag = False
        self.dust_noise_flag = False
        self.clean_noise_flag = False
        self.deredden_flag = False
        self.photo_z_flag = False
        self.extinction_info = False
        self.zbin_in = Obs_parameter["zbin_in"]
        self.zbin_out = Obs_parameter["zbin_out"]
        self.Detected_map_shell_masked = {"clean": None, "dust": None}

    def Sample_photometry(self):
        Renormalizaed_ExpNcount = self.ExpNcount_map_shell
        self.Ncount_map_shell =  np.random.poisson(Renormalizaed_ExpNcount)
        self.Ncount_map_shell_masked = self.Ncount_map_shell*self.mask

        sampled_catalogue_shell_index = np.random.choice(self.template_cata_shell.index,
                                                         size=sum(self.Ncount_map_shell_masked), replace=True)
        self.sampled_catalogue_shell = self.template_cata_shell.iloc[sampled_catalogue_shell_index].reset_index(
            drop=True)

        healpix_id = [[i] * self.Ncount_map_shell_masked[i] for i in range(len(self.Ncount_map_shell_masked))]
        healpix_id = list(chain.from_iterable(healpix_id))
        self.sampled_catalogue_shell.insert(2, 'healpix_index', healpix_id)
        self.sampling_flag = True

    def Add_extinction_info(self):
        assert self.sampling_flag == True, "run self.Sample_photometry() first"
        assert self.extinction_info == False, "this step is done, check self.sampled_catalogue_shell"
        EBV_in_ = [[self.EBV_in[i]] * self.Ncount_map_shell_masked[i] for i in range(len(self.Ncount_map_shell_masked))]
        EBV_in_ = list(chain.from_iterable(EBV_in_))
        self.sampled_catalogue_shell.insert(3, 'EBV_in', EBV_in_)

        EBV_out_ = [[self.EBV_out[i]] * self.Ncount_map_shell_masked[i] for i in
                    range(len(self.Ncount_map_shell_masked))]
        EBV_out_ = list(chain.from_iterable(EBV_out_))
        self.sampled_catalogue_shell.insert(4, 'EBV_out', EBV_out_)

        Rv_ = [[self.Rv_map[i]] * self.Ncount_map_shell_masked[i] for i in
                    range(len(self.Ncount_map_shell_masked))]
        Rv_ = list(chain.from_iterable(Rv_))
        self.sampled_catalogue_shell.insert(5, 'true_Rv', Rv_)

        for band in "ugrizy":
            column_name = "true_coefficient_{}".format(band)
            cofficient_fun = self.Band_coefficient_in[band]
            coefficient = cofficient_fun(Rv_)
            self.sampled_catalogue_shell[column_name] = coefficient
        self.extinction_info = True

    def Add_Exintction(self):
        assert self.extinction_info == True, "run Add_extinction_info() first"
        assert self.clean_noise_flag == False, "mag has noise"
        assert self.extinction_flag == False, "this step is done, check self.sampled_catalogue_shell"
        for band in "ugrizy":
            mag_column = self.sampled_catalogue_shell["mag_{}_lsst".format(band)].copy()
            EBV = self.sampled_catalogue_shell["EBV_in"].copy()
            band_coefficient_in = self.sampled_catalogue_shell["true_coefficient_{}".format(band)].copy()
            mag_dust = mag_column + band_coefficient_in * EBV
            self.sampled_catalogue_shell["mag_{}_lsst_dust".format(band)]=mag_dust
        self.extinction_flag = True
        del EBV
        del mag_column
        del band_coefficient_in
        del mag_dust

    def Add_noise(self, dust_noise = True, clean_noise = False):
        if dust_noise == True:
            assert self.extinction_flag == True, "run Add_extinction() first"
            assert self.dust_noise_flag == False, "this step is done, check self.sampled_catalogue_shell"
            self.sampled_catalogue_shell = self.ErrorModel_obs(self.sampled_catalogue_shell, random_state=42)
            self.dust_noise_flag =True

        if clean_noise == True:
            assert self.clean_noise_flag == False, "this step is done, check self.sampled_catalogue_shell"
            self.sampled_catalogue_shell = self.ErrorModel_clean(self.sampled_catalogue_shell, random_state=42)
            self.clean_noise_flag = True


    def Dereddening(self):
        assert self.dust_noise_flag == True, "run Add_noise() for extinction first"
        assert self.deredden_flag == False, "this step is done, check self.sampled_catalogue_shell"
        for band in "ugrizy":
            mag_column = self.sampled_catalogue_shell["mag_{}_lsst_dust".format(band)].copy()
            EBV = self.sampled_catalogue_shell ["EBV_out"].copy()
            mag_de = mag_column - self.Band_coefficient_out[band] * EBV
            self.sampled_catalogue_shell["mag_{}_lsst_dust_derred".format(band)]= mag_de
            del mag_column
            del EBV

    def Estimate_photo_z(self, Photo_z_est_choice = 0, choice = "dust"):
        assert choice in ["dust", "clean"], "dust option much be either dust or clean"
        if choice == "dust":
            assert self.dust_noise_flag == True, "dust noise not added"
            input_column_name = ["mag_{}_lsst_dust_derred".format(i) for i in "ugrizy"]
            output_column_name = "dust_photo_z"
        else:
            assert self.clean_noise_flag == True, "clean noise not added"
            input_column_name = ['mag_{}_lsst'.format(i) for i in "ugrizy"]
            output_column_name = "clean_photo_z"
        assert output_column_name not in self.sampled_catalogue_shell.columns, "work done already"

        import_cata =self.sampled_catalogue_shell[input_column_name].copy()
        import_cata = import_cata.replace([np.inf, -np.inf], np.nan).dropna()
        Photo_z_est = self.photoz_est[Photo_z_est_choice]
        photoz = Photo_z_est(import_cata, input_column_name)
        new_data = pd.Series(photoz, index=import_cata.index)
        self.sampled_catalogue_shell[output_column_name] = new_data
        self.sampled_catalogue_shell.loc[(self.sampled_catalogue_shell[output_column_name] > max(self.zbin_out)) |
                                         (self.sampled_catalogue_shell[output_column_name] < min(self.zbin_out)),
                                            output_column_name] = 100
        self.sampled_catalogue_shell[output_column_name] = self.sampled_catalogue_shell[output_column_name].fillna(-1)
        del import_cata

    def Statistics_on_shell(self,zbin_out, choice = "dust"):
        """return the statistics on shell, which reflects the distribution of
        samples from this true redshift shell on photo-z shell.
        So combined with information on other shell, it can use to reach
        1. Ncount maps with , #= number of photo_z tomographic bins
        2. # true redshift distribution on photot-z shell(each len = z_bin_in).
        """
        assert choice in ["dust", "clean"], "dust option much be either dust or clean"
        healpix_bins = np.arange(len(self.Ncount_map_shell_masked) + 1) - 0.05
        healpix_data = self.sampled_catalogue_shell["healpix_index"].copy()
        if choice == "dust":
            photo_z_data =  self.sampled_catalogue_shell["dust_photo_z"]
            photo_z_name = "dust_photo_z"
        else:
            photo_z_data = self.sampled_catalogue_shell["clean_photo_z"]
            photo_z_name = "clean_photo_z"

        ####Set every photo_z that np. nan to -1
        zbin_out_all = np.concatenate((np.array([-2]),zbin_out, np.array([101]) ))
        hist, xedges, yedges = np.histogram2d(healpix_data, photo_z_data,
                                               bins=[healpix_bins, zbin_out_all])
        del healpix_data
        del photo_z_data
        return hist


def Simulate_shells(all_density_map, all_Ncount_hist,
                    template_cata_shell, Obs_parameter, zbin_out,
                    expectation_total_sample = 2e8, choice = "dust", photo_z_est_choice = 0):
    assert choice in ["dust", "clean", "both"]
    assert len(all_density_map) == len(all_Ncount_hist)
    assert len(all_density_map) == len(template_cata_shell)
    Ncount_Exp_shells_map = ExpNcount_map_shells(all_density_map, all_Ncount_hist,
                                                 expectation_total_sample = expectation_total_sample)

    clean_hist = []
    dust_hist = []
    Detected_shells_maps = {"clean":[], 'dust':[]}
    Ncount_sampled_shells_maps = {"clean":[], 'dust':[]}
    Expected_sampled_shells_maps = {"clean":[], 'dust':[]}

    for i in tqdm.tqdm(range(len(Ncount_Exp_shells_map))):
        Ncount_map = Ncount_Exp_shells_map[i]
        cata_one_shell = Catalogue_one_shell(Ncount_map,
                                                        template_cata_shell[i], Obs_parameter)
        cata_one_shell.Sample_photometry()

        if choice in ["dust", "both"]:
            cata_one_shell.Add_extinction_info()
            cata_one_shell.Add_Exintction()
            cata_one_shell.Add_noise(clean_noise=False, dust_noise=True)
            cata_one_shell.Dereddening()
            cata_one_shell.Estimate_photo_z(Photo_z_est_choice = photo_z_est_choice, choice = "dust")
            hist_dust_shell = cata_one_shell.Statistics_on_shell(zbin_out, choice = "dust")
            dust_hist.append(hist_dust_shell)

        if choice in ["clean", "both"]:
            cata_one_shell.Add_noise(clean_noise=True, dust_noise=False)
            cata_one_shell.Estimate_photo_z(Photo_z_est_choice = photo_z_est_choice, choice = "clean")
            hist_clean_shell = cata_one_shell.Statistics_on_shell(zbin_out, choice="clean")
            clean_hist.append(hist_clean_shell)

    return {"clean": np.array(clean_hist), "dust":np.array(dust_hist)}











