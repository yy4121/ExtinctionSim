import numpy as np
import pandas as pd
import tqdm
class Catalogue_one_box_shell:
    def __init__(self, EBV_in, Rv, EBV_out, target_num, template_cata_shell, Obs_parameter):
        self.EBV_in = EBV_in
        self.Rv = Rv
        self.EBV_out = EBV_out

        self.template_cata_shell = template_cata_shell
        self.target_num = target_num
        self.Band_coefficient_in = Obs_parameter["Band_coefficient_in"]
        self.zbin_in = Obs_parameter["zbin_in"]
        self.zbin_out = Obs_parameter["zbin_out"]

        self.ErrorModel_clean = Obs_parameter["ErrorModel_clean"]
        self.ErrorModel_obs = Obs_parameter["ErrorModel_obs"]
        self.Band_coefficient_out = Obs_parameter["Band_coefficient_out"]
        self.photoz_est = Obs_parameter["photoz_est"]
        self.dust_noise_flag = False
        self.clean_noise_flag = False
        self.deredden_flag = False
        self.extinction_flag = False

    def Sample_cata_box_shell(self):
        Sampled_index = np.random.choice(self.template_cata_shell.index, size= self.target_num, replace=True)
        self.box_shell_catalogue = self.template_cata_shell.iloc[Sampled_index].reset_index(drop=True)


    def Add_extinction_info(self):
        self.box_shell_catalogue["EBV_in"] =  [self.EBV_in] * self.target_num
        self.box_shell_catalogue["true_Rv"] = [self.Rv] * self.target_num
        self.box_shell_catalogue["EBV_out"] = [self.EBV_out] * self.target_num
        ##No need for this, since information are defined here

        for band in "ugrizy":
            column_name = "true_coefficient_{}".format(band)
            cofficient_fun = self.Band_coefficient_in[band]
            coefficient = cofficient_fun(self.Rv)
            self.box_shell_catalogue[column_name] = coefficient
        self.extinction_info = True
        return

    def Add_extinction(self):
        assert self.clean_noise_flag == False
        for band in "ugrizy":
            mag_column = self.box_shell_catalogue["mag_{}_lsst".format(band)].copy()
            EBV = self.box_shell_catalogue["EBV_in"].copy()
            band_coefficient_in = self.box_shell_catalogue["true_coefficient_{}".format(band)].copy()
            mag_dust = mag_column + band_coefficient_in * EBV
            self.box_shell_catalogue["mag_{}_lsst_dust".format(band)] = mag_dust
        self.extinction_flag = True
        del EBV
        del mag_column
        del mag_dust

    def Add_noise(self, dust_noise=True, clean_noise=False):
        if dust_noise == True:
            assert self.extinction_flag == True, "run Add_extinction() first"
            assert self.dust_noise_flag == False, "this step is done, check self.sampled_catalogue_shell"
            self.box_shell_catalogue = self.ErrorModel_obs(self.box_shell_catalogue, random_state=42)
            self.dust_noise_flag = True

        if clean_noise == True:
            assert self.clean_noise_flag == False
            self.box_shell_catalogue = self.ErrorModel_clean(self.box_shell_catalogue, random_state=42)
            self.clean_noise_flag = True

    def Dereddening(self):
        assert self.dust_noise_flag == True, "run Add_noise() for extinction first"
        assert self.deredden_flag == False, "this step is done, check self.sampled_catalogue_shell"

        for band in "ugrizy":
            mag_column = self.box_shell_catalogue["mag_{}_lsst_dust".format(band)].copy()
            EBV = self.box_shell_catalogue["EBV_out"].copy()
            mag_de = mag_column - self.Band_coefficient_out[band] * EBV
            self.box_shell_catalogue["mag_{}_lsst_dust_derred".format(band)] = mag_de
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

        assert output_column_name not in self.box_shell_catalogue.columns, "work done already"
        import_cata =self.box_shell_catalogue[input_column_name].copy()
        import_cata = import_cata.replace([np.inf, -np.inf], np.nan).dropna()
        Photo_z_est = self.photoz_est[Photo_z_est_choice]
        photoz = Photo_z_est(import_cata, input_column_name)
        new_data = pd.Series(photoz, index=import_cata.index)
        self.box_shell_catalogue[output_column_name] = new_data
        self.box_shell_catalogue.loc[(self.box_shell_catalogue[output_column_name] > max(self.zbin_out)) |
                                     (self.box_shell_catalogue[output_column_name] < min(self.zbin_out)),
                                        output_column_name] = 100
        self.box_shell_catalogue[output_column_name] = self.box_shell_catalogue[output_column_name].fillna(-1)
        del import_cata

    def Statistics_on_box_shell(self, choice = "dust"):
        assert choice in ["dust", "clean"], "dust option much be either dust or clean"
        zbin_out_all = np.concatenate((np.array([-2]), self.zbin_out, np.array([101])))
        if choice == "dust":
            photo_z = self.box_shell_catalogue["dust_photo_z"].copy()

        else:
            photo_z = self.box_shell_catalogue["clean_photo_z"].copy()
        hist, _= np.histogram(photo_z, bins=zbin_out_all)
        return hist


def dusteffect_one_box(One_EBV_Rv_label, EBV_in_bin, Rv_bin, EBV_out_bin, template_cata_shell,target_num_list,
                     Obs_parameter, choice = "dust"):
    assert choice in ["dust", "clean", "both"]
    assert len(template_cata_shell) ==len(target_num_list)
    EBV_in_id, Rv_id, EBV_out_id = One_EBV_Rv_label
    EBV_in_uniform = (EBV_in_bin[EBV_in_id] + EBV_in_bin[EBV_out_id+1])/2
    EBV_out_uniform = (EBV_out_bin[EBV_out_id] + EBV_out_bin[EBV_out_id+1])/2
    Rv_uniform = (Rv_bin[Rv_id] + Rv_bin[Rv_id+1])/2
    dust_effect_on_box = []
    clean_effect_on_box =[]
    for z_id in range(len(target_num_list)):
        box_shell = Catalogue_one_box_shell(EBV_in_uniform, Rv_uniform, EBV_out_uniform, target_num_list[z_id],
                                            template_cata_shell[z_id],Obs_parameter)
        box_shell.Sample_cata_box_shell()
        if choice in ["dust", "both"]:
            box_shell.Add_extinction_info()
            box_shell.Add_extinction()
            box_shell.Add_noise(dust_noise=True, clean_noise=False)
            box_shell.Dereddening()
            box_shell.Estimate_photo_z(choice="dust")
            box_result_dust = box_shell.Statistics_on_box_shell(choice="dust")
            dust_effect_on_box.append(box_result_dust)


        if choice in ["clean", "both"]:
            box_shell.Add_noise(dust_noise=False, clean_noise=True)
            box_shell.Estimate_photo_z(choice="clean")
            box_result_clean = box_shell.Statistics_on_box_shell(choice="clean")
            clean_effect_on_box.append(box_result_clean)





    del box_shell


    return {"clean": np.array(clean_effect_on_box), "dust": np.array(dust_effect_on_box)}


def all_dusteffect(EBV_Rv_label, EBV_in_bin, Rv_bin, EBV_out_bin, template_cata_shell,target_num_list,
                     Obs_parameter, choice = "dust"):

    assert choice in ["dust", "clean", "both"]
    dust_effect_clean = {}
    dust_effect_dust = {}
    for One_EBV_Rv_label in tqdm.tqdm(EBV_Rv_label):
        One_dust_effect = dusteffect_one_box(One_EBV_Rv_label, EBV_in_bin, Rv_bin, EBV_out_bin, template_cata_shell,target_num_list,
                     Obs_parameter, choice = choice)

        if choice in ["clean", "both"]:
            dust_effect_clean[One_EBV_Rv_label] = One_dust_effect["clean"]

        if choice in ["dust", "both"]:
            dust_effect_dust[One_EBV_Rv_label] = One_dust_effect["dust"]

    return {"clean": dust_effect_clean, "dust": dust_effect_dust}





