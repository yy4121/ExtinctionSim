import Simulation.pys.para.model_para as Mp
import Simulation.pys.full_sim.Catalogue_one_shell as cata_shell
import Simulation.pys.full_sim.cl2realization as realization
import Simulation.pys.others.Statistics_acc as St
import numpy as np
from time import time
from importlib import reload
import tqdm
import csv
import pickle as pkl

shell_cl = np.load('../../data/test_model/camb_40cls.npy')

with open('../../data/test_model/test_dust_pack.pkl', 'rb') as f:
    dust_pack = pkl.load(f)

file_name = "../../data/test_model/Fixed_import_simulation.csv"

obs_f = {"EBV_in": dust_pack['EBV_in'], #np.zeros_like(EBV_in),
         "EBV_out": Mp.SFD, #np.zeros_like(EBV_out),
         "Band_coefficient_in": dust_pack['band_coefficient_Rv'],
         "Band_coefficient_out": dust_pack['band_coefficient_ccm_31'],
         "ErrorModel_clean": Mp.errModel_clean,
         "ErrorModel_obs":Mp.errModel_dust,
         "nside":Mp.nside,
         "photoz_est": [Mp.photo_z_estimator_RF],
         "Rv_map":  dust_pack["Rv_map"],
         "mask": Mp.SFD<0.2,
         "zbin_in": Mp.zbin_in,
         "zbin_out": Mp.zbin_out}

def save_large_array_to_csv(filename, row):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def generate_sim():
    G = realization.GenerateShell(40, shell_cl, Mp.lmax, Mp.nside)
    density_map = G.realization_density(lognormal=True)
    results = cata_shell.Simulate_shells(density_map, Mp.Ncount_hist, Mp.template_cata_shell,
                                         obs_f, Mp.zbin_out, choice="dust", expectation_total_sample=2e8)
    dust_completed = St.sim_data(results['dust'][:,:,1:-1], obs_f["mask"], print_ = True)
    dust_obs_cl = np.array(dust_completed.cl_photoz_shell([0, 1, 2, 3, 4], 1024)).reshape(-1)
    save_large_array_to_csv(file_name, dust_obs_cl)


if __name__ == "__main__":
    for i in tqdm.tqdm(range(1)):
        generate_sim()