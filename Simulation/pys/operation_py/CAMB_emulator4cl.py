import Simulation.pys.para.initial_para as Cp
import Simulation.pys.para.model_para as Mp
import Simulation.pys.full_sim.Cls_theory as Cl
from importlib import reload
import csv
import tqdm
import numpy as np
reload(Cp)


def sim_for_emu():
    para_C = Cp.prior_cosmo_Linear()
    shell = Cl.Shell_Cls(para_C, Mp.zbin_in,upper_f=0.8, plot=False)
    shell_cl = shell.cls_shell_theory(Mp.lmax)
    return para_C, shell_cl



def save_large_array_to_csv(filename, row):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

file_name = "../../data/camb_emulator_full4.csv"

#for i in tqdm.tqdm(range(1000)):
    # para_, shell_cl_ =sim_for_emu()
    # print(np.shape([para_["H0"], para_['ombh2'], para_['omch2'], para_["ns"], para_["s8"]]))
    # para_part = np.concatenate([para_["H0"], para_['ombh2'], para_['omch2'], para_["ns"], para_["s8"]])
    # G = realization.GenerateShell(40, shell_cl_, Mp.lmax, Mp.nside)
    # cl_part = np.concatenate(G.auto_cls_glass())
    # all = np.concatenate([para_part, cl_part])
    # file_name = "../Model_data/camb_emulator.csv"
    # save_large_array_to_csv(file_name, all)




for i in tqdm.tqdm(range(500)):
    try:
        para_, shell_cl_ = sim_for_emu()
        para_part = np.array([para_["H0"], para_["ombh2"], para_["omch2"], para_["ns"], para_["s8"]])
        #G = realization.GenerateShell(40, shell_cl_, Mp.lmax, Mp.nside)
        #cl_part = np.concatenate(G.auto_cls_glass())
        cl_part = np.concatenate(shell_cl_)
        all_data = np.concatenate([para_part, cl_part])
        #save_large_array_to_csv(file_name, all_data)

    except SystemExit as e:
        if e.code == 1:
            print(f"\n[Iteration {i}] sim_for_emu exited with code 1. Skipping.")
            continue
        else:
            raise  # Re-raise other exit codes

    except Exception as e:
        print(f"\n[Iteration {i}] Exception occurred: {e}. Skipping.")
        continue
