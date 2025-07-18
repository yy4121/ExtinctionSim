import pandas as pd
import numpy as np
import tqdm
import healpy as hp
from astropy.io import fits
import pickle as pkl
from astropy import units as u
#
# import os
# print(os.getcwd())

"""Catalogue parameter"""
cols = ["ra", "dec","redshift", "redshift_true", "mag_u_lsst", "mag_g_lsst", "mag_r_lsst", "mag_i_lsst", "mag_z_lsst", "mag_y_lsst"]
template_cata = pd.read_csv('../../data/model_import_data/CosmoDC2/table_irsa_catalog_search_results1.tbl'.format(1), skiprows=39, sep=r'\s+', names=cols)

for i in range(2, 9):
    df1 = pd.read_csv('../../data/Model_import_data/CosmoDC2/table_irsa_catalog_search_results{}.tbl'.format(i), skiprows=39, sep=r'\s+', names=cols)
    template_cata = pd.concat([template_cata, df1])
template_cata.insert(0,'id',np.arange(len(template_cata)))

template_cata = template_cata[template_cata["mag_i_lsst"]<26.5]


"""model parameter"""
nside = 512
lmax = 1024
zbin_in =np.linspace(0.2, 1.5, 41) #glass.distance_grid(cosmo, 0.2, 1.2, dx=700.0)#np.linspace(0.2, 1.2, 13)##np.linspace(0, 1.5, 18)#np.linspace(0.2, 1.5, 18) #np.linspace(0, 3.0, 39)#np.linspace(0, 1.5, 20) ###np.linspace(0, 2, 20)
zcenter_in = (zbin_in[:-1] + zbin_in[1:])/2
zbin_out = np.linspace(0.5, 1.2, 6)#np.linspace(0, 3.0, 4)#np.linspace(0, 1.5, 4) ###np.linspace(0, 2, 10)
zcenter_out = (zbin_out[:-1] + zbin_out[1:])/2
Ncount_hist = np.histogram(template_cata["redshift_true"], bins=zbin_in)[0]


"""Template Catalogue for sampling"""

template_cata_shell = []
for i in tqdm.tqdm(range(len(zcenter_in))):
    zmin_shell = zbin_in[i]
    zmax_shell = zbin_in[i + 1]
    shell_cata = template_cata.query("@zmin_shell < redshift_true < @zmax_shell").reset_index(drop=True)
    template_cata_shell.append(shell_cata)


"""SFD map """
hdulist = fits.open("../../data/model_import_data/EBV_map/EBV_SFD98_1_512.fits")
data = np.array(hdulist[1].data)
header = hdulist[0].header
E_BV = []
for i in range(len(data)):
    if i in range(0, 10000):
        map_ = list(data[i][0])
        E_BV.extend(map_)
    else:
        E_BV.extend([0]*1024)
SFD = np.array(hp.reorder(np.array(E_BV), n2r=True))
SFD = hp.ud_grade(SFD, nside_out=nside)*0.86

"""Planck dust map"""
hdulist2 = fits.open('../../data/model_import_data/EBV_map/HFI_CompMap_ThermalDustModel_2048_R1.20.fits')
data = np.array(hdulist2[1].data)
map_low_res = hp.ud_grade(data["EBV"], order_in='NESTED', order_out='RING', nside_out=nside)
Planck = np.array(hp.Rotator(coord=['G', 'C']).rotate_map_pixel(map_low_res))




"""Photo_z estimator"""
PZestimator_RF = pkl.load(open('../../data/model_import_data/Photo_z_emulator/LSST_RF.pkl', 'rb'))
def photo_z_estimator_RF(all_columns, ugriz_name):
    gu = all_columns[ugriz_name[1]] - all_columns[ugriz_name[0]]
    rg = all_columns[ugriz_name[2]] - all_columns[ugriz_name[1]]
    ir = all_columns[ugriz_name[3]] - all_columns[ugriz_name[2]]
    zi = all_columns[ugriz_name[4]] - all_columns[ugriz_name[3]]
    yz = all_columns[ugriz_name[5]] - all_columns[ugriz_name[4]]
    return PZestimator_RF.predict(np.array([gu, rg, ir,zi, yz]).T)



PZestimator_HGBR = pkl.load(open('../../data/model_import_data/Photo_z_emulator/LSST_HGBR.pkl', 'rb'))
def photo_z_estimator_HGBR(all_columns, ugriz_name):
    gu = all_columns[ugriz_name[1]] - all_columns[ugriz_name[0]]
    rg = all_columns[ugriz_name[2]] - all_columns[ugriz_name[1]]
    ir = all_columns[ugriz_name[3]] - all_columns[ugriz_name[2]]
    zi = all_columns[ugriz_name[4]] - all_columns[ugriz_name[3]]
    yz = all_columns[ugriz_name[5]] - all_columns[ugriz_name[4]]
    return PZestimator_HGBR.predict(np.array([gu, rg, ir,zi, yz]).T)



"""Everything about observation"""
#Band_coefficient = {"u": 4.134990, "g": 3.15247, "r":2.31857, "i": 1.77388, "z":1.36388, "y": 1.130125}
from photerr import LsstErrorModel
errModel_dust = LsstErrorModel(nYrObs=10,
                          renameDict={"u": "mag_u_lsst_dust", "g": "mag_g_lsst_dust", "r": "mag_r_lsst_dust", "i": "mag_i_lsst_dust",
                                      "z": "mag_z_lsst_dust", "y": "mag_y_lsst_dust"})
errModel_clean = LsstErrorModel(nYrObs=10,
                          renameDict={"u": "mag_u_lsst", "g": "mag_g_lsst", "r": "mag_r_lsst", "i": "mag_i_lsst",
                                      "z": "mag_z_lsst", "y": "mag_y_lsst"})
#, absFlux=True, sigLim=0#




"""Band filter and SED"""

# from astropy.table import Table
# lsst_filter = {}
# for i in "ugrizy":
#     item ={}
#     filter = Table.read("http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=LSST/LSST.{}".format(i))
#     item["wl"] = np.array(filter["Wavelength"].value)*u.AA
#     item["F_trans"] = np.array(filter["Transmission"].value)
#     lsst_filter[i] = item

SED_wl = np.loadtxt("../../data/model_import_data/SED/LAMBDA_SLN.DAT") * u.AA
SED1 = np.loadtxt("../../data/model_import_data/SED/T07000G45M05V000K2ANWNVR20N.ASCR")
SED2 = np.loadtxt("../../data/model_import_data/SED/T07000G45M05V000K2SNWNVR20N.ASCR")
