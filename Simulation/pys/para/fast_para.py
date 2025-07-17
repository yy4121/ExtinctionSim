import numpy as np


EBV_in_num = 30
EBV_in_bins = np.linspace(0, 0.3, EBV_in_num+1)#np.quantile(Mp.SFD[Mp.SFD<0.2], np.linspace(0, 1, EBV_in_num + 1))
EBV_in_bins[0] = 0

EBV_out_num = 20
EBV_out_bins = np.linspace(0, 0.2, EBV_out_num+1)#np.quantile(Mp.SFD[Mp.SFD<0.2], np.linspace(0, 1, EBV_in_num + 1))
EBV_out_bins[0] = 0


Rv_num = 15
Rv_bins = np.linspace(2, 5, Rv_num+1)
