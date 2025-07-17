import numpy as np
import numpy.random as npr
"""Cosmological parameter"""
H0_0=67
ombh2_0=0.022
omch2_0=0.122
ns_0=0.965
s8_0 = 0.811
p_cosmo_0 = {"H0": H0_0, "ombh2": ombh2_0, "omch2": omch2_0, "ns": ns_0, "s8": s8_0}

H0_sig = 7/2
ombh2_sig=0.007/2
omch2_sig=0.05/2
ns_sig=0.02/2
s8_sig = 0.25/2
p_cosmo_sig = {"H0": H0_sig, "ombh2": ombh2_sig, "omch2": omch2_sig, "ns":ns_sig, "s8": s8_sig}
keys_cosmo = ["H0", "ombh2", "omch2", "ns", "s8"]

def prior_cosmo_Gaussian():
    p = {}
    for key in keys_cosmo:
        p[key] = npr.normal(loc = p_cosmo_0[key], scale=p_cosmo_sig[key])
    return p


def prior_cosmo_Linear():
    p = {}
    for key in keys_cosmo:
        p[key] = npr.uniform(low = p_cosmo_0[key]-2*p_cosmo_sig[key], high = p_cosmo_0[key]+2*p_cosmo_sig[key])
    return p

def prior_cosmo_f():
    p = {}
    for key in keys_cosmo:
        p[key] = p_cosmo_0[key]
    return p



"""EBVmap"""
EBV_N1_0 = 1
EBV_N2_0 = 1
EBV_mix_0 = 1

EBV_N1_sig = 0.1
EBV_N2_sig = 0.1
EBV_mix_sig = None


p_EBV_0 = {"EBV_N1": EBV_N1_0, "EBV_N2": EBV_N2_0, "EBV_mix": EBV_mix_0}
p_EBV_sig = {"EBV_N1": EBV_N1_sig, "EBV_N2": EBV_N2_sig, "EBV_mix": EBV_mix_sig}
keys_EBV = ["EBV_N1", "EBV_N2", "EBV_mix"]

def prior_EBV():
    p = {}
    for key in keys_EBV:
        if key != "EBV_mix":
            p[key] = npr.normal(loc = p_EBV_0[key], scale=p_EBV_sig[key])
        else:
            p[key] = npr.uniform(low=0.3, high=1.0)
    return p


"""Dust law"""
ccm89_infrared_a = np.array([0.574])
ccm89_infrared_b = np.array([-0.527])
ccm89_optical_a = np.array([1, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
ccm89_optical_b = np.array([0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])

p_dust_law_0 = {"infrared_a" : ccm89_infrared_a,
              "infrared_b" : ccm89_infrared_b,
              "optical_a" : ccm89_optical_a,
              "optical_b" : ccm89_optical_b}

p_dust_law_sig = {"infrared_a" : ccm89_infrared_a*0.01,
              "infrared_b" : ccm89_infrared_b*0.01,
              "optical_a" : ccm89_optical_a*0.01,
              "optical_b" : ccm89_optical_b*0.01}

keys_dust_law = ["infrared_a","infrared_b", "optical_a", "optical_b"]
def prior_dust_law():
    p = {}
    for key in keys_dust_law:
        p[key] = npr.normal(loc=p_dust_law_0[key], scale=np.abs(p_dust_law_sig[key]))
    return p


"""Rv"""
Rv_a_0 = 0.72
Rv_b_0 = 3.1
Rv_sig_0 = 0.3

Rv_a_sig = 0.72*0.02
Rv_b_sig = 3.1*0.02
Rv_sig_sig = 0.0


p_Rv_0 = {"Rv_a": Rv_a_0, "Rv_b": Rv_b_0, "Rv_sig":Rv_sig_0}
p_Rv_sig = {"Rv_a": Rv_a_sig, "Rv_b": Rv_b_sig, "Rv_sig":Rv_sig_sig}
keys_Rv = ["Rv_a", "Rv_b", "Rv_sig"]

def prior_Rv():
    p = {}
    for key in keys_Rv:
        p[key] = npr.normal(loc=p_Rv_0[key], scale=np.abs(p_Rv_sig[key]))
    return p

def prior_repara_dustlaw():
    m = npr.uniform(low=0.0, high=1.0, size=3)
    p_ccm89, p_od94, p_fitz99 = m/np.sum(m)
    w = npr.uniform(low=1-0.05, high=1+0.05)
    return {"w":w, "p_ccm89": p_ccm89, "p_od94": p_od94, "p_fitz99": p_fitz99}
