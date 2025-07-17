from astropy.io import fits
import numpy as np
import healpy as hp
from astropy import units as u
from scipy.interpolate import interp1d
import numpy.random as rm

def EBV_true(map1, map2, p_EBV):
    N1 = p_EBV["EBV_N1"]
    N2 = p_EBV["EBV_N1"]
    mix = p_EBV["EBV_mix"]
    return map1*N1*(mix) + map2*N2*(1-mix)


def Rv_map(EBV_map, p_Rv):
    mask_high = EBV_map > 0.4
    mask_low = EBV_map < 0.4
    a = p_Rv["Rv_a"]
    b = p_Rv["Rv_b"]
    sig = p_Rv["Rv_sig"]
    Rv_map_high = 3.1 * np.ones_like(EBV_map)
    Rv_map_low = a*EBV_map + b
    noise = rm.normal(loc=0, scale=sig, size=len(EBV_map))
    Rv_map = Rv_map_low*mask_low + Rv_map_high*mask_high + noise
    Rv_map[Rv_map < 2] = 2
    Rv_map[Rv_map > 5] = 5
    return Rv_map



class dust_law:
    def __init__(self, p_dust_law):
        self.para_infrared_a = p_dust_law["infrared_a"]
        self.para_infrared_b = p_dust_law["infrared_b"]
        self.para_optical_a = p_dust_law["optical_a"]
        self.para_optical_b = p_dust_law["optical_b"]

    def A_Av(self, lambda_, Rv=3.1):
        assert type(lambda_) == u.quantity.Quantity, "wavelength does not has unit, times unit such as u.nm"
        x = (1/lambda_).to(u.um**-1)
        assert min(x.value)>=0.3 and max(x.value)<= 3.3, "wavelength outside the range, min(x) = {}, max(x) = {}".format(min(x.value), max(x.value))
        y = x.value-1.82
        mask_infrared = x.value<1.1
        mask_optical = x.value>1.1
        infrared_a = np.array([np.sum(self.para_infrared_a * x_i.value**1.61) for x_i in x]) * mask_infrared
        infrared_b =  np.array([np.sum(self.para_infrared_b * x_i.value**1.61)  for x_i in x]) * mask_infrared

        order = np.arange(8)

        optical_a = np.array([np.sum(self.para_optical_a * y_i**order) for y_i in y]) * mask_optical
        optical_b = np.array([np.sum(self.para_optical_b * y_i**order) for y_i in y]) * mask_optical
        a = infrared_a + optical_a
        b = infrared_b + optical_b
        return x, a + b/Rv

    def band_coefficient_FixedRv(self, filter_curve, SED_wl, SED, EBV_model=0.4, Rv =3.1):
        """Return band coefficient for ugrizy, considering Rv = 3.1"""
        result = {}
        SED_interu = interp1d(SED_wl, SED, kind='linear', fill_value=0., bounds_error=False)
        for band in "ugrizy":
            filter_wl = filter_curve[band]["wl"]
            filter_trans = filter_curve[band]["F_trans"]
            assert SED_wl.unit == filter_wl.unit
            flux_0 = SED_interu(filter_wl) * filter_trans
            _, Awl_Av = self.A_Av(filter_wl, Rv = Rv)
            Av = Rv * EBV_model
            Awl = Awl_Av * Av
            ratio_extinction = 10**(-Awl/2.5)
            flux_extincted = flux_0 * ratio_extinction
            f = np.sum(flux_extincted)/np.sum(flux_0)
            A_band = -2.5*np.log10(f)
            band_coefficient = A_band/EBV_model
            result[band] = band_coefficient
        return result


    def band_coefficient_Rv(self, filter_curve, SED_wl, SED, EBV_model=0.4, Rv_range = (1, 6)):
        """Return band coefficient for ugrizy, considering Rv = 3.1"""
        Rv_line = np.linspace(min(Rv_range), max(Rv_range), 20)
        band_co_list= []
        for Rv in Rv_line:
            band_co = self.band_coefficient_FixedRv(filter_curve, SED_wl, SED,EBV_model,Rv)
            band_co_list.append([band_co[key] for key in "ugrizy"])
        band_co_list = np.array(band_co_list).T
        result = {}
        for i, band in enumerate(band_co_list):
            interp = interp1d(Rv_line, band, kind='linear')
            result["ugrizy"[i]] = interp

        return result


import extinction as ext

class dust_law_repara:
    def __init__(self, para):
        self.p_ccm89 = para["p_ccm89"]
        self.p_od94 = para["p_od94"]
        self.p_fitz99 = para["p_fitz99"]
        self.w = para["w"]

    def A_Av(self, lambda_, Rv=3.1):
        assert type(lambda_) == u.quantity.Quantity, "wavelength does not has unit, times unit such as u.nm"
        x = lambda_.to(u.AA)
        ccm89_part = ext.ccm89(x, 1.0, Rv)
        od94_part = ext.odonnell94(x, 1.0, Rv)
        fitz99_part = ext.fitzpatrick99(x, 1.0, Rv)
        return x, (self.p_ccm89*ccm89_part + self.p_od94*od94_part + self.p_fitz99*fitz99_part)*self.w

    def band_coefficient_FixedRv(self, filter_curve, SED_wl, SED, EBV_model=0.4, Rv =3.1):
        """Return band coefficient for ugrizy, considering Rv = 3.1"""
        result = {}
        SED_interu = interp1d(SED_wl, SED, kind='linear', fill_value=0., bounds_error=False)
        for band in "ugrizy":
            filter_wl = filter_curve[band]["wl"]
            filter_trans = filter_curve[band]["F_trans"]
            assert SED_wl.unit == filter_wl.unit
            flux_0 = SED_interu(filter_wl) * filter_trans
            _, Awl_Av = self.A_Av(filter_wl, Rv = Rv)
            Av = Rv * EBV_model
            Awl = Awl_Av * Av
            ratio_extinction = 10**(-Awl/2.5)
            flux_extincted = flux_0 * ratio_extinction
            f = np.sum(flux_extincted)/np.sum(flux_0)
            A_band = -2.5*np.log10(f)
            band_coefficient = A_band/EBV_model
            result[band] = band_coefficient
        return result
