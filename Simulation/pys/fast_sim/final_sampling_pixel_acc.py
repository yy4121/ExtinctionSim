
"""
Functions here are parts of the simulation pipeline, which samples photo-z only.
For here, assume few items ready:
1. "box", labeled by (EBV_in, Rv, EBV_out)
    Samples(galaxies) in same box has same EBV_in, Rv, and EBV_out.
    one box has 40 shells (real-z).
    each of them has corresponding items:
    (a) one "dust effect" item
    (b) one "mask" item

2. "dust_effect" items, label by (EBV_in, Rv, EBV_out) (same as box)
    Distribution of photot-z for all the samples in the box
    for each "dust effect" item, it is a mixture matrix with size (40, 7)
    dim1: true redshift bin index, can be one of 0,1,....,40
    dim2: photo-z binindex  0: undetected samples(np.nan after adding noise)
                            1,2,3,4,5,6: five photo-z shells
                            7: detected, but has a photo-z outside the photo-z shells, "escaping item"

3. "mask" item, label by (EBV_in, Rv, EBV_out)
    A mask that pick out all healpix pixels with corresponding EBV_in, Rv, and EBV_out.
    Note: it is different with sky mask, which EBV_out<0.2

4. Expected Nount map: shape = (40, 12*nside**2)
    Assume:
        1. An expected total number count of galaxies in full sky(in all redshfit bins and no mask),
            typically 2e8
        2. A distribution of redshift N(z), got directly from CosmoDC2 simulation.
        2. 40 Density map
    Trough calculations, can get expected number count(Ncount) maps, still decimal
    Pixels covered by sky mask has expected Ncount = 0, as a final step
    Through poisson sample, it can convert to sampled map(integers)

The following sub-pipline line has 3 iterations;
interation A, on box:
    interation B, on redshift shells in the box:
        interation C, on unmasked pixel in the redshift shell
"""

import numpy as np
import tqdm

from numba import njit
@njit
def fast_hist2d(x, y, nbins_x, nbins_y):
    hist = np.zeros((nbins_x, nbins_y), dtype=np.int32)
    for i in range(x.shape[0]):
        hist[x[i], y[i]] += 1
    return hist


class EVB_Rv_box:
    """
    Object that do operation in one box only.
    """
    def __init__(self, pixel, dust_effect_on_box,  Sampled_map):
        """
        :param mask: box mask, sum(box_mask)=number of healpix pixel in this box
        :param dust_effect_on_box: distribution of photo_z in this box
        :param Sampled_map: sampled_map(Only masked by the sky mask)
        """
        self.shell_num = np.shape(Sampled_map)[0]
        self.pixel = pixel
        self.dust_effect = dust_effect_on_box
        self.SampledNcount_map_all_z = Sampled_map[:, pixel]
        # Now ths Smapled map only cover the pixel in box, shape = (40, n << 12nside2),
        #                                      where n is the number of pixel in this box
        self.redshift_mix_matrix = self.dust_effect

    def Build_zshell_hist(self, zshell_id):
        """
        Iteration on redshift shells
        :param zshell_id:
        :return:
        """
        Ncount_map_at_z = self.SampledNcount_map_all_z[zshell_id]
        boxpixel_list = np.repeat(np.arange(len(Ncount_map_at_z)), Ncount_map_at_z)
        boxpixel_list = np.array(boxpixel_list)
        photo_z_distribution_at_z = self.redshift_mix_matrix[zshell_id]
        ##### ignore pls
        # photo_z_pool = []
        # for i, count in enumerate(photo_z_distribution_at_z):
        #     photo_z_pool.extend([i] * int(count))
        # photo_z_list = np.random.choice(np.array(photo_z_pool), size=sum( Ncount_map_at_z),replace=True)
        ######
        photo_z_p_at_z = photo_z_distribution_at_z / sum(photo_z_distribution_at_z)
        rng = np.random.default_rng()
        photo_z_list = rng.choice(len(photo_z_distribution_at_z),
                                  size=sum(Ncount_map_at_z), p=photo_z_p_at_z, replace = True)
        photo_z_list = np.array(photo_z_list)
        #######
        #hist_bin_pixel = np.arange(np.sum(self.mask)+1)-0.5
        #hist_bin_photo_z = np.arange(len(photo_z_distribution_at_z) + 1)-0.5
        nbins_x = len(self.pixel)
        nbins_y = len(photo_z_distribution_at_z)
        hist = fast_hist2d(boxpixel_list, photo_z_list, nbins_x ,  nbins_y)
        return hist

    def Build_all_z_hist(self):
        """
        For one box, iteration on reshift
                        iteration on pixel
        :return: an object of size (40, n, 7), where n is the number of pixel in this box
        """
        result = []
        for zshell_id in range(self.shell_num):
            hist = self.Build_zshell_hist(zshell_id)
            result.append(hist)
        return np.array(result)


def all_shell_box(usable_label_all, Rv_EBV_pixel_all, sky_mask, dust_effect_all, ExpNcount_maps):
    """
    :param usable_label_all: a list of (EBV_in, Rv, EBV_out) item, it has 4500 items(boxes) originally,
    but around 500 of them are included here(i.e. usable and has pre-calculated dust effects). Data outside the coverage are considered masked
    :param Rv_EBV_mask_all: mask for all 4500 boxes
    :param sky_mask: SFD < 0.2
    :param dust_effect_all: dust effect for box in usable_label_all
    :param ExpNcount_maps:
    :return:
    A object of size (40, 12nside2, 7), data = number count count
    """
    zdim_in = np.shape(ExpNcount_maps)[0]
    map_size = np.shape(ExpNcount_maps)[1]
    zdim_out = np.shape(dust_effect_all[usable_label_all[0]])[1]
    fill_in = np.zeros((zdim_in, map_size, zdim_out))
    existing_pxixel = []
    Sampled_Ncount = np.random.poisson(ExpNcount_maps) * sky_mask

    for i in tqdm.tqdm(range(len(usable_label_all))):
        # iteration A on box
        label = usable_label_all[i]
        pixel_here = Rv_EBV_pixel_all[label]
        dust_effect_here = dust_effect_all[label]
        true_healpix = pixel_here
        box = EVB_Rv_box(pixel_here, dust_effect_here, Sampled_Ncount)
        # iteration B and C in this box
        hist = box.Build_all_z_hist()
        fill_in[:, true_healpix, :] = hist
        existing_pxixel.extend(list(true_healpix))
    ## make sure no pixel is calculated twice
    has_duplicates = len(existing_pxixel) != len(set(existing_pxixel))
    assert not has_duplicates

    mask = np.zeros(np.shape(fill_in)[1], dtype=int)
    mask[existing_pxixel] = 1

    return fill_in, mask == 1













