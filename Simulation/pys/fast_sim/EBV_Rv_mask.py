import numpy as np

def EBV_Rv_submask(EBV_in_map, Rv_map,EBV_out_map, EBV_in_bins, Rv_bins, EBV_out_bins, sky_mask, check = False, cut = 200):
    """apply sky map and generate mask that can generate Ncount map into EBV and Rv layers"""

    assert EBV_in_map.shape == Rv_map.shape == sky_mask.shape, "Input maps must have the same shape"

    # Force float to avoid integer wrap or truncation

    # Get quantile-based bin edges

    # Expand bin edges slightly to avoid exclusion from floating point rounding
    EBV_in_bins[0] -= 1e-8
    EBV_in_bins[-1] += 1e-8
    Rv_bins[0] -= 1e-8
    Rv_bins[-1] += 1e-8
    EBV_out_bins[0] -= 1e-8
    EBV_out_bins[-1] += 1e-8

    EBV_in_num = len(EBV_in_bins)-1
    Rv_num = len(Rv_bins)-1
    EBV_out_num = len(EBV_out_bins) - 1

    # Digitize and clip to valid bin range
    EBV_in_idx = np.digitize(EBV_in_map, EBV_in_bins, right=True) - 1
    Rv_idx = np.digitize(Rv_map, Rv_bins, right=True) - 1
    EBV_out_idx = np.digitize(EBV_out_map, EBV_out_bins, right=True) - 1

    EBV_in_idx = np.clip(EBV_in_idx, 0, EBV_in_num - 1)
    Rv_idx = np.clip(Rv_idx, 0, Rv_num - 1)
    EBV_out_idx = np.clip(EBV_out_idx, 0, EBV_out_num - 1)

    # Initialize subshells
    subshell = []
    label = []

    for i in range(EBV_in_num):
        for j in range(Rv_num):
            for k in range(EBV_out_num):
                mask = (EBV_in_idx == i) & (Rv_idx == j)&(EBV_out_idx ==k) & (sky_mask)
                subshell.append(mask)
                label.append([i, j, k])

    # Stack and return
    subshell = np.array(subshell)
    label = np.array(label)
    if check:
        assert np.array_equal(np.sum(subshell, axis = 0), sky_mask)

    return subshell, label, {
        "bin_EBV_dim0": EBV_in_bins,
        "bin_Rv_dim1": Rv_bins,
        "bin_EBV_dim2": EBV_out_bins,
    }

def ignore_mask_under_count(masks, cut):
    return_id = []
    lost_EBV_Rv_sample = 0
    lost_map = np.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        if np.sum(mask)<cut:
            lost_EBV_Rv_sample += 1
            lost_map += mask
        else:
            return_id.append(i)
    return return_id, lost_map, np.sum(lost_map)/len(mask)


def compile_usable(masks, label, usable_index):
    usable_label = label[usable_index]
    usable_mask = masks[usable_index]
    EBV_Rv = {}
    for i, l in enumerate(usable_label):
        EBV_Rv[tuple(l)] = usable_mask[i]
    return EBV_Rv


import healpy as hp
def retrived_EBV_Rvmap(compiled_mask, EBV_in_bins, Rv_bins, EBV_out_bins):
    map_size = len(compiled_mask[list(compiled_mask.keys())[0]])
    EBV_in_map = np.zeros(map_size, dtype=np.float32)
    EBV_out_map = np.zeros_like(EBV_in_map)
    Rv_map = np.zeros_like(EBV_in_map)
    mask_total = np.zeros_like(EBV_in_map)
    for keys in list(compiled_mask.keys()):
        mask = compiled_mask[keys] == True
        #hp.mollview(mask)
        EBV_in_index = keys[0]
        Rv_index = keys[1]
        EBV_out_index = keys[2]

        EBV_in_value = (EBV_in_bins[EBV_in_index] + EBV_in_bins[EBV_in_index+1])/2
        Rv_value = (Rv_bins[Rv_index] + Rv_bins[Rv_index + 1]) / 2
        EBV_out_value = (EBV_out_bins[EBV_out_index] + EBV_out_bins[EBV_out_index + 1]) / 2

        EBV_in_map[mask] = EBV_in_value
        Rv_map[mask] = Rv_value
        EBV_out_map[mask] = EBV_out_value
        mask_total[mask] = 1
    return EBV_in_map, Rv_map, EBV_out_map, mask_total==1



