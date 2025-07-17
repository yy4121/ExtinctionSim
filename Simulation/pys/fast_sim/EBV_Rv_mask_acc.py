import numpy as np

def EBV_Rv_submask(EBV_in_map, Rv_map, EBV_out_map,
                        EBV_in_bins, Rv_bins, EBV_out_bins,
                        sky_mask, check=False):

    EBV_in_bins[0] -= 1e-8
    EBV_in_bins[-1] += 1e-8
    Rv_bins[0] -= 1e-8
    Rv_bins[-1] += 1e-8
    EBV_out_bins[0] -= 1e-8
    EBV_out_bins[-1] += 1e-8

    EBV_in_idx = np.clip(np.digitize(EBV_in_map, EBV_in_bins, right=True) - 1, 0, len(EBV_in_bins) - 2)
    Rv_idx = np.clip(np.digitize(Rv_map, Rv_bins, right=True) - 1, 0, len(Rv_bins) - 2)
    EBV_out_idx = np.clip(np.digitize(EBV_out_map, EBV_out_bins, right=True) - 1, 0, len(EBV_out_bins) - 2)

    flat_idx = (EBV_in_idx * len(Rv_bins[:-1]) + Rv_idx) * len(EBV_out_bins[:-1]) + EBV_out_idx
    total_bins = len(EBV_in_bins) - 1
    total_rv = len(Rv_bins) - 1
    total_out = len(EBV_out_bins) - 1
    num_labels = total_bins * total_rv * total_out

    # For each bin index, get a mask
    subshell = np.zeros((num_labels, EBV_in_map.size), dtype=bool)
    valid = sky_mask.flatten()

    for idx in range(num_labels):
        subshell[idx, :] = (flat_idx == idx) & valid

    # Generate labels
    label = np.array([
        [i, j, k]
        for i in range(total_bins)
        for j in range(total_rv)
        for k in range(total_out)
    ])

    if check:
        assert np.array_equal(np.sum(subshell, axis=0), sky_mask.flatten())

    return subshell, label, {
        "bin_EBV_dim0": EBV_in_bins,
        "bin_Rv_dim1": Rv_bins,
        "bin_EBV_dim2": EBV_out_bins,
    }


import numpy as np

import numpy as np

def EBV_Rv_submask_fast(
    EBV_in_map, Rv_map, EBV_out_map,
    EBV_in_bins, Rv_bins, EBV_out_bins,
    sky_mask, check=False
):
    # Slightly expand bin edges to avoid boundary issues
    EBV_in_bins = EBV_in_bins.copy()
    Rv_bins = Rv_bins.copy()
    EBV_out_bins = EBV_out_bins.copy()
    for bins in (EBV_in_bins, Rv_bins, EBV_out_bins):
        bins[0] -= 1e-8
        bins[-1] += 1e-8

    # Digitize the maps into bin indices
    EBV_in_idx = np.clip(
        np.digitize(EBV_in_map, EBV_in_bins, right=True) - 1,
        0, len(EBV_in_bins) - 2
    )
    Rv_idx = np.clip(
        np.digitize(Rv_map, Rv_bins, right=True) - 1,
        0, len(Rv_bins) - 2
    )
    EBV_out_idx = np.clip(
        np.digitize(EBV_out_map, EBV_out_bins, right=True) - 1,
        0, len(EBV_out_bins) - 2
    )

    # Bin dimensions
    total_bins = len(EBV_in_bins) - 1
    total_rv   = len(Rv_bins) - 1
    total_out  = len(EBV_out_bins) - 1
    num_labels = total_bins * total_rv * total_out

    # Flatten and compute a unique ID for each pixel's bin combination
    flat_idx = (
        (EBV_in_idx * total_rv) + Rv_idx
    ) * total_out + EBV_out_idx

    # Only keep valid pixels
    valid_mask = sky_mask.flatten()
    valid_idx  = flat_idx.flatten()[valid_mask]

    # Create a subshell mask only for valid pixels
    unique_ids = np.arange(num_labels)
    # subshell_valid shape: (num_labels, n_valid_pixels)
    subshell_valid = (valid_idx[None, :] == unique_ids[:, None])

    # Rebuild full-sized subshell (num_labels x Npix), fill invalid with False
    Npix = flat_idx.size
    subshell = np.zeros((num_labels, Npix), dtype=bool)
    subshell[:, valid_mask] = subshell_valid

    # Generate label array of shape (num_labels, 3)
    label = np.array(np.meshgrid(
        np.arange(total_bins),
        np.arange(total_rv),
        np.arange(total_out),
        indexing="ij"
    )).reshape(3, -1).T

    # Optional consistency check
    if check:
        assigned_counts = np.sum(subshell, axis=0)
        assert np.array_equal(assigned_counts[valid_mask], np.ones_like(valid_idx))

    return subshell, label, {
        "bin_EBV_dim0": EBV_in_bins,
        "bin_Rv_dim1":  Rv_bins,
        "bin_EBV_dim2": EBV_out_bins,
    }



def ignore_mask_under_count(masks, cut):
    counts = np.sum(masks, axis=1)
    keep = counts >= cut
    selected_indices = np.where(keep)[0].tolist()
    frac_lost = 1-np.sum(counts[selected_indices]) / np.sum(counts)
    return selected_indices, frac_lost


def select_mask_of_toplarge_count(masks, num):
    counts = np.sum(masks, axis=1)
    selected_indices = np.argsort(counts)[-num:][::-1]  # top `num`
    frac_lost = 1-np.sum(counts[selected_indices]) / np.sum(counts)
    return selected_indices, frac_lost

def top_until_fraction_indices(arr, fraction):
    # Sort descending
    sorted_indices = np.argsort(arr)[::-1]
    sorted_vals = arr[sorted_indices]
    total_sum = sorted_vals.sum()
    if total_sum == 0:
        return np.array([], dtype=int)
    cum_sum = np.cumsum(sorted_vals)
    target_sum = fraction * total_sum
    cutoff_idx = np.searchsorted(cum_sum, target_sum)

    return sorted_indices[:cutoff_idx + 1]


def select_top_until_fraction(masks, fraction):
    counts = np.sum(masks, axis=1)
    selected_indices = top_until_fraction_indices(counts, fraction)
    frac_lost = 1-np.sum(counts[selected_indices]) / np.sum(counts)
    return selected_indices, frac_lost






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



