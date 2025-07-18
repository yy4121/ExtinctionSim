import numpy as np

def EBV_Rv_subpix(
    EBV_in_map, Rv_map, EBV_out_map,
    EBV_in_bins, Rv_bins, EBV_out_bins,
    sky_mask, check=False
):
    eps = 1e-8
    EBV_in_bins[0]  -= eps
    EBV_in_bins[-1] += eps
    Rv_bins[0]      -= eps
    Rv_bins[-1]     += eps
    EBV_out_bins[0] -= eps
    EBV_out_bins[-1]+= eps

    # Digitize → bin indices
    EBV_in_idx  = np.clip(np.digitize(EBV_in_map,  EBV_in_bins,  right=True) - 1, 0, len(EBV_in_bins)  - 2)
    Rv_idx      = np.clip(np.digitize(Rv_map,      Rv_bins,      right=True) - 1, 0, len(Rv_bins)      - 2)
    EBV_out_idx = np.clip(np.digitize(EBV_out_map, EBV_out_bins, right=True) - 1, 0, len(EBV_out_bins) - 2)

    # Flatten everything
    valid = sky_mask.flatten()
    EBV_in_idx  = EBV_in_idx.flatten()
    Rv_idx      = Rv_idx.flatten()
    EBV_out_idx = EBV_out_idx.flatten()

    # Bin counts
    total_bins = len(EBV_in_bins) - 1
    total_rv   = len(Rv_bins) - 1
    total_out  = len(EBV_out_bins) - 1
    num_labels = total_bins * total_rv * total_out

    # Single bin ID per pixel
    flat_idx = (EBV_in_idx * total_rv + Rv_idx) * total_out + EBV_out_idx

    # Mask invalid pixels
    pix_ids = np.arange(flat_idx.size)[valid]
    bin_ids = flat_idx[valid]

    # Sort by bin → then split efficiently
    sort_order = np.argsort(bin_ids)
    bin_ids_sorted = bin_ids[sort_order]
    pix_ids_sorted = pix_ids[sort_order]

    # Where bin changes → split indices
    unique_bins, split_idx = np.unique(bin_ids_sorted, return_index=True)

    # Create result list: for each bin index, which pixels belong to it
    pixel_lists = [[] for _ in range(num_labels)]
    for bin_id, start, end in zip(unique_bins, split_idx, list(split_idx[1:]) + [len(bin_ids_sorted)]):
        pixel_lists[bin_id] = pix_ids_sorted[start:end].tolist()

    # Label triple (i,j,k)
    label = np.array([[i, j, k]
                      for i in range(total_bins)
                      for j in range(total_rv)
                      for k in range(total_out)], dtype=int)

    # Optional check: every valid pixel appears once
    if check:
        all_pix_concat = np.concatenate([np.array(lst) for lst in pixel_lists if lst])
        assert np.array_equal(np.sort(all_pix_concat), np.nonzero(valid)[0])

    return pixel_lists, label, {
        "bin_EBV_dim0": EBV_in_bins,
        "bin_Rv_dim1":  Rv_bins,
        "bin_EBV_dim2": EBV_out_bins,
    }




def ignore_mask_under_count(sub_index, cut):
    counts = np.array([len(i) for i in sub_index])
    keep = counts >= cut
    selected_indices = np.where(keep)[0].tolist()
    frac_lost = 1-np.sum(counts[selected_indices]) / np.sum(counts)
    return selected_indices, frac_lost


def select_mask_of_toplarge_count(sub_index, num):
    counts = np.array([len(i) for i in sub_index])
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


def select_top_until_fraction(sub_index, fraction):
    counts = np.array([len(i) for i in sub_index])
    selected_indices = top_until_fraction_indices(counts, fraction)
    frac_lost = 1-np.sum(counts[selected_indices]) / np.sum(counts)
    return selected_indices, frac_lost






def compile_usable(indices, label, usable_index):
    usable_label = label[usable_index]
    usable_mask = [indices[i] for i in usable_index]
    EBV_Rv = {}
    for i, l in enumerate(usable_label):
        EBV_Rv[tuple(l)] = usable_mask[i]
    return EBV_Rv

