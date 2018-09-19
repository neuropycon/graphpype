"""
Support function for correl_mat.py mostly, some for gather_cormats
"""
from scipy import stats
import numpy as np

import pandas as pd

import statsmodels.formula.api as smf
import itertools as it
import scipy.signal as filt

from .utils import check_np_shapes
from .utils_dtype_coord import where_in_coords


def mean_select_mask_data(data_img, data_mask):
    """
    extrating ts by averaging the time series of all voxels with the same
    index
    """
    assert len(data_img.shape) == 4, \
        ("Error, data_img should be a 4Dfile, shape is {}".format(
            data_img.shape))

    assert check_np_shapes(data_img.shape[:3], data_mask.shape), \
        ("Error, Image and mask are incompatible {} {}".format(
            data_img.shape[:3], data_mask.shape))

    masked_data_matrix = data_img[data_mask == 1, :]

    try:
        mean_mask_data_matrix = np.nanmean(masked_data_matrix, axis=0)

    except AttributeError:

        print("no nanmean (version of numpy is too old), using mean only")
        mean_mask_data_matrix = np.mean(masked_data_matrix, axis=0)

    return mean_mask_data_matrix


def mean_select_indexed_mask_data(data_img, data_indexed_mask,
                                  min_BOLD_intensity=50, percent_signal=0.5,
                                  background_val=-1.0):
    """
    extrating ts by averaging the time series of all voxels with the same
    index
    """
    assert len(data_img.shape) == 4, \
        ("Error, data_img should be a 4Dfile, shape is {}".format(
            data_img.shape))

    assert check_np_shapes(data_img.shape[:3], data_indexed_mask.shape), \
        ("Error, Image and mask are incompatible {} {}".format(
            data_img.shape[:3], data_indexed_mask.shape))

    # sequence_roi_index
    sequence_roi_index = np.unique(data_indexed_mask)

    if sequence_roi_index[0] == background_val:
        sequence_roi_index = sequence_roi_index[1:]

    # mean_masked_ts
    mean_masked_ts = []
    keep_rois = np.zeros(shape=(sequence_roi_index.shape[0]), dtype=bool)

    for i, roi_index in enumerate(sequence_roi_index):

        roi_x, roi_y, roi_z = np.where(data_indexed_mask == roi_index)

        all_voxel_roi_ts = data_img[roi_x, roi_y, roi_z, :]
        nb_voxels, nb_volumes = all_voxel_roi_ts.shape

        # testing if at least 50% of the voxels in the ROIs have values
        # always higher than min bold intensity
        nb_signal_voxels = np.sum(np.sum(
            all_voxel_roi_ts > min_BOLD_intensity, axis=1) == nb_volumes)

        percent_voxel_signal = nb_signal_voxels/float(nb_voxels)

        if percent_voxel_signal > percent_signal:
            keep_rois[i] = True

            try:
                mean_all_voxel_roi_ts = np.nanmean(all_voxel_roi_ts, axis=0)

            except AttributeError:
                print("Warning, no nanmean (version of numpy is too old),\
                      using mean only")
                mean_all_voxel_roi_ts = np.mean(all_voxel_roi_ts, axis=0)

            mean_masked_ts.append(mean_all_voxel_roi_ts)
        else:
            print("ROI {} was not selected : {} {} ".format(
                roi_index, nb_signal_voxels, round(percent_voxel_signal, 2)))

    assert len(mean_masked_ts) != 0, "min_BOLD_intensity {} and \
        percent_signal {} are to restrictive".format(min_BOLD_intensity,
                                                     percent_signal)

    mean_masked_ts = np.array(mean_masked_ts, dtype='f')
    return mean_masked_ts, keep_rois


def regress_parameters(data_matrix, covariates):
    """covariate regression"""

    # formatting dataframe
    if data_matrix.shape[1] == covariates.shape[0]:
        print("Transposing data_matrix shape {} -> {}".format(
            data_matrix.shape, np.transpose(data_matrix).shape))
        data_matrix = np.transpose(data_matrix)

    data_names = ['Var_'+str(i) for i in range(data_matrix.shape[1])]
    covar_names = ['Cov_'+str(i) for i in range(covariates.shape[1])]

    df = pd.DataFrame(np.concatenate((data_matrix, covariates), axis=1),
                      columns=data_names + covar_names)

    # computing regression
    resid_data = [smf.ols(formula=var+" ~ "+" + ".join(covar_names),
                          data=df).fit().resid.values for var in data_names]

    resid_data_matrix = np.array(resid_data, dtype=float)

    return resid_data_matrix


def filter_data(data_matrix, N=5, Wn=0.04, btype='highpass'):
    """filter_data"""
    b, a = filt.iirfilter(N=N, Wn=Wn, btype=btype)
    return filt.filtfilt(b, a, x=data_matrix, axis=1)


def normalize_data(data_matrix):
    """normalize_data"""
    z_score_data_matrix = stats.zscore(data_matrix, axis=1)

    return z_score_data_matrix


def return_conf_cor_mat(ts_mat, weight_vect, conf_interval_prob=0.01):

    """
    Compute correlation matrices over a time series and weight vector,
    either Rsquared-value matrix (cor_mat),
    or Z (after R-to-Z values) matrix (Z_cor_mat)
    It also return possibly thresholded matrices (conf_cor_mat and
    Z_conf_cor_mat) according to a confidence interval probability
    conf_interval_prob
    """
    if ts_mat.shape[1] == len(weight_vect):
        print("Transposing data_matrix shape {} -> {}".format(
            ts_mat.shape, np.transpose(ts_mat).shape))
        ts_mat = np.transpose(ts_mat)

    assert ts_mat.shape[0] == len(weight_vect), \
        ("Error, incompatible regressor length {} {}".format(ts_mat.shape[0],
                                                             len(weight_vect)))

    keep = weight_vect > 0.0
    w = weight_vect[keep]
    ts_mat = ts_mat[keep, :]

    # confidence interval for variance computation
    norm = stats.norm.ppf(1-conf_interval_prob/2)
    deg_freedom = w.sum()/w.max()-3

    s, n = ts_mat.shape

    Z_cor_mat = np.zeros((n, n), dtype=float)
    Z_conf_cor_mat = np.zeros((n, n), dtype=float)
    cor_mat = np.zeros((n, n), dtype=float)
    conf_cor_mat = np.zeros((n, n), dtype=float)

    ts_mat2 = ts_mat*np.sqrt(w)[:, np.newaxis]

    for i, j in it.combinations(list(range(n)), 2):

        keep_val = ~(np.isnan(ts_mat2[:, i]) | np.isnan(ts_mat2[:, j]))

        s1 = ts_mat2[keep_val, i]
        s2 = ts_mat2[keep_val, j]

        cor_mat[i, j] = (s1*s2).sum()/np.sqrt((s1*s1).sum() * (s2*s2).sum())
        Z_cor_mat[i, j] = np.arctanh(cor_mat[i, j])

        assert not np.isnan(Z_cor_mat[i, j]), \
            ("Error Z_cor_mat {}{} should not be NAN value".format(i, j))

        assert not np.isinf(Z_cor_mat[i, j]), \
            ("Error Z_cor_mat {}{} should not be infinite value".format(i, j))

    pos_Z = (np.sign(Z_cor_mat) == +1.0)
    neg_Z = (np.sign(Z_cor_mat) == -1.0)
    signif_pos = (Z_cor_mat > norm/np.sqrt(deg_freedom)) & pos_Z
    signif_neg = (Z_cor_mat < -norm/np.sqrt(deg_freedom)) & neg_Z

    Z_conf_cor_mat[signif_pos] = Z_cor_mat[signif_pos]
    Z_conf_cor_mat[signif_neg] = Z_cor_mat[signif_neg]

    conf_cor_mat[signif_pos] = cor_mat[signif_pos]
    conf_cor_mat[signif_neg] = cor_mat[signif_neg]

    return cor_mat, Z_cor_mat, conf_cor_mat, Z_conf_cor_mat


def where_in_labels(labels, corres_labels):
    """find indexes of labels in corres_labels"""
    label_indexes = [corres_labels.index(lab) for lab in labels]
    return np.array(label_indexes, dtype='int64')


def return_corres_correl_mat(mat, coords, corres_coords):
    """computing corres matrix using reference (corres_coords) and coords"""
    assert mat.shape[0] == mat.shape[1], \
        ("Error, matrix should be square {}".format(mat.shape))
    assert mat.shape[0] == coords.shape[0], ("Error, matrix {} and coords {} \
        should have the same length".format(mat.shape[0], coords.shape[1]))

    where_in_corres = where_in_coords(coords, corres_coords)

    print(where_in_corres)
    corres_size = corres_coords.shape[0]

    print(np.min(where_in_corres), np.max(where_in_corres),
          where_in_corres.shape)

    corres_mat = np.zeros((corres_size, corres_size), dtype=float)
    possible_edge_mat = np.zeros((corres_size, corres_size), dtype=int)

    print(corres_mat.shape)

    for i, j in it.combinations(range(coords.shape[0]), 2):

        corres_mat[where_in_corres[i], where_in_corres[j]] = mat[i, j]
        corres_mat[where_in_corres[j], where_in_corres[i]] = mat[i, j]

        possible_edge_mat[where_in_corres[i], where_in_corres[j]] = 1
        possible_edge_mat[where_in_corres[j], where_in_corres[i]] = 1

    return corres_mat, possible_edge_mat


def return_corres_correl_mat_labels(mat, labels, corres_labels):
    """computing corres matrix using reference (corres_labels) and labels"""
    assert isinstance(labels, list), "Error, labels should be a list"
    assert isinstance(corres_labels, list), \
        ("corres_labels, labels should be a list")

    where_in_corres = where_in_labels(labels, corres_labels)

    corres_size = len(corres_labels)

    print(np.min(where_in_corres), np.max(where_in_corres),
          where_in_corres.shape)

    corres_mat = np.zeros((corres_size, corres_size), dtype=float)
    possible_edge_mat = np.zeros((corres_size, corres_size), dtype=int)

    for i, j in it.product(range(len(labels)), repeat=2):
        corres_mat[where_in_corres[i], where_in_corres[j]] = mat[i, j]
        possible_edge_mat[where_in_corres[i], where_in_corres[j]] = 1

    print(corres_mat.shape)

    return corres_mat, possible_edge_mat
