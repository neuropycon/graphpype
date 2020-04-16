import os
import string
import numpy as np
import nibabel as nib

from graphpype.utils_cor import (mean_select_mask_data,
                                 mean_select_indexed_mask_data,
                                 regress_parameters, return_conf_cor_mat,
                                 filter_data, normalize_data,
                                 return_corres_correl_mat,
                                 where_in_labels,
                                 return_corres_correl_mat_labels,
                                 spearmanr_by_hand)

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_nii_min")

img_file = os.path.join(data_path, "wrsub-01_task-rest_bold.nii")
mask_file = os.path.join(data_path, "rwc1sub-01_T1w.nii")

data_path_HCP = load_test_data("data_nii_HCP")
indexed_mask_file = os.path.join(data_path_HCP, "indexed_mask-ROI_HCP.nii")


def test_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    assert os.path.exists(img_file)
    assert os.path.exists(mask_file)

    assert os.path.exists(data_path_HCP)
    assert os.path.exists(indexed_mask_file)


# test selecting signal from ROIs
def test_mean_select_mask_data():
    """
    test reading 4D functional and indexed_mask and extract time series
    given a percent of voxels having strong enough BOLD signal
    """
    data_img = nib.load(img_file).get_data()
    data_mask = nib.load(mask_file).get_data()
    val = mean_select_mask_data(data_img, data_mask)

    assert val.shape[0] == data_img.shape[3]


def test_mean_select_indexed_mask_data():
    """test reading 4D functional and mask and extract time serie"""
    data_img = nib.load(img_file).get_data()
    data_indexed_mask = nib.load(indexed_mask_file).get_data()
    mean_masked_ts, keep_rois = mean_select_indexed_mask_data(
        data_img, data_indexed_mask, background_val=0.0)

    assert mean_masked_ts.shape[1] == data_img.shape[3]
    assert keep_rois.shape[0] == len(np.unique(data_indexed_mask))-1


###############################################################################
# test regressing out signals
time_length = 100
nb_ROI = 10
nb_reg = 5

ts_mat = np.random.rand(nb_ROI, time_length) * \
    np.random.choice([-1, 1], size=(nb_ROI, time_length))
# read in columns, for example from rp files
ts_covar = np.random.rand(time_length, nb_reg)


def test_regress_parameters():
    """test regress_parameters"""
    reg_mat = regress_parameters(ts_mat, ts_covar)
    assert np.all(reg_mat.shape == ts_mat.shape)


def test_filter_data():
    """test filtering time series"""
    filt_mat = filter_data(ts_mat)
    assert np.all(filt_mat.shape == ts_mat.shape)


def test_normalize_data():
    """test normalize time series"""
    norm_mat = normalize_data(ts_mat)
    assert np.all(norm_mat.shape == ts_mat.shape)


def test_return_conf_cor_mat():
    """compute weighted correlation matrices"""
    # TODO find a real assert test...
    resid_mat = regress_parameters(ts_mat, ts_covar)
    filt_mat = filter_data(resid_mat)
    norm_mat = normalize_data(filt_mat)
    weight_vect = np.ones(time_length)
    res = return_conf_cor_mat(norm_mat, weight_vect)
    print(res)


def test_spearmanr_by_hand():
    """test corres matrix based on coords"""
    # TODO find a real assert test...
    """compute weighted correlation matrices"""
    # TODO find a real assert test...
    resid_mat = regress_parameters(ts_mat, ts_covar)
    filt_mat = filter_data(resid_mat)
    norm_mat = normalize_data(filt_mat)

    rho_mat, pval_mat = spearmanr_by_hand(norm_mat)
    print(rho_mat, pval_mat)


###############################################################################
mat = np.random.rand(nb_ROI, nb_ROI)
nb_ref_ROI = nb_ROI + 5
coords = np.random.randint(low=-70, high=70, size=(nb_ROI, 3))

ref_coords = np.concatenate(
    (np.random.randint(low=-70, high=70, size=(5, 3)), coords), axis=0)

np.random.shuffle(ref_coords)

labels = [let for let in string.ascii_letters[:nb_ROI]]
ref_labels = [let for let in string.ascii_letters[:nb_ref_ROI]]
np_labels = np.array(labels, dtype='str')
np_ref_labels = np.array(ref_labels, dtype='str')
np.random.shuffle(np_ref_labels)


def test_return_corres_correl_mat():
    """test corres matrix based on coords"""
    # TODO find a real assert test...
    ref_mat = return_corres_correl_mat(mat, coords, ref_coords)
    print(ref_mat)


def test_where_in_labels():
    """test where_in_labels"""
    # TODO find a real assert test...
    where_in_corres = where_in_labels(labels, np_ref_labels.tolist())
    print(where_in_corres)


def test_return_corres_correl_mat_labels():
    """test corres matrix based on coords"""
    # TODO find a real assert test...
    ref_mat = return_corres_correl_mat_labels(
        mat, labels, np_ref_labels.tolist())
    print(ref_mat)
