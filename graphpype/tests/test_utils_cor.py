import os

import numpy as np
import nibabel as nib

from graphpype.utils_dtype_coord import where_in_coords
from graphpype.utils_cor import (mean_select_mask_data, mean_select_indexed_mask_data,
                                 regress_parameters, filter_data, normalize_data,
                                 return_conf_cor_mat,
                                 return_corres_correl_mat,
                                 where_in_labels, return_corres_correl_mat_labels)

import neuropycon_data as nd

data_path = os.path.join(nd.__path__[0], "data", "data_nii")

print(data_path)

img_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")

mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")

indexed_mask_file = os.path.join(data_path, "Atlas", "indexed_mask-Atlas.nii")


def test_neuropycon_data():
    """
    test if neuropycon_data is installed
    """

    try:
        import neuropycon_data

    except ImportError:
        print("neuropycon_data not installed")

    assert os.path.exists(neuropycon_data.__path__[
                          0]), "warning, could not find path {}, {}".format(neuropycon_data.__path__)

    assert os.path.exists(os.path.join(neuropycon_data.__path__[0], 'data')), "warning, could not find path {}, {}".format(
        os.path.join(neuropycon_data.__path__[0], 'data'), os.listdir(os.path.join(neuropycon_data.__path__[0])))

    assert os.path.exists(os.path.join(neuropycon_data.__path__[0], 'data', 'data_nii')), "warning, could not find path {}, {}".format(
        data_path, os.listdir(os.path.join(neuropycon_data.__path__[0], 'data')))

    assert os.path.exists(img_file), "warning, could not find path {}, {}".format(
        img_file, os.listdir(data_path))


# test selecting signal from ROIs
def test_mean_select_mask_data():
    """
    test reading 4D functional and indexed_mask and extract time series given a percent of voxels having strong enough BOLD signal 
    """

    data_img = nib.load(img_file).get_data()

    print(data_img.shape)

    data_mask = nib.load(mask_file).get_data()

    print(np.unique(data_mask))
    val = mean_select_mask_data(data_img, data_mask)

    print(val.shape)

    assert val.shape[0] == data_img.shape[3], "Error, time series do not have the same length {} != {}".format(
        val.shape[0], data_img.shape[1])


def test_mean_select_indexed_mask_data():
    """
    test reading 4D functional and mask and extract time serie
    """

    data_img = nib.load(img_file).get_data()

    print(data_img.shape)

    data_indexed_mask = nib.load(indexed_mask_file).get_data()

    print(np.unique(data_indexed_mask))
    mean_masked_ts, keep_rois = mean_select_indexed_mask_data(
        data_img, data_indexed_mask)

    print(mean_masked_ts.shape)

    assert mean_masked_ts.shape[1] == data_img.shape[3], "Error, time series do not have the same length {} != {}".format(
        mean_masked_ts.shape[1], data_img.shape[3])

    assert keep_rois.shape[0] == len(np.unique(data_indexed_mask)) - 1, "Error, keep_rois does not have the same length as unique indexes{} != {}".format(
        keep_rois.shape[0],  len(np.unique(data_indexed_mask)) - 1)

# test regressing out signals


time_length = 100
nb_ROI = 10
nb_reg = 5

ts_mat = np.random.rand(nb_ROI, time_length) * \
    np.random.choice([-1, 1], size=(nb_ROI, time_length))
# read in columns, for example from rp files
ts_covar = np.random.rand(time_length, nb_reg)


def test_regress_parameters():
    """
    test regression of covariates to a set of time series
    """

    resid_mat = regress_parameters(ts_mat, ts_covar)
    assert np.all(resid_mat.shape == ts_mat.shape), "Error, residuals shape {} should match original ts shape {}".format(
        resid_mat.shape, ts_mat.shape)


def test_filter_data():
    """
    test filtering time series
    """
    filt_mat = filter_data(ts_mat)
    assert np.all(filt_mat.shape == ts_mat.shape), "Error, filtered ts shape {} should match original ts shape {}".format(
        filt_mat.shape, ts_mat.shape)


def test_normalize_data():
    """
    test normalize time series
    """
    norm_mat = normalize_data(ts_mat)
    assert np.all(norm_mat.shape == ts_mat.shape), "Error, normalized ts shape {} should match original ts shape {}".format(
        norm_mat.shape, ts_mat.shape)

# test correlation matrices


def test_return_conf_cor_mat():
    """
    compute weighted correlation matrices, possibly thresolded given a confidence interval probability
    """
    resid_mat = regress_parameters(ts_mat, ts_covar)
    filt_mat = filter_data(resid_mat)
    norm_mat = normalize_data(filt_mat)

    weight_vect = np.ones(time_length)
    res = return_conf_cor_mat(norm_mat, weight_vect)

    print(res)

# test return_corres_correl_mat with coords and labels


mat = np.random.rand(nb_ROI, nb_ROI)

nb_ref_ROI = nb_ROI + 5

coords = np.random.rand(nb_ROI, 3)

print(coords)

ref_coords = np.concatenate((np.random.rand(5, 3), coords), axis=0)

#np.random.shuffle(ref_coords)

print(ref_coords)

import string


labels = [let for let in string.ascii_letters[:nb_ROI]]
ref_labels = [let for let in string.ascii_letters[:nb_ref_ROI]]

np_labels = np.array(labels, dtype='str')
print(np_labels)

np_ref_labels = np.array(ref_labels, dtype='str')
np.random.shuffle(np_ref_labels)
print(np_ref_labels)

def test_where_in_coords():
    """
    test_where_in_coords
    """
    where_in_corres = where_in_coords(coords,ref_coords)
    
    print(where_in_corres)

def test_return_corres_correl_mat():
    """
    test corres matrix based on coords 
    """
    ref_mat = return_corres_correl_mat(mat, coords, ref_coords)

    print(ref_mat)


def test_where_in_labels():
    """
    test where_in_labels
    """
    where_in_corres = where_in_labels(labels, np_ref_labels.tolist())

    print(where_in_corres)


def test_return_corres_correl_mat_labels():
    """
    test corres matrix based on coords 
    """
    ref_mat = return_corres_correl_mat_labels(mat, labels, np_ref_labels.tolist())

    print(ref_mat)


if __name__ == '__main__':

    ## select ts from mask
    #test_neuropycon_data()
    #test_mean_select_mask_data()  # OK
    #test_mean_select_indexed_mask_data()

    ## regress
    #test_regress_parameters()
    #test_filter_data()
    #test_normalize_data()

    ## cormat
    #test_return_conf_cor_mat()

    # corres
    test_where_in_coords()
    
    #test_return_corres_correl_mat()
    #test_where_in_labels()
    #test_return_corres_correl_mat_labels()
