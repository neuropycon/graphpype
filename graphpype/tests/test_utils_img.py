import os
# import numpy as np
# import nibabel as nib

# from graphpype.utils_img import (return_data_img_from_roi_mask)
from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_nii_HCP")
indexed_mask_file = os.path.join(data_path, "indexed_mask-ROI_HCP.nii")


def test_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    # assert os.path.exists(indexed_mask_file)


# def test_return_data_img_from_roi_mask():
    # """test_return_data_img_from_roi_mask"""
    # data_img = nib.load(indexed_mask_file).get_data()
    # test_vect = np.random.rand(len(np.unique(data_img))-1)

    # data_img_vect = return_data_img_from_roi_mask(indexed_mask_file,
    #                                               test_vect)
    # data_img_vect_vals = np.unique(data_img_vect.get_data())[1:]

    # assert all(data_img_vect_vals == np.unique(test_vect))
