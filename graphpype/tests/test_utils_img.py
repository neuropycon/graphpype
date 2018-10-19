import os
import numpy as np
import nibabel as nib

from graphpype.utils_img import (return_data_img_from_roi_mask)

try:
    import neuropycon_data as nd

except ImportError:
    print("neuropycon_data not installed")

data_path = os.path.join(nd.__path__[0], "data", "data_nii")
indexed_mask_file = os.path.join(data_path, "Atlas", "indexed_mask-Atlas.nii")


def test_neuropycon_data():
    """test if neuropycon_data is installed"""
    assert os.path.exists(nd.__path__[0])
    assert os.path.exists(os.path.join(nd.__path__[0], 'data'))
    assert os.path.exists(os.path.join(nd.__path__[0], 'data', 'data_nii'))
    assert os.path.exists(indexed_mask_file)


def test_return_data_img_from_roi_mask():
    """test_return_data_img_from_roi_mask"""
    data_img = nib.load(indexed_mask_file).get_data()
    test_vect = np.random.rand(len(np.unique(data_img))-1)

    data_img_vect = return_data_img_from_roi_mask(indexed_mask_file, test_vect)
    data_img_vect_vals = np.unique(data_img_vect.get_data())[1:]

    assert all(data_img_vect_vals == np.unique(test_vect))
