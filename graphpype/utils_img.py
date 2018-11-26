"""
not sure what this is doing, may be merged to labelled mask?
"""
# TODO test_utils_img
import numpy as np
import nibabel as nib


def return_data_img_from_roi_mask(roi_mask_file, data_vect):
    """ create a Nifti file from a numpy array with one value for each roi"""
    data_vect = np.array(data_vect)

    roi_mask = nib.load(roi_mask_file)
    roi_mask_data = roi_mask.get_data()

    # skipping background and transforming to integer indexes
    unique_vals = np.unique(roi_mask_data)[1:].astype(int)

    err_msg = "warning, ROI roi_mask not compatible with data_vect"

    assert np.all(np.arange(data_vect.shape[0]) == unique_vals), err_msg

    data = np.zeros((roi_mask.shape), dtype=data_vect.dtype) - 1

    for roi_index in unique_vals:
        data[roi_mask_data == roi_index] = data_vect[roi_index]

    data_img = nib.Nifti1Image(data, roi_mask.affine, roi_mask.header)
    return data_img
