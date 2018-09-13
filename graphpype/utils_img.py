
from define_variables import *


def return_data_img_from_roi_mask(roi_mask_file, data_vect):

    data_vect = np.array(data_vect)

    roi_mask = nib.load(roi_mask_file)

    roi_mask_data = roi_mask.get_data()

    roi_mask_data_shape = roi_mask.get_data().shape

    # print roi_mask_data_shape

    roi_mask_header = roi_mask.get_header()

    roi_mask_affine = roi_mask.get_affine()

    print(np.unique(roi_mask_data))

    print(data_vect)

    print(np.arange(data_vect.shape[0], dtype=float))

    if np.any(np.arange(data_vect.shape[0], dtype=float) != np.unique(roi_mask_data)[1:]):
        print("warning, ROI roi_mask not compatible with data  {} {}".format(np.arange(
            data_vect.shape[0], dtype=float).shape, np.unique(roi_mask_data)[1:].shape))

    data = np.zeros((roi_mask.shape), dtype=data_vect.dtype) - 1

    for roi_index in np.unique(roi_mask_data)[1:]:

        data[roi_mask_data == roi_index] = data_vect[roi_index]

    # print 'data img'
    data_img = nib.Nifti1Image(data, roi_mask_affine, roi_mask_header)

    return data_img
