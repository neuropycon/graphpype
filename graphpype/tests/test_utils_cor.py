import os

import numpy as np

from graphpype.pipelines.conmat_to_graph import create_pipeline_conmat_to_graph_density

data_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data", "data_nii")

img_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")

mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")


def test_mean_select_mask_data():

    import nibabel as nib
    from graphpype.utils_cor import mean_select_mask_data
    
    data_img = nib.load(img_file).get_data()
    
    print (data_img.shape)
    
    data_mask = nib.load(mask_file).get_data()
    
    print (np.unique(data_img))
    val = mean_select_mask_data(data_img,data_mask)
    
    print (val.shape)
    
    assert val.shape[0] ==  data_img.shape[3], "Error, time series do not have the same length {} != {}".format(val.shape[0], data_img.shape[1]) 
    


if __name__ == '__main__':

    test_mean_select_mask_data()
