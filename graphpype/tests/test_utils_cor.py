import os

import numpy as np

#data_path = os.path.join(os.path.dirname(
    #os.path.realpath(__file__)), "data", "data_nii")


import neuropycon_data as nd

data_path = os.path.join(nd.__path__[0], "data", "data_nii")

print (data_path)


img_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")

mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")

indexed_mask_file = os.path.join(data_path, "Atlas","indexed_mask-Atlas.nii")


def test_mean_select_mask_data():

    import nibabel as nib
    from graphpype.utils_cor import mean_select_mask_data
    
    data_img = nib.load(img_file).get_data()
    
    print (data_img.shape)
    
    data_mask = nib.load(mask_file).get_data()
    
    print (np.unique(data_mask))
    val = mean_select_mask_data(data_img,data_mask)
    
    print (val.shape)
    
    assert val.shape[0] ==  data_img.shape[3], "Error, time series do not have the same length {} != {}".format(val.shape[0], data_img.shape[1]) 
    

def test_mean_select_indexed_mask_data():
    
    
    import nibabel as nib
    from graphpype.utils_cor import mean_select_indexed_mask_data
    
    data_img = nib.load(img_file).get_data()
    
    print (data_img.shape)
    
    data_indexed_mask = nib.load(indexed_mask_file).get_data()
    
    print (np.unique(data_indexed_mask))
    mean_masked_ts, keep_rois = mean_select_indexed_mask_data(data_img,data_indexed_mask)
    
    print (mean_masked_ts.shape)
    
    assert mean_masked_ts.shape[1] ==  data_img.shape[3], "Error, time series do not have the same length {} != {}".format(mean_masked_ts.shape[1], data_img.shape[3]) 
    
    assert keep_rois.shape[0] ==  len(np.unique(data_indexed_mask)) - 1 , "Error, keep_rois does not have the same length as unique indexes{} != {}".format(keep_rois.shape[0] ,  len(np.unique(data_indexed_mask)) - 1 ) 
    

if __name__ == '__main__':

    test_mean_select_mask_data() ## OK
    test_mean_select_indexed_mask_data()
