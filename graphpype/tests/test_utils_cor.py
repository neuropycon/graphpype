import os

import numpy as np

# data_path = os.path.join(os.path.dirname(
# os.path.realpath(__file__)), "data", "data_nii")


import neuropycon_data as nd

data_path = os.path.join(nd.__path__[0], "data", "data_nii")

print(data_path)


img_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")

mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")

indexed_mask_file = os.path.join(data_path, "Atlas", "indexed_mask-Atlas.nii")

def test_neuropycon_data():
    
    try:
        import neuropycon_data
        
    except ImportError:
        print ("neuropycon_data not installed")
        
    assert os.path.exists(neuropycon_data.__path__[0]), "warning, could not find path {}, {}".format(neuropycon_data.__path__)
    
    assert os.path.exists(os.path.join(neuropycon_data.__path__[0],'data')), "warning, could not find path {}, {}".format(os.path.join(neuropycon_data.__path__[0],'data'),os.listdir(os.path.join(neuropycon_data.__path__[0])))
    
    assert os.path.exists(os.path.join(neuropycon_data.__path__[0],'data','data_nii')), "warning, could not find path {}, {}".format(data_path,os.listdir(os.path.join(neuropycon_data.__path__[0],'data')))
   
    assert os.path.exists(img_file), "warning, could not find path {}, {}".format(img_file,os.listdir(data_path))

def test_mean_select_mask_data():

    import nibabel as nib
    from graphpype.utils_cor import mean_select_mask_data

    data_img = nib.load(img_file).get_data()

    print(data_img.shape)

    data_mask = nib.load(mask_file).get_data()

    print(np.unique(data_mask))
    val = mean_select_mask_data(data_img, data_mask)

    print(val.shape)

    assert val.shape[0] == data_img.shape[3], "Error, time series do not have the same length {} != {}".format(
        val.shape[0], data_img.shape[1])


def test_mean_select_indexed_mask_data():

    import nibabel as nib
    from graphpype.utils_cor import mean_select_indexed_mask_data

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


if __name__ == '__main__':

    test_neuropycon_data()
    test_mean_select_mask_data()  # OK
    test_mean_select_indexed_mask_data()
