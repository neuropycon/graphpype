import os

import numpy as np

data_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data", "data_con")

conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

labels_file = os.path.join(data_path, "ROI_labels-Atlas.txt")
coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")


indexed_mask_file = os.path.join(data_path, "Atlas","indexed_mask-Atlas.nii")


from graphpype.utils_net import return_net_list

##### test return_net_list from conmat to sparse format
def test_return_net_list():

    conmat = np.load(conmat_file)
    
    print (conmat.shape)
    
    int_factor = 1000
    
    list_conmat = return_net_list(conmat, int_factor)
    
    print (list_conmat.shape)
    
    assert list_conmat.shape[1] == 3, "Error, list_conmat should be format index_i, index_j value (sparse representation of graph) and have shape 3 instead of  {} ".format(list_conmat.shape[1]) 
    
if __name__ == '__main__':

    test_return_net_list() ## OK
