#######################################################################################################################
### test over visbrain plug-ins  ###
#######################################################################################################################

import os

import numpy as np

import neuropycon_data as nd

data_path = os.path.join(nd.__path__[0], "data", "data_con")

print(data_path)

conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")

Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")

Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")

################################################ test if vispy and visbrain package are available

def test_vispy():
    """
    testing if vispy is installed
    """
    
    try:
        import vispy
        
    except ImportError:
        print ("vispy not installed")
        
def test_visbrain():
    """
    testing if vispy is installed
    """
    try:
        import visbrain
        
    except ImportError:
        print ("visbrain not installed")
        
        
if __name__ == '__main__':

    test_vispy()
    test_visbrain()
