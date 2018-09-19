#######################################################################################################################
### test over visbrain plug-ins  ###
#######################################################################################################################

import os

import numpy as np

#import neuropycon_data as nd

#data_path = os.path.join(nd.__path__[0], "data", "data_con")

#print(data_path)

#conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

#coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

#lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")

#Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")

#Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")

#info_nodes_file = os.path.join(
    #data_path, "data_graph", "Z_List-info_nodes.txt")


#from graphpype.utils_net import (return_net_list, read_lol_file,
                                 #read_Pajek_corres_nodes, read_Pajek_corres_nodes_and_sparse_matrix,
                                 #compute_modular_matrix, get_strength_values_from_info_nodes_file,
                                 #get_strength_pos_values_from_info_nodes_file, get_strength_neg_values_from_info_nodes_file,
                                 #get_degree_pos_values_from_info_nodes_file, get_degree_neg_values_from_info_nodes_file,
                                 #export_Louvain_net_from_list)

################################################ test if vispy and visbrain package are available

def test_vispy():
    
    try:
        import vispy
        
    except ImportError:
        print ("vispy not installed")
        
def test_visbrain():
    
    try:
        import visbrain
        
    except ImportError:
        print ("visbrain not installed")
        
        
if __name__ == '__main__':

    test_vispy()
    test_visbrain()
