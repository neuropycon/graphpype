#######################################################################################################################
### test over file handling (.net, .lol, ect) required for radatools and possibly other graph tools (Louvain-Traag) ###
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

info_nodes_file = os.path.join(
    data_path, "data_graph", "Z_List-info_nodes.txt")


from graphpype.utils_net import (return_net_list, read_lol_file,
                                 read_Pajek_corres_nodes, read_Pajek_corres_nodes_and_sparse_matrix,
                                 compute_modular_matrix, get_strength_values_from_info_nodes_file,
                                 get_strength_pos_values_from_info_nodes_file, get_strength_neg_values_from_info_nodes_file,
                                 get_degree_pos_values_from_info_nodes_file, get_degree_neg_values_from_info_nodes_file,
                                 export_Louvain_net_from_list)

# test return_net_list from conmat to sparse format


#def test_return_net_list():

    #conmat = np.load(conmat_file)

    #print(conmat.shape)

    #int_factor = 1000

    #list_conmat = return_net_list(conmat, int_factor)

    #print(list_conmat.shape)

    #assert list_conmat.shape[1] == 3, "Error, list_conmat should be format index_i, index_j value (sparse representation of graph) and have shape 3 instead of  {} ".format(
        #list_conmat.shape[1])


#def test_read_lol_file():

    #community_vect = read_lol_file(lol_file)

    #print(community_vect)


#def test_read_Pajek_corres_nodes():

    #corres = read_Pajek_corres_nodes(Pajek_net_file)

    #print(corres)


#def test_read_Pajek_corres_nodes_and_sparse_matrix():

    #corres, sp = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)

    #print(corres)

    #print(sp)


#def test_compute_modular_matrix():

    #corres, sp = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)

    #community_vect = read_lol_file(lol_file)

    #mod_mat = compute_modular_matrix(sp, community_vect)

    #print(mod_mat)

    #print(np.unique(mod_mat))

## Node properties


#def test_get_strength_values_from_info_nodes_file():

    #Strength = get_strength_values_from_info_nodes_file(info_nodes_file)

    #print(Strength)


#def test_get_strength_pos_values_from_info_nodes_file():

    #Strength_Pos = get_strength_pos_values_from_info_nodes_file(
        #info_nodes_file)

    #print(Strength_Pos)


#def test_get_strength_neg_values_from_info_nodes_file():

    #Strength_Neg = get_strength_neg_values_from_info_nodes_file(
        #info_nodes_file)

    #print(Strength_Neg)


#def test_get_degree_pos_values_from_info_nodes_file():

    #Degree_Pos = get_degree_pos_values_from_info_nodes_file(info_nodes_file)

    #print(Degree_Pos)


#def test_get_degree_neg_values_from_info_nodes_file():

    #Degree_Neg = get_degree_neg_values_from_info_nodes_file(info_nodes_file)

    #print(Degree_Neg)

## testing Louvain Traag file building (for sake of compatibility of older codes)


def test_export_Louvain_net_from_list():

    print("***************************************")
    
    print (data_path)
    
    try:
        import neuropycon_data
        
    except ImportError:
        print ("neuropycon_data not installed")
        
    assert os.path.exists(neuropycon_data.__path__[0]), "warning, could not find path {}, {}".format(neuropycon_data.__path__)
    
    print(os.listdir(neuropycon_data.__path__[0]))
    
    assert os.path.exists(os.path.join(neuropycon_data.__path__[0],'data')), "warning, could not find path {}, {}".format(os.path.join(neuropycon_data.__path__[0],'data'),os.listdir(os.path.join(neuropycon_data.__path__[0],'data')))
    
    assert os.path.exists(data_path), "warning, could not find path {}, {}".format(data_path,os.listdir(neuropycon_data.__path__[0]))

    assert os.path.exists(coords_file), "warning, could not find path {}, {}".format(coords_file,os.listdir(data_path))

    print(os.listdir(data_path))
    
    coords = np.loadtxt(coords_file)

    Z_list = np.loadtxt(Z_list_file)

    Z_Louvain_file = os.path.abspath("Z_Louvain.txt")

    export_Louvain_net_from_list(Z_Louvain_file, Z_list, coords)


if __name__ == '__main__':

    # test_return_net_list() ## OK
    # test_read_lol_file()
    # test_read_Pajek_corres_nodes()
    # test_read_Pajek_corres_nodes_and_sparse_matrix()
    # test_compute_modular_matrix()

    # test_get_strength_values_from_info_nodes_file()
    # test_get_strength_pos_values_from_info_nodes_file()
    # test_get_strength_neg_values_from_info_nodes_file()
    # test_get_degree_pos_values_from_info_nodes_file()
    # test_get_degree_neg_values_from_info_nodes_file()

    test_export_Louvain_net_from_list()
