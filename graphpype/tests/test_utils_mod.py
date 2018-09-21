#######################################################################################################################
### test over file handling (.net, .lol, ect) required for radatools and possibly other graph tools (Louvain-Traag) ###
#######################################################################################################################

import os

import numpy as np

import neuropycon_data as nd

data_path = os.path.join(nd.__path__[0], "data", "data_con")

print(data_path)

#conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

#coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")

#Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")

Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")

info_nodes_file = os.path.join(
    data_path, "data_graph", "Z_List-info_nodes.txt")

info_global_file = os.path.join(
    data_path, "data_graph", "Z_List-info_global.txt")

info_dists_file = os.path.join(
    data_path, "data_graph", "Z_List-info_dists.txt")


node_roles_file = os.path.join(
    data_path, "data_graph", "node_roles.txt")

from graphpype.utils_net import (read_Pajek_corres_nodes_and_sparse_matrix)

from graphpype.utils_mod import (get_modularity_value_from_lol_file, read_lol_file,compute_modular_matrix,
                                 get_max_degree_from_node_info_file, get_values_from_global_info_file, get_values_from_signed_global_info_file,
                                 get_path_length_from_info_dists_file, get_path_length_from_info_dists_file,    
                                 get_strength_values_from_info_nodes_file,
                                 get_strength_pos_values_from_info_nodes_file, get_strength_neg_values_from_info_nodes_file,
                                 get_degree_pos_values_from_info_nodes_file, get_degree_neg_values_from_info_nodes_file,
                                 compute_roles)
####################### tests

##### lol file

### community_vect
def test_read_lol_file():
    """
    Test reading modular partition as radatools representation lol_file
    """
    community_vect = read_lol_file(lol_file)

    print(community_vect)

### modularity value
def test_get_modularity_value_from_lol_file():
    """
    get_modularity_value_from_lol_file
    """
    val = get_modularity_value_from_lol_file(lol_file)
    
    print (val)


def test_compute_modular_matrix():
    """
    Test computing modular matrix where edges between nodes belonging to the same module are 
    given the same value
    """
    corres, sp = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)

    community_vect = read_lol_file(lol_file)

    mod_mat = compute_modular_matrix(sp, community_vect)

    print(mod_mat)


#### Global properties (info_global) #########################################################

def test_get_max_degree_from_node_info_file():
    """
    get_max_degree_from_node_info_file
    """
    val = get_max_degree_from_node_info_file(info_nodes_file)
    
    print (val)
    
def test_get_values_from_global_info_file():
    """
    get_values_from_global_info_file
    """

    val = get_values_from_global_info_file(info_global_file)
    
    print (val)
    

def test_get_values_from_signed_global_info_file():
    """
    get_values_from_signed_global_info_file
    """
    
    val = get_values_from_signed_global_info_file(info_global_file)
    
    print (val)



def test_get_path_length_from_info_dists_file():
    """
    get_path_length_from_info_dists_file
    """
    
    val = get_path_length_from_info_dists_file(info_dists_file)
    
    print (val)


def test_get_path_length_from_info_dists_file():
    """
    get_path_length_from_info_dists_file
    """
    
    val = get_path_length_from_info_dists_file(info_dists_file)
    
    print (val)

#### Node properties (info-nodes) #########################################################

def test_get_strength_values_from_info_nodes_file():

    Strength = get_strength_values_from_info_nodes_file(info_nodes_file)

    print(Strength)


def test_get_strength_pos_values_from_info_nodes_file():

    Strength_Pos = get_strength_pos_values_from_info_nodes_file(
        info_nodes_file)

    print(Strength_Pos)


def test_get_strength_neg_values_from_info_nodes_file():

    Strength_Neg = get_strength_neg_values_from_info_nodes_file(
        info_nodes_file)

    print(Strength_Neg)


def test_get_degree_pos_values_from_info_nodes_file():

    Degree_Pos = get_degree_pos_values_from_info_nodes_file(info_nodes_file)

    print(Degree_Pos)


def test_get_degree_neg_values_from_info_nodes_file():

    Degree_Neg = get_degree_neg_values_from_info_nodes_file(info_nodes_file)

    print(Degree_Neg)

############## node roles (computation)
def test_compute_roles():
    """
    test_compute_roles
    """
    
    community_vect = read_lol_file(lol_file)

    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)

    val = compute_roles(community_vect,sparse_matrix,role_type = '4roles')
    
    return val
    
if __name__ == '__main__':

    #### modularity
    test_get_modularity_value_from_lol_file()
    test_read_lol_file()
    test_compute_modular_matrix()
    
    ### info_global
    test_get_max_degree_from_node_info_file()
    test_get_values_from_global_info_file()
    test_get_values_from_signed_global_info_file()
    
    #### info_dists
    test_get_path_length_from_info_dists_file()
    test_get_path_length_from_info_dists_file()

    ### info_dists
    test_get_strength_values_from_info_nodes_file()
    test_get_strength_pos_values_from_info_nodes_file()
    test_get_strength_neg_values_from_info_nodes_file()
    test_get_degree_pos_values_from_info_nodes_file()
    test_get_degree_neg_values_from_info_nodes_file()

    ### node roles
    test_compute_roles()
