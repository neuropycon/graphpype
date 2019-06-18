import os

from graphpype.utils_net import (read_Pajek_corres_nodes_and_sparse_matrix)
from graphpype.utils_mod import (get_modularity_value_from_lol_file,
                                 read_lol_file, compute_modular_matrix,
                                 get_max_degree_from_node_info_file,
                                 get_values_from_global_info_file,
                                 get_values_from_signed_global_info_file,
                                 get_path_length_from_info_dists_file,
                                 get_strength_values_from_info_nodes_file,
                                 get_strength_pos_values_from_info_nodes_file,
                                 get_strength_neg_values_from_info_nodes_file,
                                 get_degree_pos_values_from_info_nodes_file,
                                 get_degree_neg_values_from_info_nodes_file,
                                 compute_roles)

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_con")

lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")
Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")
info_nodes_file = os.path.join(
    data_path, "data_graph", "Z_List-info_nodes.txt")
info_global_file = os.path.join(
    data_path, "data_graph", "Z_List-info_global.txt")
info_dists_file = os.path.join(
    data_path, "data_graph", "Z_List-info_dists.txt")
node_roles_file = os.path.join(
    data_path, "data_graph", "node_roles.txt")


def test_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    assert os.path.exists(Pajek_net_file)
    assert os.path.exists(info_nodes_file)
    assert os.path.exists(info_global_file)
    assert os.path.exists(info_dists_file)
    assert os.path.exists(node_roles_file)


def test_read_lol_file():
    """
    Test reading modular partition as radatools representation lol_file
    """
    # TODO explicit assert test
    community_vect = read_lol_file(lol_file)
    print(community_vect)


def test_get_modularity_value_from_lol_file():
    """
    test_get_modularity_value_from_lol_file
    """
    # TODO explicit assert test
    val = get_modularity_value_from_lol_file(lol_file)
    print(val)


def test_compute_modular_matrix():
    """
    Test computing modular matrix where edges between nodes
    belonging to the same module are given the same value
    """
    # TODO explicit assert test
    corres, sp = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    community_vect = read_lol_file(lol_file)
    mod_mat = compute_modular_matrix(sp, community_vect)
    print(mod_mat)


def test_info_global():
    """
    Test Global properties (info_global)
    """
    # TODO explicit assert test
    val = get_max_degree_from_node_info_file(info_nodes_file)
    print(val)

    # TODO explicit assert test
    val = get_values_from_global_info_file(info_global_file)
    print(val)

    # TODO explicit assert test
    val = get_values_from_signed_global_info_file(info_global_file)
    print(val)

    # TODO explicit assert test
    val = get_path_length_from_info_dists_file(info_dists_file)
    print(val)

    # TODO explicit assert test
    val = get_path_length_from_info_dists_file(info_dists_file)
    print(val)


def test_node_properties():
    """
    test node properties
    """
    # TODO explicit assert test
    Strength = get_strength_values_from_info_nodes_file(info_nodes_file)
    print(Strength)

    # TODO explicit assert test
    Strength_Pos = get_strength_pos_values_from_info_nodes_file(
        info_nodes_file)
    print(Strength_Pos)

    # TODO explicit assert test
    Strength_Neg = get_strength_neg_values_from_info_nodes_file(
        info_nodes_file)
    print(Strength_Neg)

    # TODO explicit assert test
    Degree_Pos = get_degree_pos_values_from_info_nodes_file(info_nodes_file)
    print(Degree_Pos)

    # TODO explicit assert test
    Degree_Neg = get_degree_neg_values_from_info_nodes_file(info_nodes_file)
    print(Degree_Neg)


def test_compute_roles():
    """
    test_node_roles
    """
    # TODO explicit assert test
    community_vect = read_lol_file(lol_file)
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        Pajek_net_file)
    val = compute_roles(community_vect, sparse_matrix, role_type='4roles')
    print(val)
