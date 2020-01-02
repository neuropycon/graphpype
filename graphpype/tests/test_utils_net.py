import os
import numpy as np

from graphpype.utils_net import (return_net_list,
                                 read_Pajek_corres_nodes,
                                 read_Pajek_corres_nodes_and_sparse_matrix,
                                 export_Louvain_net_from_list)


from graphpype.utils_tests import load_test_data

from graphpype.utils import _make_tmp_dir

data_path = load_test_data("data_con")
conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")
coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")
Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")
Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")


def test_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    assert os.path.exists(conmat_file)
    assert os.path.exists(coords_file)
    assert os.path.exists(Z_list_file)
    assert os.path.exists(Pajek_net_file)


def test_return_net_list():
    """
    test transforming matrix as np.array to list of edges
    """
    _make_tmp_dir()
    conmat = np.load(conmat_file)
    list_conmat = return_net_list(conmat, int_factor=1000)
    assert list_conmat.shape[1] == 3


def test_read_Pajek_corres_nodes():
    """Test reading corres node vector given a Pajek .net file"""
    _make_tmp_dir()
    corres = read_Pajek_corres_nodes(Pajek_net_file)
    print(corres)


def test_read_Pajek_corres_nodes_and_sparse_matrix():
    """Test reading corres node vector and sparse graph representation
    given a Pajek .net file"""
    _make_tmp_dir()
    corres, sp = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    assert len(corres) == sp.todense().shape[0]


def test_export_Louvain_net_from_list():
    """testing Louvain Traag file building (for sake of compatibility of older
    codes)"""
    _make_tmp_dir()
    coords = np.loadtxt(coords_file)
    Z_list = np.loadtxt(Z_list_file)
    Z_Louvain_file = os.path.abspath("Z_Louvain.txt")
    export_Louvain_net_from_list(Z_Louvain_file, Z_list, coords)
    assert os.path.exists(Z_Louvain_file)
