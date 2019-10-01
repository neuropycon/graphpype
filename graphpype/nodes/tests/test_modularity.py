import os
from graphpype.nodes.modularity import (ComputeNetList,
                                        ComputeNodeRoles, ComputeModuleMatProp)
from graphpype.utils import _make_tmp_dir

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_con")

conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")
coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")
Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")
lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")


def test_neuropycon_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    assert os.path.exists(conmat_file)
    assert os.path.exists(coords_file)
    assert os.path.exists(Z_list_file)
    assert os.path.exists(Pajek_net_file)
    assert os.path.exists(lol_file)


def test_compute_net_list():
    """ test ComputeNetList"""
    _make_tmp_dir()
    compute_net_list = ComputeNetList()
    compute_net_list.inputs.Z_cor_mat_file = conmat_file

    val = compute_net_list.run().outputs
    print(val)
    assert os.path.exists(val.net_List_file)
    os.remove(val.net_List_file)


def test_compute_node_roles():
    """ test ComputeNodeRoles"""
    _make_tmp_dir()

    compute_node_roles = ComputeNodeRoles(role_type="4roles")
    compute_node_roles.inputs.rada_lol_file = lol_file
    compute_node_roles.inputs.Pajek_net_file = Pajek_net_file
    compute_node_roles.inputs.compute_ndi = True

    val = compute_node_roles.run().outputs
    print(val)
    assert os.path.exists(val.node_roles_file)
    assert os.path.exists(val.all_Z_com_degree_file)
    assert os.path.exists(val.all_participation_coeff_file)

    assert os.path.exists(val.ndi_values_file)

    os.remove(val.node_roles_file)
    os.remove(val.all_Z_com_degree_file)
    os.remove(val.all_participation_coeff_file)


def test_compute_module_mat_prop():
    """ test ComputeModuleMatProp"""
    _make_tmp_dir()

    compute_module_graph_prop = ComputeModuleMatProp()
    compute_module_graph_prop.inputs.rada_lol_file = lol_file
    compute_module_graph_prop.inputs.Pajek_net_file = Pajek_net_file
    compute_module_graph_prop.inputs.group_conmat_file = conmat_file

    val = compute_module_graph_prop.run().outputs
    print(val)

    assert os.path.exists(val.df_avgmat_files[0])
