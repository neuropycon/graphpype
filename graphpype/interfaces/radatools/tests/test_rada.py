"""test rada"""
import os
from graphpype.interfaces.radatools.rada import (CommRada, PrepRada,
                                                 NetPropRada)
from graphpype.utils import _make_tmp_dir
from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_con")

Z_list_file = os.path.join(data_path, "data_graph", "Z_List.txt")
Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")
lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")


def test_neuropycon_data():
    """test if test_data is accessible"""
    assert os.path.exists(Z_list_file)
    assert os.path.exists(Pajek_net_file)


def test_radatools():
    """test if radatools is installed and accessible from anywhere"""
    assert not os.system("List_To_Net.exe")


def test_prep_rada():
    """test PrepRada"""
    _make_tmp_dir()

    prep_rada = PrepRada()
    prep_rada.inputs.net_List_file = Z_list_file
    prep_rada.inputs.network_type = "U"
    val = prep_rada.run().outputs

    assert os.path.exists(val.Pajek_net_file)
    # os.remove(val.Pajek_net_file)


def test_comm_rada():
    """test CommRada"""
    _make_tmp_dir()

    comm_rada = CommRada()
    comm_rada.inputs.Pajek_net_file = Pajek_net_file
    comm_rada.inputs.optim_seq = "WS trfr 1"

    val = comm_rada.run().outputs

    assert os.path.exists(val.rada_lol_file)
    assert os.path.exists(val.rada_log_file)
    assert os.path.exists(val.lol_log_file)


def test_net_prop_rada():
    """test NetPropRada"""
    _make_tmp_dir()

    net_prop_rada = NetPropRada()
    net_prop_rada.inputs.Pajek_net_file = Pajek_net_file

    val = net_prop_rada.run().outputs

    assert os.path.exists(val.global_file)
    assert os.path.exists(val.dists_file)
    assert os.path.exists(val.degrees_file)
    assert os.path.exists(val.nodes_file)
    assert os.path.exists(val.edges_betw_file)
    assert os.path.exists(val.rada_log_file)
