import os

from graphpype.pipelines.conmat_to_graph import (
    create_pipeline_conmat_to_graph_density)

from graphpype.utils_tests import load_test_data

def test_conmat_to_graph_simple():

    data_path = load_test_data("data_con")

    conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

    labels_file = os.path.join(data_path, "ROI_labels-Atlas.txt")
    coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

    wf = create_pipeline_conmat_to_graph_density(
        main_path=data_path, pipeline_name="conmat_to_graph_simple")

    wf.inputs.inputnode.conmat_file = conmat_file

    wf.write_graph(graph2use="colored")

    wf.run()

    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "graph.json"))
    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "graph.png"))


    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "compute_net_List", "Z_List.txt"))
    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "prep_rada", "Z_List.net"))
    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "community_rada", "Z_List.lol"))
    assert os.path.exists(os.path.join(data_path, "conmat_to_graph_simple_den_1_0", "node_roles", "node_roles.txt"))





"""
def test_conmat_to_graph_params():

    wf = create_pipeline_conmat_to_graph_density(
        main_path=data_path, pipeline_name="conmat_to_graph_params",
        con_den=0.1, plot=True, optim_seq="WS trfr 1")

    wf.inputs.inputnode.conmat_file = conmat_file

    wf.run()


def test_conmat_to_graph_labels_coords():

    wf = create_pipeline_conmat_to_graph_density(
        main_path=data_path, pipeline_name="conmat_to_graph_labels_coords",
        con_den=0.1, plot=True, optim_seq="WS trfr 1")

    wf.inputs.inputnode.conmat_file = conmat_file

    wf.inputs.inputnode.labels_file = labels_file
    wf.inputs.inputnode.coords_file = coords_file

    wf.run()
"""
