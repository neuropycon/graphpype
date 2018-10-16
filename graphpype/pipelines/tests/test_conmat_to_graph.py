import os
"""
from graphpype.pipelines.conmat_to_graph import (
    create_pipeline_conmat_to_graph_density)
"""
data_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "..", "tests", "data", "data_con")

conmat_file = os.path.join(data_path, "Z_cor_mat_resid_ts.npy")

labels_file = os.path.join(data_path, "ROI_labels-Atlas.txt")
coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

"""
def test_conmat_to_graph_simple():

    wf = create_pipeline_conmat_to_graph_density(
        main_path=data_path, pipeline_name="conmat_to_graph_simple")

    wf.inputs.inputnode.conmat_file = conmat_file

    wf.run()


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
