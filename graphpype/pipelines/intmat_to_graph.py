"""
Support function for net handling
"""
import nipype.interfaces.utility as niu

import nipype.pipeline.engine as pe

from graphpype.nodes.modularity import ComputeIntNetList, ComputeNodeRoles
from graphpype.interfaces.radatools.rada import PrepRada, CommRada, NetPropRada

# threshold-based graphs


def create_pipeline_intmat_to_graph_threshold(
        main_path, analysis_name="int_graph_thr_pipe", threshold=50,
        mod=False, plot=False, radatools_optim=""):
    """

    Description:

    Pipeline from integer matrices (normally coclassification matrices)
    to graph analysis

    Threshold is value based, normally a pourcentage (threshold, 50 -> 50%)

    Inputs (inputnode):

        * int_mat_file
        * coords_file
        * labels_file


    Comments:

    Was used for coclassification, not so much used anymore

    """
    # TODO plot=True is kept for sake of clarity but is now unused
    # TODO should be merged with create_pipeline_net_list_to_graph in
    # conmat_to_graph

    pipeline = pe.Workflow(name=analysis_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['int_mat_file', 'coords_file', 'labels_file']),
        name='inputnode')

    # compute Z_list from coclass matrix
    compute_net_list = pe.Node(
        interface=ComputeIntNetList(), name='compute_net_list')
    compute_net_list.inputs.threshold = threshold

    pipeline.connect(inputnode, 'int_mat_file',
                     compute_net_list, 'int_mat_file')

    # radatools

    # --- prepare net_list for radatools processing
    prep_rada = pe.Node(interface=PrepRada(), name='prep_rada')

    pipeline.connect(compute_net_list, 'net_List_file',
                     prep_rada, 'net_List_file')

    if mod:

        # compute community with radatools
        community_rada = pe.Node(interface=CommRada(
        ), name='community_rada', iterfield=["Pajek_net_file"])
        community_rada.inputs.optim_seq = radatools_optim

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         community_rada, 'Pajek_net_file')

        # node roles
        node_roles = pe.Node(interface=ComputeNodeRoles(
            role_type="4roles"), name='node_roles')

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         node_roles, 'Pajek_net_file')
        pipeline.connect(community_rada, 'rada_lol_file',
                         node_roles, 'rada_lol_file')

    # compute network properties with rada
    net_prop = pe.Node(interface=NetPropRada(optim_seq="A"), name='net_prop')

    pipeline.connect(prep_rada, 'Pajek_net_file', net_prop, 'Pajek_net_file')

    return pipeline
