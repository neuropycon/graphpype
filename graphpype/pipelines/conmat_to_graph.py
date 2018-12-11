"""
Pipeline to compute graph and modularity with radatools
"""
import nipype.pipeline.engine as pe

import nipype.interfaces.utility as niu

from graphpype.interfaces.radatools.rada import PrepRada, NetPropRada, CommRada
from graphpype.nodes.modularity import ComputeNetList, ComputeNodeRoles


def create_pipeline_conmat_to_graph_density(
        main_path, pipeline_name="graph_den_pipe", con_den=1.0, multi=False,
        mod=True, plot=False, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from connectivity matrices to graph analysis

    Threshold is density based

    Inputs (inputnode):

        * conmat_file
        * coords_file
        * labels_file
    """
    # TODO plot=True is kept for sake of clarity but is now unused
    pipeline = pe.Workflow(name=pipeline_name + "_den_" +
                           str(con_den).replace(".", "_"))
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['conmat_file', 'coords_file', 'labels_file']),
            name='inputnode')

    if multi:

        # density-based graphs

        # net_list
        compute_net_List = pe.MapNode(interface=ComputeNetList(
        ), name='compute_net_List', iterfield=["Z_cor_mat_file"])
        compute_net_List.inputs.density = con_den

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        # radatools

        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(
                interface=ComputeNodeRoles(role_type="4roles"),
                name='node_roles',
                iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')
    else:

        # density-based graphs

        # net_list
        compute_net_List = pe.Node(
            interface=ComputeNetList(), name='compute_net_List')
        compute_net_List.inputs.density = con_den

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        # radatools

        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(), name='prep_rada')
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.Node(
                interface=CommRada(), name='community_rada')
            community_rada.inputs.optim_seq = optim_seq

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
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_conmat_to_graph_threshold(
        main_path, pipeline_name="graph_thr_pipe", con_thr=1.0, multi=False,
        mod=True, plot=True, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from connectivity matrices to graph analysis

    Threshold is value based (con_thr)

    Inputs (inputnode):

        * conmat_file
        * coords_file
        * labels_file
    """
    # TODO Warning, need to be checked...
    # TODO Warning, should be merged with previous function
    # create_pipeline_conmat_to_graph_density

    # TODO plot=True is kept for sake of clarity but is now unused
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['conmat_file', 'coords_file', 'labels_file']),
        name='inputnode')

    if not multi:
        # density-based graphs

        # net_list
        compute_net_List = pe.Node(
            interface=ComputeNetList(), name='compute_net_List')
        compute_net_List.inputs.threshold = con_thr

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        # radatools
        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(),
                            name='prep_rada', iterfield=["net_List_file"])

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.Node(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

        # compute network properties with rada
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    else:

        # density-based graphs
        # net_list
        compute_net_List = pe.MapNode(interface=ComputeNetList(
        ), name='compute_net_List', iterfield=["Z_cor_mat_file"])
        compute_net_List.inputs.threshold = con_thr

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        # radatools
        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(
                interface=ComputeNodeRoles(role_type="4roles"),
                name='node_roles',
                iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_net_list_to_graph(
        main_path, pipeline_name="graph_net_pipe", multi=False, mod=True,
        plot=False, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from net_List (txt file) to graph analysis

    Inputs (inputnode):

        * net_List_file
        * coords_file
        * labels_file

    Could be used in the previous functions
    (create_pipeline_conmat_to_graph_density and
    create_pipeline_conmat_to_graph_threshold)
    """
    # TODO plot=True is kept for sake of clarity but is now unused
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['net_List_file', 'coords_file', 'labels_file']),
        name='inputnode')

    if not multi:

        # density-based graphs

        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(), name='prep_rada')
        prep_rada.inputs.network_type = "U"

        pipeline.connect(inputnode, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.Node(
                interface=CommRada(), name='community_rada')
            community_rada.inputs.optim_seq = optim_seq

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
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    else:
        assert False, "Error, never tested"
        # density-based graphs

        # radatools

        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(inputnode, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(interface=ComputeNodeRoles(
                role_type="4roles"), name='node_roles',
                    iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline
