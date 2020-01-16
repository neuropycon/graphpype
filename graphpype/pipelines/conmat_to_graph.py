"""
Pipeline to compute graph and modularity with radatools
"""
import nipype.pipeline.engine as pe

import nipype.interfaces.utility as niu


from graphpype.interfaces.bct import KCore
from graphpype.interfaces.radatools.rada import PrepRada, NetPropRada, CommRada
from graphpype.nodes.modularity import (ComputeNetList, ComputeNodeRoles,
                                        ComputeModuleMatProp)


def create_pipeline_conmat_to_graph_density(
        main_path, pipeline_name="graph_den_pipe", con_den=1.0, multi=False,
        mod=True, plot=False, optim_seq="WS trfr 100", compute_ndi=False):
    """
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

            node_roles.inputs.compute_ndi = compute_ndi

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

            node_roles.inputs.compute_ndi = compute_ndi

        # compute network properties with rada
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_conmat_to_graph_threshold(
        main_path, pipeline_name="graph_thr_pipe", con_thr=1.0, multi=False,
        mod=True, plot=True, optim_seq="WS trfr 100", compute_ndi=False):
    """
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

            # node roles
            node_roles = pe.Node(
                interface=ComputeNodeRoles(role_type="4roles"),
                name='node_roles')

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

            node_roles.inputs.compute_ndi = compute_ndi

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

            node_roles.inputs.compute_ndi = compute_ndi

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_net_list_to_graph(
        main_path, pipeline_name="graph_net_pipe", multi=False, mod=True,
        plot=False, optim_seq="WS trfr 100", compute_ndi=False):
    """
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

            node_roles.inputs.compute_ndi = compute_ndi

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
            node_roles.inputs.compute_ndi = compute_ndi

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


# create_pipeline_bct_graph
def create_pipeline_bct_graph(
        main_path, pipeline_name="graph_bct_pipe", con_den=1.0):
    """
    Description:

    Pipeline for computing module based graph properties

    Threshold is density based

    Inputs (inputnode):

        * conmat_files

    """
    # TODO plot=True is kept for sake of clarity but is now unused
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    # input node
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['conmat_file']),
        name='inputnode')

    # compute binary version
    bin_mat = pe.Node(interface=ComputeNetList(export_np_bin=True,
                                               density=con_den),
                      name="bin_mat")

    pipeline.connect(inputnode, 'conmat_file',
                     bin_mat, 'Z_cor_mat_file')

    # compute K core
    k_core = pe.Node(
        interface=KCore(),
        name="k_core")

    k_core.inputs.is_directed = False

    pipeline.connect(bin_mat, 'np_bin_mat_file',
                     k_core, 'np_mat_file')

    return pipeline


# create_pipeline_graph_module_properties
def create_pipeline_graph_module_properties(
        main_path, pipeline_name="graph_mod_pipe", con_den=1.0, multi=False,
        plot=True, export_excel=False):
    """
    Description:

    Pipeline for computing module based graph properties

    Threshold is density based

    Inputs (inputnode):

        * conmat_files
        * lol_file

    """
    # TODO plot=True is kept for sake of clarity but is now unused
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    # input node
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['group_conmat_file', 'rada_lol_file', 'Pajek_net_file']),
        name='inputnode')

    # module to graph properties
    mod_graph = pe.Node(
        interface=ComputeModuleMatProp(),
        name="mod_graph")

    mod_graph.inputs.export_excel = export_excel

    pipeline.connect(inputnode, 'Pajek_net_file',
                     mod_graph, 'Pajek_net_file')

    pipeline.connect(inputnode, 'rada_lol_file',
                     mod_graph, 'rada_lol_file')

    pipeline.connect(inputnode, 'group_conmat_file',
                     mod_graph, 'group_conmat_file')
    return pipeline
