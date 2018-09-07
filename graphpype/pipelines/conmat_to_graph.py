# -*- coding: utf-8 -*-
"""
Pipeline to compute graph and modularity with radatools and (possibly if installed) plot with igraph
"""

#import sys
#import time

#import numpy as np
#import scipy.sparse as sp

import nipype.pipeline.engine as pe
#from nipype.utils.misc import show_files

import nipype.interfaces.utility as niu

from graphpype.interfaces.radatools import PrepRada, NetPropRada, CommRada
from graphpype.nodes.modularity import ComputeNetList, ComputeNodeRoles

#import imp

try:
    import igraph
    can_plot_igraph = True
    from graphpype.interfaces.plot_igraph.plots import PlotIGraphModules

except ImportError:
    can_plot_igraph = False


def create_pipeline_conmat_to_graph_density(main_path, pipeline_name="graph_den_pipe", con_den=1.0, multi=False, mod=True, plot=False, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from connectivity matrices to graph analysis

    Threshold is density based

    Inputs (inputnode):

        * conmat_file
        * coords_file
        * labels_file


    """

    pipeline = pe.Workflow(name=pipeline_name + "_den_" +
                           str(con_den).replace(".", "_"))
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=['conmat_file', 'coords_file', 'labels_file']),
                        name='inputnode')

    if plot == True and can_plot_igraph == False:

        print("warning, no igraph is installed, plot is set back to zero.")

        plot = False

    if multi:

        ################################################ density-based graphs #################################################

        # net_list
        compute_net_List = pe.MapNode(interface=ComputeNetList(
        ), name='compute_net_List', iterfield=["Z_cor_mat_file"])
        compute_net_List.inputs.density = con_den

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(interface=ComputeNodeRoles(
                role_type="4roles"), name='node_roles', iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.MapNode(interface=PlotIGraphModules(
                ), name='plot_igraph_modules_rada', iterfield=['Pajek_net_file', 'rada_lol_file', 'node_roles_file'])

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

                pipeline.connect(node_roles, 'node_roles_file',
                                 plot_igraph_modules_rada, 'node_roles_file')
                pipeline.connect(inputnode, 'labels_file',
                                 plot_igraph_modules_rada, 'labels_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])
        #net_prop.inputs.radatools_path = radatools_path

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

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(), name='prep_rada')
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

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

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.Node(
                    interface=PlotIGraphModules(), name='plot_igraph_modules_rada')

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

                pipeline.connect(node_roles, 'node_roles_file',
                                 plot_igraph_modules_rada, 'node_roles_file')

                pipeline.connect(inputnode, 'coords_file',
                                 plot_igraph_modules_rada, 'coords_file')
                pipeline.connect(inputnode, 'labels_file',
                                 plot_igraph_modules_rada, 'labels_file')

        # compute network properties with rada
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')
        #net_prop.inputs.radatools_path = radatools_path

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_conmat_to_graph_threshold(main_path, pipeline_name="graph_thr_pipe", con_thr=1.0, multi=False, mod=True, plot=True, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from connectivity matrices to graph analysis

    Threshold is value based (con_thr)

    Inputs (inputnode):

        * conmat_file
        * coords_file
        * labels_file

    !Warning, need to be checked...
    !Warning, should be merged with previous function create_pipeline_conmat_to_graph_density

    """
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['conmat_file', 'coords_file', 'labels_file']), name='inputnode')

    if plot == True and can_plot_igraph == False:

        plot = False

    if multi == False:

        # density-based graphs

        # net_list
        compute_net_List = pe.Node(
            interface=ComputeNetList(), name='compute_net_List')
        compute_net_List.inputs.threshold = con_thr
        #compute_net_List.inputs.density = None

        pipeline.connect(inputnode, 'conmat_file',
                         compute_net_List, 'Z_cor_mat_file')

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(),
                            name='prep_rada', iterfield=["net_List_file"])

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

            # compute community with radatools
            community_rada = pe.Node(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.Node(interface=PlotIGraphModules(
                ), name='plot_igraph_modules_rada', iterfield=['Pajek_net_file', 'rada_lol_file'])

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

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

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            community_rada.inputs.optim_seq = optim_seq

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(interface=ComputeNodeRoles(
                role_type="4roles"), name='node_roles', iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.MapNode(interface=PlotIGraphModules(
                ), name='plot_igraph_modules_rada', iterfield=['Pajek_net_file', 'rada_lol_file', 'node_roles_file'])

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

                pipeline.connect(node_roles, 'node_roles_file',
                                 plot_igraph_modules_rada, 'node_roles_file')
                pipeline.connect(inputnode, 'labels_file',
                                 plot_igraph_modules_rada, 'labels_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])
        #net_prop.inputs.radatools_path = radatools_path

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline


def create_pipeline_net_list_to_graph(main_path, pipeline_name="graph_net_pipe", multi=False, mod=True, plot=False, optim_seq="WS trfr 100"):
    """
    Description:

    Pipeline from net_List (txt file) to graph analysis

    Inputs (inputnode):

        * net_List_file
        * coords_file
        * labels_file

    Could be used in the previous functions (create_pipeline_conmat_to_graph_density and create_pipeline_conmat_to_graph_threshold)
    """

    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=['net_List_file', 'coords_file', 'labels_file']),
                        name='inputnode')

    if plot == True and can_plot_igraph == False:

        plot = False

    if multi == False:

        # density-based graphs

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.Node(interface=PrepRada(), name='prep_rada')
        prep_rada.inputs.network_type = "U"

        pipeline.connect(inputnode, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

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

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.Node(
                    interface=PlotIGraphModules(), name='plot_igraph_modules_rada')

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

                pipeline.connect(node_roles, 'node_roles_file',
                                 plot_igraph_modules_rada, 'node_roles_file')

                pipeline.connect(inputnode, 'coords_file',
                                 plot_igraph_modules_rada, 'coords_file')
                pipeline.connect(inputnode, 'labels_file',
                                 plot_igraph_modules_rada, 'labels_file')

        # compute network properties with rada
        net_prop = pe.Node(interface=NetPropRada(
            optim_seq="A"), name='net_prop')
        #net_prop.inputs.radatools_path = radatools_path

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    else:

        ################################################ density-based graphs #################################################

        ##### radatools ################################################################

        # prepare net_list for radatools processing
        prep_rada = pe.MapNode(interface=PrepRada(),
                               name='prep_rada', iterfield=["net_List_file"])
        prep_rada.inputs.network_type = "U"

        pipeline.connect(compute_net_List, 'net_List_file',
                         prep_rada, 'net_List_file')

        if mod == True:

            # compute community with radatools
            community_rada = pe.MapNode(interface=CommRada(
            ), name='community_rada', iterfield=["Pajek_net_file"])
            #community_rada.inputs.optim_seq = radatools_optim

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             community_rada, 'Pajek_net_file')

            # node roles
            node_roles = pe.MapNode(interface=ComputeNodeRoles(
                role_type="4roles"), name='node_roles', iterfield=['Pajek_net_file', 'rada_lol_file'])

            pipeline.connect(prep_rada, 'Pajek_net_file',
                             node_roles, 'Pajek_net_file')
            pipeline.connect(community_rada, 'rada_lol_file',
                             node_roles, 'rada_lol_file')

            if plot == True:

                # plot_igraph_modules_rada
                plot_igraph_modules_rada = pe.MapNode(interface=PlotIGraphModules(
                ), name='plot_igraph_modules_rada', iterfield=['Pajek_net_file', 'rada_lol_file', 'node_roles_file'])

                pipeline.connect(prep_rada, 'Pajek_net_file',
                                 plot_igraph_modules_rada, 'Pajek_net_file')
                pipeline.connect(community_rada, 'rada_lol_file',
                                 plot_igraph_modules_rada, 'rada_lol_file')

                pipeline.connect(node_roles, 'node_roles_file',
                                 plot_igraph_modules_rada, 'node_roles_file')

        # compute network properties with rada
        net_prop = pe.MapNode(interface=NetPropRada(
            optim_seq="A"), name='net_prop', iterfield=["Pajek_net_file"])
        #net_prop.inputs.radatools_path = radatools_path

        pipeline.connect(prep_rada, 'Pajek_net_file',
                         net_prop, 'Pajek_net_file')

    return pipeline
