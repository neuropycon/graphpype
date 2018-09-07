# -*- coding: utf-8 -*-
"""
Support function for igraph
"""

import numpy as np
import os


import igraph as ig

import math

from graphpype.utils_igraph import project2D_np

from graphpype.utils_dtype_coord import where_in_coords, find_index_in_coords
from graphpype.utils_igraph import add_non_null_labels, return_base_weighted_graph

from graphpype.utils_color import igraph_colors, new_igraph_colors, static_igraph_colors


def plot_3D_igraph_int_mat_modules(plot_nbs_adj_mat_file, int_matrix, coords=np.array([]), labels=[], edge_colors=new_igraph_colors, node_col_labels=np.array([]), nodes_sizes=np.array([]), view_from='_from_left'):

    g = return_base_weighted_graph(int_matrix)

    print(labels)

    if len(labels) == len(g.vs):

        add_non_null_labels(g, labels)

    print(np.unique(int_matrix))

    print(int_matrix)

    if len(edge_colors) < len(np.unique(int_matrix)[1:]):

        print("Warning, edge_colors {} < np.unique(int_matrix)[1:] {}".format(
            len(edge_colors), len(np.unique(int_matrix)[1:])))

        0/0

    for i, index in enumerate(np.unique(int_matrix)[1:]):

        colored_egde_list = g.es.select(weight_eq=index)

        print(len(colored_egde_list), np.sum(int_matrix == index))

        colored_egde_list["color"] = edge_colors[i]

        print(i, index, len(colored_egde_list))

    print(g.es['color'])

    if node_col_labels.size == len(g.vs) and nodes_sizes.size == len(g.vs):

        for i, v in enumerate(g.vs):

            if node_col_labels[i] != 0:

                print(node_col_labels[i])

                v["color"] = edge_colors[node_col_labels[i]-1]

                v["size"] = nodes_sizes[i]

            else:

                v["color"] = "Black"

                v["size"] = 0.1

        print(g.vs["size"])

        print(g.vs["color"])

    else:

        vertex_degree = np.array(g.degree())*0.2

    if coords.shape[0] != len(g.vs):

        layout2D = g.layout_fruchterman_reingold()

    else:

        if view_from == '_from_left':

            view = [0.0, 0.0]
        elif view_from == '_from_front':

            view = [0., 90.0]

        elif view_from == '_from_top':

            view = [90., -90]

        elif view_from == '_from_behind':

            view = [0., -90.0]

        layout2D = project2D_np(
            coords, angle_alpha=view[0], angle_beta=view[1]).tolist()

    # print g
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
    ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D, edge_curved=False)


def plot_3D_igraph_bin_mat(plot_nbs_adj_mat_file, int_matrix, coords=np.array([]), labels=[], color="blue"):

    g = return_base_weighted_graph(int_matrix)

    print(labels)

    if len(labels) == len(g.vs):

        add_non_null_labels(g, labels)

    vertex_degree = np.array(g.degree())*0.2

    if coords.shape[0] != len(g.vs):

        layout2D = g.layout_fruchterman_reingold()

    else:

        layout2D = project2D_np(coords).tolist()

    print(len(g.es))

    if len(g.es) > 0:

        # print g.es['weight']

        edge_col = []

        for w in g.es['weight']:

            edge_col.append(color)

        g.es['color'] = edge_col

        ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D,
                vertex_size=0.2,    edge_width=1)

    else:
        ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D,
                vertex_size=0.2,    edge_width=0.01)


def plot_3D_igraph_int_mat(plot_nbs_adj_mat_file, int_matrix, coords=np.array([]), labels=[], edge_colors=['Gray', 'Blue', 'Red'], node_col_labels=np.array([]), nodes_sizes=np.array([]), view_from='_from_left'):

    g = return_base_weighted_graph(int_matrix)

    print(g)

    print(labels)

    if len(labels) == len(g.vs):

        add_non_null_labels(g, labels)

    print(np.unique(int_matrix))

    for i, index in enumerate(np.unique(int_matrix)[1:]):

        colored_egde_list = g.es.select(weight_eq=index)

        print(len(colored_egde_list), np.sum(int_matrix == index))

        colored_egde_list["color"] = edge_colors[i]

        print(i, index, len(colored_egde_list))

        print(g.es['color'])

    if node_col_labels.size == len(g.vs) and nodes_sizes.size == len(g.vs):

        for i, v in enumerate(g.vs):

            if node_col_labels[i] != 0:

                print(node_col_labels[i])

                v["color"] = edge_colors[node_col_labels[i]-1]

                v["size"] = nodes_sizes[i]

            else:

                v["color"] = "Black"

                v["size"] = 0.1

        print(g.vs["size"])

        print(g.vs["color"])

    else:

        vertex_degree = np.array(g.degree())*0.2

    if coords.shape[0] != len(g.vs):

        layout2D = g.layout_fruchterman_reingold()

    else:

        if view_from == '_from_left':

            view = [0.0, 0.0]
        elif view_from == '_from_front':

            view = [0., 90.0]

        elif view_from == '_from_top':

            view = [90., -90]

        elif view_from == '_from_behind':

            view = [0., -90.0]

        layout2D = project2D_np(
            coords, angle_alpha=view[0], angle_beta=view[1]).tolist()

    # print g
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
    ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D, edge_curved=False)


def plot_3D_igraph_signed_int_mat(plot_nbs_adj_mat_file, int_matrix, coords=np.array([]), labels=[]):

    print(np.sum(int_matrix == 1))
    print(np.sum(int_matrix == 0))
    print(np.sum(int_matrix == -1))

    g = return_base_weighted_graph(int_matrix)

    print(g.es['weight'])

    print(labels)

    if len(labels) == len(g.vs):

        add_non_null_labels(g, labels)

    vertex_degree = np.array(g.degree())*0.2

    if coords.shape[0] != len(g.vs):

        layout2D = g.layout_fruchterman_reingold()

    else:

        layout2D = project2D_np(coords).tolist()

    print(len(g.es))

    if len(g.es) > 0:

        # print g.es['weight']

        edge_col = []

        for w in g.es['weight']:

            #(e0,e1) = e.tuple

            # print int(e.weight)

            #comp_index = int(e.weight)

            if int(w) == -1:
                edge_col.append('green')
            elif int(w) == -2:
                edge_col.append('cyan')
            elif int(w) == -3:
                edge_col.append('blue')
            elif int(w) == -4:
                edge_col.append('darkblue')

            elif int(w) == 1:
                edge_col.append('yellow')
            elif int(w) == 2:
                edge_col.append('orange')
            elif int(w) == 3:
                edge_col.append('darkorange')
            elif int(w) == 4:
                edge_col.append('red')

            print(w, int(w))

        #g_all.es['names'] = edge_list_names
        #g_all.vs['names'] = node_list_names
        g.es['color'] = edge_col

        ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D,
                vertex_size=0.2,    edge_width=1)

    else:
        ig.plot(g, plot_nbs_adj_mat_file, layout=layout2D,
                vertex_size=0.2,    edge_width=0.01)

# using relative coords directly


from graphpype.utils_igraph import add_non_null_labels, add_vertex_colors, create_module_edge_list, add_node_shapes
from graphpype.utils_plot import plot_colorbar


def plot_3D_igraph_all_modules(community_vect, Z_list, node_coords=np.array([]), node_labels=[], layout='', node_roles=np.array([]), plot_color_bar=True):

    if (community_vect.shape[0] != Z_list.shape[0] or community_vect.shape[0] != Z_list.shape[1]):
        print("Warning, community_vect {} != Z_list {}".format(
            community_vect.shape[0], Z_list.shape))

    # creating from coomatrix and community_vect
    g_all, used_colors = create_module_edge_list(
        Z_list, community_vect, list_colors=static_igraph_colors)

    print(used_colors)

    # vertex colors

    used_colors = add_vertex_colors(
        g_all, community_vect, list_colors=static_igraph_colors)

    print(used_colors)

    if plot_color_bar == True:

        colorbar_file = os.path.abspath("colorbar.eps")

        plot_colorbar(colorbar_file, used_colors)

    # print g_all.vs['color']
    # print g_all

    if len(node_labels) != 0:
        print("non void labels found")
        add_non_null_labels(g_all, node_labels)

        # print g_all.vs['label']

    else:
        print("empty labels")

    if node_roles.size != 0:

        # print node_roles
        add_node_shapes(g_all, node_roles)

        # print g_all.vs["shape"]

    else:
        print("no shapes")

    if layout == 'FR':

        print("plotting with Fruchterman-Reingold layout")

        g_all['layout'] = g_all.layout_fruchterman_reingold()

        Z_list_all_modules_file = os.path.abspath("All_modules_FR.eps")

        ig.plot(g_all, Z_list_all_modules_file,
                edge_width=0.1, edge_curved=False)

        return Z_list_all_modules_file

    else:

        if node_coords.size != 0:

            print("non void coords found")

            views = [[0.0, 0.0], [0., 90.0], [90., 0.0], [0., -90.0]]

            suf = ["_from_left", "_from_front", "_from_top", "_from_behind"]

            Z_list_all_modules_files = []

            for i, view in enumerate(views):

                print(view)

                Z_list_all_modules_file = os.path.abspath(
                    "All_modules_3D" + suf[i] + ".eps")

                Z_list_all_modules_files.append(Z_list_all_modules_file)

                layout2D = project2D_np(
                    node_coords, angle_alpha=view[0], angle_beta=view[1])

                g_all['layout'] = layout2D.tolist()

                ig.plot(g_all, Z_list_all_modules_file,
                        edge_width=0.1, edge_curved=False)

            return Z_list_all_modules_files

        else:

            print("Warning, should have coordinates, or specify layout = 'FR' (Fruchterman-Reingold layout) in options")


from graphpype.utils_igraph import select_edge_list_outside_module


def plot_3D_igraph_single_modules(community_vect, Z_list, node_coords=np.array([]), node_labels=[], layout='', node_roles=np.array([]), plot_color_bar=True, nb_min_nodes_by_module=5):

    if (community_vect.shape[0] != Z_list.shape[0] or community_vect.shape[0] != Z_list.shape[1]):
        print("Warning, community_vect {} != Z_list {}".format(
            community_vect.shape[0], Z_list.shape))

    # creating from coomatrix and community_vect
    g_all, used_colors = create_module_edge_list(
        Z_list, community_vect, list_colors=static_igraph_colors)

    print(used_colors)

    # vertex colors

    used_colors = add_vertex_colors(
        g_all, community_vect, list_colors=static_igraph_colors)

    print(used_colors)

    if plot_color_bar == True:

        colorbar_file = os.path.abspath("colorbar.eps")

        plot_colorbar(colorbar_file, used_colors)

    # print g_all.vs['color']
    # print g_all

    if len(node_labels) != 0:
        print("non void labels found")
        add_non_null_labels(g_all, node_labels)

        # print g_all.vs['label']

    else:
        print("empty labels")

    if node_roles.size != 0:

        # print node_roles
        add_node_shapes(g_all, node_roles)

        # print g_all.vs["shape"]

    else:
        print("no shapes")

      ##################
    Z_list_all_modules_files = []

    for mod_index in np.unique(community_vect):

        print("Module index %d has %d nodes" %
              (mod_index, np.sum(community_vect == mod_index)))

        if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:

            print("Not enough nodes (%d), skipping plot" %
                  (np.sum(community_vect == mod_index)))
            continue

        g_sel = g_all.copy()

        # delete edges that are not within module
        select_edge_list_outside_module(
            g_sel, Z_list, community_vect, mod_index)

        print(g_sel)

        # change size and colors of nodes
        vertex_col = []
        vertex_size = []
        #vertex_shape = []

        for i, v in enumerate(g_sel.vs):
            cur_mod_index = community_vect[i]

            print(v)

            if (mod_index == cur_mod_index):
                vertex_col.append(v["color"])
                vertex_size.append(v["size"])
                # vertex_shape.append(v["shape"])
            else:
                vertex_col.append("lightgrey")
                vertex_size.append(1)
                # vertex_shape.append("cross")

        g_sel.vs['color'] = vertex_col
        g_sel.vs['size'] = vertex_size
        #g_sel.vs['shape'] = vertex_shape

        # single view
        view = [0.0, 0.0]

        Z_list_all_modules_file = os.path.abspath(
            "module_" + str(mod_index) + "_from_left.eps")

        layout2D = project2D_np(
            node_coords, angle_alpha=view[0], angle_beta=view[1])

        g_sel['layout'] = layout2D.tolist()

        ig.plot(g_sel, Z_list_all_modules_file,
                edge_width=0.1, edge_curved=False)

        Z_list_all_modules_files.append(Z_list_all_modules_file)

        # all view
        #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]

        #suf = ["_from_left","_from_front","_from_top","_from_behind"]

        #Z_list_all_modules_files = []

        # for i,view in enumerate(views):

        # print view

        #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + suf[i] + ".eps")

        #layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])

        #g_sel['layout'] = layout2D.tolist()

        ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)

        # Z_list_all_modules_files.append(Z_list_all_modules_file)

    return Z_list_all_modules_files

# def plot_3D_igraph_single_modules(community_vect,coomatrix,node_coords = np.array([]),node_labels = [],nb_min_nodes_by_module = 5):

    #import collections

    #dist_com = collections.Counter(community_vect)

    # print dist_com

    # if (community_vect.shape[0] != node_coords.shape[0]):
    # print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])

    # print community_vect.shape
    # print node_coords.shape
    # print coomatrix.shape

    # threshoding the number of dictictly displayed modules with the number of igraph colors

    #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1

    # print np.unique(community_vect)

    #Z_list_all_modules_files = []

    # extract edge list (with coords belonging to )

    #g_all = ig.Graph(zip(coomatrix.row, coomatrix.col), directed=False, edge_attrs={'weight': coomatrix.data})

    # print g_all

    # for mod_index in np.unique(community_vect):

    # print "Module index %d has %d nodes"%(mod_index,np.sum(community_vect == mod_index))

    # if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:

    # print "Not enough nodes (%d), skipping plot"%(np.sum(community_vect == mod_index))
    # continue

    #g_sel = g_all.copy()

    #edge_mod_id = []
    #edge_col_intra = []

    # print g_sel

    # for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):

    # if (community_vect[u] == community_vect[v] and community_vect[u] == mod_index):

    # edge_col_intra.append(igraph_colors[community_vect[u]])
    # else:

    #eid = g_sel.get_eid(u,v)

    # edge_mod_id.append(eid)

    # g_sel.delete_edges(edge_mod_id)

    #g_sel.es['color'] = edge_col_intra

    # node colors

    #vertex_col = []

    # for i,v in enumerate(g_sel.vs):
    #cur_mod_index = community_vect[i]

    # if (mod_index == cur_mod_index):
    # vertex_col.append(igraph_colors[mod_index])
    # else:
    # vertex_col.append("black")

    #g_sel.vs['color'] = vertex_col

    # node_labels
    # add_non_null_labels(g_sel,node_labels)

    # single view
    #view = [0.0,0.0]

    #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + "_from_left.eps")

    #layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])

    #g_sel['layout'] = layout2D.tolist()

    ##ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = 0.1, edge_curved = False)
    ##ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
    #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)

    # Z_list_all_modules_files.append(Z_list_all_modules_file)

    # all view
    ##views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]

    ##suf = ["_from_left","_from_front","_from_top","_from_behind"]

    ##Z_list_all_modules_files = []

    # for i,view in enumerate(views):

    # print view

    ##Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + suf[i] + ".eps")

    ##layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])

    ##g_sel['layout'] = layout2D.tolist()

    ###ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
    ##ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)

    # Z_list_all_modules_files.append(Z_list_all_modules_file)

    # return Z_list_all_modules_files


def plot_3D_igraph_single_modules_coomatrix_rel_coords(community_vect, node_rel_coords, coomatrix, node_labels=[], nb_min_nodes_by_module=100):

    import collections

    dist_com = collections.Counter(community_vect)

    print(dist_com)

    if (community_vect.shape[0] != node_rel_coords.shape[0]):
        print("Warning, community_vect {} != node_rel_coords {}".format(
            community_vect.shape[0], node_rel_coords.shape[0]))

    print(community_vect.shape)
    print(node_rel_coords.shape)
    print(coomatrix.shape)

    # threshoding the number of dictictly displayed modules with the number of igraph colors

    community_vect[community_vect > len(
        igraph_colors)-1] = len(igraph_colors)-1

    print(np.unique(community_vect))

    Z_list_all_modules_files = []

    # extract edge list (with coords belonging to )

    g_all = ig.Graph(list(zip(coomatrix.row, coomatrix.col)),
                     directed=False, edge_attrs={'weight': coomatrix.data})

    # node_labels
    add_non_null_labels(g_all, node_labels)

    print(g_all)

    for mod_index in np.unique(community_vect):

        print("Module index %d has %d nodes" %
              (mod_index, np.sum(community_vect == mod_index)))

        if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:

            print("Not enough nodes (%d), skipping plot" %
                  (np.sum(community_vect == mod_index)))
            continue

        g_sel = g_all.copy()

        edge_mod_id = []
        edge_col_intra = []

        print(g_sel)

        for u, v, w in zip(coomatrix.row, coomatrix.col, coomatrix.data):

            if (community_vect[u] == community_vect[v] and community_vect[u] == mod_index):

                edge_col_intra.append(igraph_colors[community_vect[u]])
            else:

                eid = g_sel.get_eid(u, v)

                edge_mod_id.append(eid)

        g_sel.delete_edges(edge_mod_id)

        g_sel.es['color'] = edge_col_intra

        # node colors

        vertex_col = []

        for i, v in enumerate(g_sel.vs):
            cur_mod_index = community_vect[i]

            if (mod_index == cur_mod_index):
                vertex_col.append(igraph_colors[mod_index])
            else:
                vertex_col.append("black")

        g_sel.vs['color'] = vertex_col

        # single view
        view = [0.0, 0.0]

        Z_list_all_modules_file = os.path.abspath(
            "module_" + str(mod_index) + "_from_left.eps")

        layout2D = project2D_np(
            node_rel_coords, angle_alpha=view[0], angle_beta=view[1])

        g_sel['layout'] = layout2D.tolist()

        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = 0.1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        ig.plot(g_sel, Z_list_all_modules_file, vertex_size=1,
                edge_width=0.1, edge_curved=False)

        Z_list_all_modules_files.append(Z_list_all_modules_file)

        # all view
        #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]

        #suf = ["_from_left","_from_front","_from_top","_from_behind"]

        #Z_list_all_modules_files = []

        # for i,view in enumerate(views):

        # print view

        #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + suf[i] + ".eps")

        #layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])

        #g_sel['layout'] = layout2D.tolist()

        ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)

        # Z_list_all_modules_files.append(Z_list_all_modules_file)

    return Z_list_all_modules_files
