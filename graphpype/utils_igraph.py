# -*- coding: utf-8 -*-
"""
Support function for igraph
"""

import numpy as np
import os


import igraph as ig

from .utils_dtype_coord import where_in_coords, find_index_in_coords
import math

from graphpype.utils_color import igraph_colors, new_igraph_colors, static_igraph_colors


def add_vertex_colors(g_all, community_vect, list_colors=static_igraph_colors):

    # threshoding the number of dictictly displayed modules with the number of igraph colors

    community_vect[community_vect > len(list_colors)-1] = len(list_colors)-1

    # extract edge list (with coords belonging to )

    print(np.unique(community_vect))

    vertex_col = []
    vertex_label_col = []

    for i, v in enumerate(g_all.vs):
        mod_index = community_vect[i]

        if (mod_index != len(list_colors)-1):
            vertex_col.append(list_colors[mod_index])
            vertex_label_col.append(list_colors[mod_index])
        else:
            vertex_col.append("lightgrey")
            vertex_label_col.append("lightgrey")

    g_all.vs['color'] = vertex_col
    g_all.vs['label_color'] = vertex_label_col

    return np.unique(np.array(vertex_col, dtype="str"))


def create_module_edge_list(coomatrix, community_vect, list_colors=static_igraph_colors):

    # threshoding the number of dictictly displayed modules with the number of igraph colors

    community_vect[community_vect > len(list_colors)-1] = len(list_colors)-1

    # extract edge list (with coords belonging to )

    print(np.unique(community_vect))

    print(len(list_colors)-1)

    edge_col_inter = []
    edge_list_inter = []
    edge_weights_inter = []

    edge_col_intra = []
    edge_list_intra = []
    edge_weights_intra = []

    for u, v, w in zip(coomatrix.row, coomatrix.col, coomatrix.data):

        if (community_vect[u] == community_vect[v]):

            edge_list_intra.append((u, v))
            edge_weights_intra.append(w)
            edge_col_intra.append(list_colors[community_vect[u]])
        else:

            edge_list_inter.append((u, v))
            edge_weights_inter.append(w)
            edge_col_inter.append("lightgrey")

    edge_list = edge_list_inter + edge_list_intra

    edge_weights = edge_weights_inter + edge_weights_intra

    edge_col = edge_col_inter + edge_col_intra

    g_all = ig.Graph(edge_list)

    g_all.es["weight"] = edge_weights

    g_all.es['color'] = edge_col
    # print g_all

    return g_all, np.unique(np.array(edge_col, dtype="str"))


def select_edge_list_outside_module(g_sel, coomatrix, community_vect, mod_index):

    edge_mod_id = []

    print(g_sel)

    for u, v, w in zip(coomatrix.row, coomatrix.col, coomatrix.data):
        if (community_vect[u] != mod_index or community_vect[v] != mod_index):
            eid = g_sel.get_eid(u, v)
            edge_mod_id.append(eid)

    g_sel.delete_edges(edge_mod_id)

    # return g_sel


######################################## igraph 3D #######################################

def project2D_np(node_coords, angle_alpha=0.0, angle_beta=0.0):

    #node_coords = np.transpose(np.vstack((node_coords[:,1],-node_coords[:,2]*0.5,node_coords[:,0])))
    node_coords = np.transpose(
        np.vstack((node_coords[:, 1], -node_coords[:, 2], node_coords[:, 0])))

    # print node_coords

    # 0/0

    angle_alpha = angle_alpha + 10.0
    angle_beta = angle_beta + 5.0

    print(node_coords.shape)

    #layout2D = project2D(node_coords.tolist(),0,0)
    layout2D = project2D(node_coords.tolist(), np.pi/180 *
                         angle_alpha, np.pi/180*angle_beta)

    #node_coords = np.transpose(np.vstack((node_coords[:,1],node_coords[:,2],node_coords[:,0])))

    # print node_coords.shape

    ##layout2D = project2D(node_coords.tolist(),0,0)
    #layout2D = project2D(node_coords.tolist(),0,0)

    return layout2D


def project2D(layout, alpha, beta):
    '''
    This method will project a set of points in 3D to 2D based on the given
    angles alpha and beta.
    '''
    # Calculate the rotation matrices based on the given angles.
    c = np.matrix([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)],
                   [0, -np.sin(alpha), np.cos(alpha)]])
    c = c * np.matrix([[np.cos(beta), 0, -np.sin(beta)],
                       [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    b = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Hit the layout, rotate, and kill a dimension
    layout = np.matrix(layout)
    #x,y,z = (b * (c * layout.transpose())).transpose()

    X = (b * (c * layout.transpose())).transpose()

    # print X.shape

    proj = [[X[i, 0], X[i, 1], X[i, 2]] for i in range(X.shape[0])]

    # print proj

    x, y, z = list(zip(*proj))

    #graph.vs['x2'], graph.vs['y2'], graph.vs['z2'] = zip(*layout2D)
    minX, maxX = min(x), max(x)
    minY, maxY = min(y), max(y)
    #minZ, maxZ = min(z), max(z)

    print(minX, maxX)
    print(minY, maxY)

    layout2D_x = (x - minX) / (maxX - minX)
    layout2D_y = (y - minY) / (maxY - minY)

    print(layout2D_x)
    print(layout2D_y)

    layout2D = np.transpose(np.vstack((layout2D_x, layout2D_y)))

    # print layout2D.shape

    return layout2D

################################################################# Methode to fill Graph properties ################################################################################


def return_base_weighted_graph(int_matrix):

    mod_list = int_matrix.tolist()

    # print mod_list

    g = ig.Graph.Weighted_Adjacency(mod_list, mode=ig.ADJ_MAX)

    return g


def add_non_null_labels(g, labels=[]):

    null_degree_index, = np.where(np.array(g.degree()) == 0)

    # print null_degree_index

    np_labels = np.array(labels, dtype='str')

    # print np_labels

    np_labels[null_degree_index] = ""

    # print np_labels

    if len(labels) == len(g.vs):

        g.vs['label'] = np_labels.tolist()

        g.vs['label_size'] = 15

        g.vs['label_dist'] = 2

        print(g.vs['label'])

# def  add_edge_colors(g,color_dict)


def add_node_shapes(g_all, node_roles):

    vertex_shape = []
    vertex_size = []

    for i, v in enumerate(g_all.vs):

        # node size
        if node_roles[i, 0] == 1:
            # vertex_size.append(5.0)
            v["size"] = 8.0
            #v["size"] = 5.0
        elif node_roles[i, 0] == 2:
            # vertex_size.append(10.0)
            v["size"] = 15.0
            #v["size"] = 10.0

        ############
        elif node_roles[i, 0] == 0:
            vertex_size.append(10.0)
            v["size"] = 1
            v["shape"] = "cross"

        # shape
        if node_roles[i, 1] == 1:
            v["shape"] = "circle"
            # vertex_shape.append("circle")

        elif node_roles[i, 1] == 2 or node_roles[i, 1] == 5:
            v["shape"] = "rectangle"
            # vertex_shape.append("rectangle")

        elif node_roles[i, 1] == 3 or node_roles[i, 1] == 6:
            v["shape"] = "triangle-up"
            # vertex_shape.append("triangle-up")

        elif node_roles[i, 1] == 4 or node_roles[i, 1] == 7:
            v["shape"] = "triangle-down"
            # vertex_shape.append("triangle-down")

    #g_all.vs["size"] = np.array(vertex_size),
    #g_all.vs["shape"] = vertex_shape
