import numpy as np
import pandas as pd

from visbrain.gui.brain.brain import Brain
from visbrain.objects import SourceObj, ConnectObj

from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
from graphpype.utils_mod import read_lol_file, compute_modular_matrix


def visu_graph(net_file, coords_file, labels_file, modality_type="fMRI",
               s_textcolor="black", c_colval={1: "orange"}):
    # coords
    coords = np.loadtxt(coords_file)
    if modality_type == "MEG":
        coords = 1000*coords
        coords = np.swapaxes(coords, 0, 1)

    # labels
    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)

    # net file
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    c_connect = np.array(sparse_matrix.todense())
    c_connect[c_connect != 0] = 1

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    # new to visbrain 0.3.7
    s_obj = SourceObj('SourceObj1', corres_coords, text=newLabels,
                      text_color="white", color='crimson', alpha=.5,
                      edge_width=2., radius_min=2., radius_max=10.)

    """Create the connectivity object :"""
    c_obj = ConnectObj('ConnectObj1', corres_coords, c_connect,
                       color_by='strength', custom_colors=c_colval)
    # ,antialias=True
    vb = Brain(source_obj=s_obj, connect_obj=c_obj)
    return vb


c_colval_signif = {4: "darkred", 3: "red", 2: "orange", 1: "yellow", -1:
                   "cyan", -2: "cornflowerblue", -3: "blue", -4: "navy"}


def visu_graph_signif(indexed_mat_file, coords_file, labels_file,
                      c_colval=c_colval_signif):

    # coords

    coords = np.loadtxt(coords_file)

    # labels
    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    # print(npLabels)

    # net file

    # indexed_mat file
    if indexed_mat_file.endswith(".csv"):

        indexed_mat = pd.read_csv(indexed_mat_file, index_col=0).values

    elif indexed_mat_file.endswith(".npy"):

        indexed_mat = np.load(indexed_mat_file)

    indexed_mat = np.load(indexed_mat_file)
    print(indexed_mat[indexed_mat != 0])

    for i in range(indexed_mat.shape[0]):
        if np.sum(indexed_mat[i, :] == 0) == indexed_mat.shape[1]:
            npLabels[i] = ""

    c_connect = np.ma.masked_array(indexed_mat, mask=indexed_mat == 0)

    # new to visbrain 0.3.7
    s_obj = SourceObj('SourceObj1', coords, text=npLabels, text_color="white",
                      color='crimson', alpha=.5, edge_width=2., radius_min=2.,
                      radius_max=10.)

    """Create the connectivity object :"""
    c_obj = ConnectObj('ConnectObj1', coords, c_connect, color_by='strength',
                       custom_colors=c_colval)  # , antialias=True

    return Brain(source_obj=s_obj, connect_obj=c_obj)


c_colval_modules = {0: "red", 1: "orange", 2: "blue", 3: "green", 4: "yellow",
                    5: "darksalmon", 6: "brown", 7: "black"}


def visu_graph_modules(net_file, lol_file, coords_file, labels_file=0,
                       inter_modules=True, modality_type="",
                       s_textcolor="white", c_colval=c_colval_modules,
                       umin=0, umax=50, x_offset=0, y_offset=0,
                       z_offset=0):
    # coords
    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords
        temp = np.copy(coords)
        coords[:, 1] = coords[:, 0]
        coords[:, 0] = temp[:, 1]

    coords[:, 2] += z_offset
    coords[:, 1] += y_offset
    coords[:, 0] += x_offset

    # labels
    if labels_file:
        labels = [line.strip() for line in open(labels_file)]
        np_labels = np.array(labels)

    # net file
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    # lol file
    community_vect = read_lol_file(lol_file)

    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[c_connect > -1.0] = False

    corres_coords = coords[node_corres, :]

    if inter_modules:
        c_colval[-1] = "grey"

    """Create the connectivity object :"""
    c_obj = ConnectObj('ConnectObj1', corres_coords, c_connect,
                       color_by='strength', custom_colors=c_colval)

    # source object
    colors_nodes = []
    for i in community_vect:
        if i in c_colval.keys():
            colors_nodes.append(c_colval[i])
        else:
            colors_nodes.append("black")

    if labels_file:
        corres_labels = np_labels[node_corres]
        s_obj = SourceObj(
            'SourceObj1', corres_coords, text=corres_labels,
            text_color=s_textcolor, text_size=10, color=colors_nodes, alpha=.5,
            edge_width=2., radius_min=10., radius_max=10.)

    else:
        s_obj = SourceObj(
            'SourceObj1', corres_coords, text_color=s_textcolor, text_size=10,
            color=colors_nodes, alpha=.5, edge_width=2., radius_min=10.,
            radius_max=10.)

    return c_obj, s_obj


def visu_graph_modules_roles(net_file, lol_file, roles_file, coords_file,
                             inter_modules=True, modality_type="",
                             s_textcolor="white", c_colval=c_colval_modules,
                             umin=0, umax=50, x_offset=0, y_offset=0,
                             z_offset=0, default_size=10, hub_to_non_hub=3):

    # coords
    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords
        temp = np.copy(coords)
        coords[:, 1] = coords[:, 0]
        coords[:, 0] = temp[:, 1]

    coords[:, 2] += z_offset
    coords[:, 1] += y_offset
    coords[:, 0] += x_offset

    # net file
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)
    corres_coords = coords[node_corres, :]

    corres_coords = coords[node_corres, :]

    # lol file
    community_vect = read_lol_file(lol_file)

    max_col = np.array([val for val in c_colval.keys()]).max()
    community_vect[community_vect > max_col] = max_col

    """Create the connectivity object :"""
    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[-1.0 <= c_connect] = False

    if inter_modules:
        c_colval[-1] = "grey"

    else:
        c_connect.mask[-1.0 == c_connect] = True

    c_obj = ConnectObj('ConnectObj1', corres_coords, c_connect,
                       color_by='strength', custom_colors=c_colval)

    """Create the source object :"""
    node_roles = np.array(np.loadtxt(roles_file), dtype='int64')

    # prov_hubs
    prov_hubs = (
        node_roles[:, 0] == 2) & (node_roles[:, 1] == 1)
    coords_prov_hubs = corres_coords[prov_hubs]
    colors_prov_hubs = [c_colval[i] for i in community_vect[prov_hubs]]

    # prov_no_hubs
    prov_no_hubs = (
        node_roles[:, 0] == 1) & (node_roles[:, 1] == 1)
    coords_prov_no_hubs = corres_coords[prov_no_hubs]
    colors_prov_no_hubs = [c_colval[i] for i in community_vect[prov_no_hubs]]

    # connec_hubs
    connec_hubs = (
        node_roles[:, 0] == 2) & (node_roles[:, 1] == 2)
    coords_connec_hubs = corres_coords[connec_hubs]
    colors_connec_hubs = [c_colval[i] for i in community_vect[connec_hubs]]

    # connec_no_hubs
    connec_no_hubs = (
        node_roles[:, 0] == 1) & (node_roles[:, 1] == 2)
    coords_connec_no_hubs = corres_coords[connec_no_hubs]
    colors_connec_no_hubs = [
        c_colval[i] for i in community_vect[connec_no_hubs]]

    list_sources = []

    if len(coords_prov_no_hubs != 0):
        s_obj1 = SourceObj(
            'prov_no_hubs', coords_prov_no_hubs, color=colors_prov_no_hubs,
            alpha=.5, edge_width=2., radius_min=default_size,
            radius_max=default_size)

        list_sources.append(s_obj1)

    if len(coords_connec_no_hubs != 0):
        s_obj2 = SourceObj(
            'connec_no_hubs', coords_connec_no_hubs,
            color=colors_connec_no_hubs, alpha=.5, edge_width=2.,
            radius_min=default_size, radius_max=default_size, symbol='square')

        list_sources.append(s_obj2)

    if len(coords_prov_hubs != 0):
        s_obj3 = SourceObj(
            'prov_hubs', coords_prov_hubs, color=colors_prov_hubs, alpha=.5,
            edge_width=2., radius_min=default_size*hub_to_non_hub,
            radius_max=default_size*hub_to_non_hub)

        list_sources.append(s_obj3)

    if len(coords_connec_hubs != 0):
        s_obj4 = SourceObj(
            'connec_hubs', coords_connec_hubs, color=colors_connec_hubs,
            alpha=.5, edge_width=2., radius_min=default_size*hub_to_non_hub,
            radius_max=default_size*hub_to_non_hub, symbol='square')

        list_sources.append(s_obj4)

    return c_obj, list_sources
