
import numpy as np
import pandas as pd

from visbrain.brain.brain import Brain
from visbrain.objects import SourceObj, ConnectObj

from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
from graphpype.utils_net import read_lol_file, compute_modular_matrix


c_colval_modules = {0: "red", 1: "orange", 2: "blue", 3: "green", 4: "yellow",
                    5: "darksalmon"}

c_colval_signif = {4: "darkred", 3: "red", 2: "orange", 1: "yellow", -1:
                   "cyan", -2: "cornflowerblue", -3: "blue", -4: "navy"}


def visu_graph_modules(net_file, lol_file, coords_file, labels_file,
                       inter_modules=True, modality_type="",
                       s_textcolor="white", c_colval=c_colval_modules,
                       umin=0, umax=50):

    # coords
    coords = np.loadtxt(coords_file)
    if modality_type == "MEG":
        coords = 1000*coords

    # labels
    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)

    # net file
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    # lol file
    community_vect = read_lol_file(lol_file)

    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[c_connect > -1.0] = False

    # Colormap properties (for connectivity) :
    # c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    if inter_modules:
        c_colval[-1] = "grey"

    # new to visbrain 0.3.7
    from visbrain.objects import SourceObj, ConnectObj
    s_obj = SourceObj('SourceObj1', corres_coords, text=newLabels,
                      text_color=s_textcolor, color='crimson', alpha=.5,
                      edge_width=2., radius_min=2., radius_max=10.)

    """Create the connectivity object :"""
    c_obj = ConnectObj('ConnectObj1', coords, c_connect, color_by='strength',
                       custom_colors=c_colval)  # , antialias=True

    vb = Brain(source_obj=s_obj, connect_obj=c_obj)

    vb.show()


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

    vb = Brain(source_obj=s_obj, connect_obj=c_obj)
    vb.show()


"""
def visu_graph_modules_roles(
        net_file, lol_file, coords_file, labels_file, node_roles_file,
        modality_type="", inter_modules=True, c_colval={0: "red", 1: 
        "lightblue", 2: "yellow", 3: "green", 4: "purple"},         
s_textcolor="black", only_right=False, umin=0, umax=50, hub_size=15.0,         
non_hub_size=5.0):

    # coords
    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords

    # labels
    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)

    # net file
    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    # lol file
    community_vect = read_lol_file(lol_file)


    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')
    
    # node roles:
    node_roles = np.array(np.loadtxt(node_roles_file), dtype='int64')


    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[np.where((c_connect > umin) & (c_connect < umax))] = False

    print(c_connect)

    # Colormap properties (for connectivity) :
    c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    if only_right:
        
        list_areas = ["amygdala", "Precuneus","vPCC", "dPCC", "Ip"]
        
        only_R_labels = []

        for lab in newLabels.tolist():
            if lab.endswith("_R") and if not lab.split("_")[0] in list_areas:
                only_R_labels.append(lab)
            else:
                only_R_labels.append("")

        newLabels = np.array(only_R_labels, dtype="str")

    if inter_modules:
        c_colval[-1] = "grey"

    coords_prov_hubs = corres_coords[(
        node_roles[:, 0] == 1) & (node_roles[:, 1] == 2)]
    coords_prov_no_hubs = corres_coords[(
        node_roles[:, 0] == 1) & (node_roles[:, 1] == 1)]

    coords_connec_hubs = corres_coords[(
        node_roles[:, 0] == 2) & (node_roles[:, 1] == 2)]
    coords_connec_no_hubs = corres_coords[(
        node_roles[:, 0] == 2) & (node_roles[:, 1] == 1)]

    print(coords_prov_no_hubs)

    # version add sources
    vb = Brain(
        s_xyz=corres_coords,  s_text=newLabels, s_textsize=2, 
        s_textcolor=s_textcolor, s_textshift=(1.5, 1.8, 3), s_opacity=0., 
        s_radiusmin=0., s_radiusmax=0.1)

    if coords_prov_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_hubs', s_xyz=coords_prov_hubs, 
                       s_symbol='disc',
                       s_color='orange', s_edgecolor='black',
                       
s_data=np.ones(shape=(coords_prov_hubs.shape[0]))*hub_size, 
s_radiusmin=hub_size, s_radiusmax=hub_size+1.0, s_opacity=1.,)# noqa  

    if coords_prov_no_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_no_hubs', s_xyz=coords_prov_no_hubs, 
s_symbol='disc',# noqa  
                       s_color='orange', s_edgecolor='black',
                       
s_data=np.ones(shape=(coords_prov_no_hubs.shape[0]))*non_hub_size, 
s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1, s_opacity=1.0)# noqa  

    if coords_connec_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_hubs', s_xyz=coords_connec_hubs, 
s_symbol='square',# noqa  
                       s_color='orange', s_edgecolor='black',
                       
s_data=np.ones(shape=(coords_connec_hubs.shape[0]))*hub_size, 
s_radiusmin=hub_size, s_radiusmax=hub_size+1.0, s_opacity=1.,)# noqa  

    if coords_connec_no_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_no_hubs', s_xyz=coords_connec_no_hubs, 
s_symbol='square',# noqa  
                       s_color='orange', s_edgecolor='black',
                       
s_data=np.ones(shape=(coords_connec_no_hubs.shape[0]))*non_hub_size, 
s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1.0, s_opacity=1.0) # noqa  
  

    # add connectivity objects
    vb.add_connect('connect_L', c_xyz=corres_coords, c_connect=c_connect,
                   c_cmap=c_cmap, c_linewidth=4., c_colval=c_colval,
                   c_dynamic=(.1, 1.))

    vb.show()


def visu_graph(net_file, coords_file, labels_file, modality_type="fMRI",
               s_textcolor="black", c_colval={1: "orange"}):
    # coords
    coords = np.loadtxt(coords_file)
    if modality_type == "MEG":
        coords = 1000*coords

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

    vb = Brain(s_xyz=corres_coords, s_text=newLabels, s_textsize=2,
               s_textcolor=s_textcolor, c_connect=c_connect, c_colval=c_colval)
    vb.show()
"""
