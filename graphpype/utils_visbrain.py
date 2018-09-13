
import numpy as np
import pandas as pd

from visbrain.brain.brain import Brain

from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
from graphpype.utils_net import read_lol_file, compute_modular_matrix


def visu_graph_modules_roles(net_file, lol_file, coords_file, labels_file, node_roles_file, modality_type="", inter_modules=True,  c_colval={0: "red", 1: "lightblue", 2: "yellow", 3: "green", 4: "purple"}, s_textcolor="black", only_right=False):

    # coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords

    print("coords: ", end=' ')
    print(coords)

    # labels
    #label_file = os.path.abspath("data/MEG/label_names.txt")

    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print(npLabels)

    # net file

    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    print(sparse_matrix)

    print(node_corres)

    print(node_corres.shape)

    # lol file
    community_vect = read_lol_file(lol_file)

    print(community_vect)

    print(np.unique(community_vect))

    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')
    print(c_connect.shape)

    #data = np.load('RealDataExample.npz')

    #s_data = data['beta']
    #s_xyz = data['xyz']

    # node roles:
    node_roles = np.array(np.loadtxt(node_roles_file), dtype='int64')

    # print node_roles

    umin = 0.0

    umax = 50

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[np.where((c_connect > umin) & (c_connect < umax))] = False

    print(c_connect)

    # Colormap properties (for connectivity) :
    c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    if only_right:

        only_R_labels = []

        for lab in newLabels.tolist():
            if lab.endswith("_R") and not lab.split("_")[0] in ["amygdala", "Precuneus", "vPCC", "dPCC", "Ip"]:
                only_R_labels.append(lab)
            else:
                only_R_labels.append("")

        newLabels = np.array(only_R_labels, dtype="str")
        print(newLabels)

    #c_colval = {0:"red",1:"darksalmon",2:"blue",3:"green",4:"yellow",5:"orange"}

    #c_colval = {0:"red",1:"lightblue",2:"blue",3:"yellow",4:"purple"}

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

    ################################## version add sources #########################

    #s_text=newLabels, s_textsize = 2,s_textcolor="white" #

    hub_size = 15.0
    non_hub_size = 5.0

    vb = Brain(s_xyz=corres_coords,  s_text=newLabels, s_textsize=2, s_textcolor=s_textcolor,
               s_textshift=(1.5, 1.8, 3), s_opacity=0., s_radiusmin=0., s_radiusmax=0.1)

    if coords_prov_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_hubs', s_xyz=coords_prov_hubs, s_symbol='disc',
                       s_color='orange', s_edgecolor='black',
                       s_data=np.ones(shape=(coords_prov_hubs.shape[0]))*hub_size, s_radiusmin=hub_size, s_radiusmax=hub_size+1.0, s_opacity=1.,)

    if coords_prov_no_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_no_hubs', s_xyz=coords_prov_no_hubs, s_symbol='disc',
                       s_color='orange', s_edgecolor='black',
                       s_data=np.ones(shape=(coords_prov_no_hubs.shape[0]))*non_hub_size, s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1, s_opacity=1.0)

    if coords_connec_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_hubs', s_xyz=coords_connec_hubs, s_symbol='square',
                       s_color='orange', s_edgecolor='black',
                       s_data=np.ones(shape=(coords_connec_hubs.shape[0]))*hub_size, s_radiusmin=hub_size, s_radiusmax=hub_size+1.0, s_opacity=1.,)

    if coords_connec_no_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_no_hubs', s_xyz=coords_connec_no_hubs, s_symbol='square',
                       s_color='orange', s_edgecolor='black',
                       s_data=np.ones(shape=(coords_connec_no_hubs.shape[0]))*non_hub_size, s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1.0, s_opacity=1.0)

    # ================ ADD CONNECTIVITY OBJECTS ================

    vb.add_connect('connect_L', c_xyz=corres_coords, c_connect=c_connect,
                   c_cmap=c_cmap, c_linewidth=4., c_colval=c_colval,
                   c_dynamic=(.1, 1.))

    vb.show()


def visu_graph_modules(net_file, lol_file, coords_file, labels_file, inter_modules=True, modality_type="", s_textcolor="white", c_colval={0: "red", 1: "orange", 2: "blue", 3: "green", 4: "yellow", 5: "darksalmon"}):

    # coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords

    print("coords: ", end=' ')
    print(coords)

    # labels
    #label_file = os.path.abspath("data/MEG/label_names.txt")

    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print(npLabels)

    # net file

    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    print(sparse_matrix)

    print(node_corres)

    print(node_corres.shape)

    # lol file
    community_vect = read_lol_file(lol_file)

    print(community_vect)

    c_connect = np.array(compute_modular_matrix(
        sparse_matrix, community_vect), dtype='float64')
    print(c_connect.shape)

    #data = np.load('RealDataExample.npz')

    #s_data = data['beta']
    #s_xyz = data['xyz']

    umin = 0.0

    umax = 50.0

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[c_connect > -1.0] = False

    print(c_connect)

    # Colormap properties (for connectivity) :
    c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    #c_colval = {-1:"grey",0:"red",1:"orange",2:"blue",3:"green",4:"yellow",5:"purple"}

    if inter_modules:
        c_colval[-1] = "grey"

    #vb = Brain(s_xyz=corres_coords, s_text=newLabels, s_textsize = 2,s_textcolor=s_textcolor, c_map=c_cmap, c_connect = c_connect, c_colval = c_colval)
    # vb.show()

    # new to visbrain 0.3.7

    from visbrain.objects import SourceObj, ConnectObj

    s_obj = SourceObj('SourceObj1', corres_coords, text=newLabels, text_color=s_textcolor, color='crimson', alpha=.5,
                      edge_width=2., radius_min=2., radius_max=10.)

    #umin, umax = 30, 31

    # 1 - Using select (0: hide, 1: display):
    #select = np.zeros_like(connect)
    #select[(connect > umin) & (connect < umax)] = 1

    # 2 - Using masking (True: hide, 1: display):
    #connect = np.ma.masked_array(connect, mask=True)
    #connect.mask[np.where((connect > umin) & (connect < umax))] = False

    #print('1 and 2 equivalent :', np.array_equal(select, ~connect.mask + 0))

    """Create the connectivity object :
    """
    c_obj = ConnectObj('ConnectObj1', corres_coords, c_connect,
                       color_by='strength', cmap=c_cmap)  # , antialias=True

    vb = Brain(source_obj=s_obj, connect_obj=c_obj)

    #vb = Brain(s_xyz=corres_coords, s_text=newLabels, s_textsize = 2,s_textcolor=s_textcolor, c_map=c_cmap, c_connect = c_connect, c_colval = c_colval)

    vb.show()


def visu_graph_signif(indexed_mat_file, coords_file, labels_file, c_colval={4: "darkred", 3: "red", 2: "orange", 1: "yellow", -1: "cyan", -2: "cornflowerblue", -3: "blue", -4: "navy"}):

    # coords

    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)

    #print("coords: ", end=' ')
    print("coords: ")

    print(coords)

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

    from visbrain.objects import SourceObj, ConnectObj

    s_obj = SourceObj('SourceObj1', coords, text=npLabels, text_color="white", color='crimson', alpha=.5,
                      edge_width=2., radius_min=2., radius_max=10.)

    """Create the connectivity object :
    """
    c_obj = ConnectObj('ConnectObj1', coords, c_connect,
                       color_by='strength',  custom_colors=c_colval)  # , antialias=True

    vb = Brain(source_obj=s_obj, connect_obj=c_obj)
    vb.show()


def visu_graph(net_file, coords_file, labels_file, modality_type="fMRI", s_textcolor="black"):

    # coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)

    if modality_type == "MEG":
        coords = 1000*coords

    print("coords: ", end=' ')
    print(coords)

    # labels
    #label_file = os.path.abspath("data/MEG/label_names.txt")

    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print(npLabels)

    # net file

    node_corres, sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(
        net_file)

    print(sparse_matrix.shape)
    print(sparse_matrix.todense())

    c_connect = np.array(sparse_matrix.todense())

    print(node_corres.shape)

    c_connect[c_connect != 0] = 1

    corres_coords = coords[node_corres, :]
    newLabels = npLabels[node_corres]

    print(corres_coords.shape)
    print(newLabels.shape)

    c_colval = {1: "orange"}

    vb = Brain(s_xyz=corres_coords, s_text=newLabels, s_textsize=2,
               s_textcolor=s_textcolor, c_connect=c_connect, c_colval=c_colval)
    vb.show()
