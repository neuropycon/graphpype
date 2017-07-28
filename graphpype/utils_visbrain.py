
import numpy as np

from visbrain.brain.brain import Brain

from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix

from graphpype.utils_net import read_lol_file,compute_modular_network

def visu_graph_modules_roles(net_file, lol_file, coords_file, labels_file,node_roles_file,modality_type = "",inter_modules = True,  c_colval = {0:"red",1:"lightblue",2:"yellow",3:"green",4:"purple"}):

    ########## coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)
    
    if modality_type == "MEG":
        coords = 1000*coords

    print "coords: ",
    print coords

    ########## labels 
    #label_file = os.path.abspath("data/MEG/label_names.txt")


    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print npLabels

    ##########  net file

    node_corres,sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(net_file)

    print sparse_matrix

    print node_corres

    print node_corres.shape

    ############# lol file
    community_vect = read_lol_file(lol_file)

    print community_vect

    print np.unique(community_vect)
    
    c_connect = np.array(compute_modular_network(sparse_matrix,community_vect),dtype = 'float64')
    print c_connect.shape

    #data = np.load('RealDataExample.npz')

    #s_data = data['beta']
    #s_xyz = data['xyz']

    ############ node roles:
    node_roles = np.array(np.loadtxt(node_roles_file),dtype = 'int64')
           
    #print node_roles
    
    umin = 0.0

    umax = 50

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[np.where((c_connect > umin) & (c_connect < umax))] = False

    print c_connect

    # Colormap properties (for connectivity) :
    c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres,:]
    newLabels = npLabels[node_corres]

    
    #c_colval = {0:"red",1:"darksalmon",2:"blue",3:"green",4:"yellow",5:"orange"}

    #c_colval = {0:"red",1:"lightblue",2:"blue",3:"yellow",4:"purple"}

    if inter_modules:
        c_colval[-1] = "grey"
        
    coords_prov_hubs = corres_coords[(node_roles[:,0] == 1) & (node_roles[:,1] == 2)]
    coords_prov_no_hubs = corres_coords[(node_roles[:,0] == 1) & (node_roles[:,1] == 1)]
    
    
    coords_connec_hubs = corres_coords[(node_roles[:,0] == 2) & (node_roles[:,1] == 2)]
    coords_connec_no_hubs = corres_coords[(node_roles[:,0] == 2) & (node_roles[:,1] == 1)]
    
    print coords_prov_no_hubs
    
    ################################## version add sources #########################

    #s_text=newLabels, s_textsize = 2,s_textcolor="white" # 

    hub_size = 15.0
    non_hub_size = 5.0
    
    vb = Brain(s_xyz=corres_coords,  s_text=newLabels, s_textsize = 2, s_textcolor="white", s_textshift=(1.5, 1.8, 3), s_opacity=0.,s_radiusmin=0., s_radiusmax=0.1)
    
    if coords_prov_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_hubs', s_xyz=coords_prov_hubs, s_symbol='disc',
                s_color='orange', s_edgecolor='black',
                s_data=np.ones(shape = (coords_prov_hubs.shape[0]))*hub_size,s_radiusmin=hub_size, s_radiusmax=hub_size+1.0,s_opacity=1.,)

    if coords_prov_no_hubs.shape[0] != 0:
        vb.add_sources('coords_prov_no_hubs', s_xyz=coords_prov_no_hubs, s_symbol='disc',
                s_color='orange', s_edgecolor='black', 
                s_data=np.ones(shape = (coords_prov_no_hubs.shape[0]))*non_hub_size,s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1, s_opacity=1.0)

    if coords_connec_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_hubs', s_xyz=coords_connec_hubs, s_symbol='square',
                s_color='orange', s_edgecolor='black',
                s_data=np.ones(shape = (coords_connec_hubs.shape[0]))*hub_size,s_radiusmin=hub_size, s_radiusmax=hub_size+1.0,s_opacity=1.,)
        
    if coords_connec_no_hubs.shape[0] != 0:
        vb.add_sources('coords_connec_no_hubs', s_xyz=coords_connec_no_hubs, s_symbol='square',
                s_color='orange', s_edgecolor='black', 
                s_data=np.ones(shape = (coords_connec_no_hubs.shape[0]))*non_hub_size,s_radiusmin=non_hub_size, s_radiusmax=non_hub_size+1.0, s_opacity=1.0)

    # ================ ADD CONNECTIVITY OBJECTS ================

    vb.add_connect('connect_L', c_xyz=corres_coords, c_connect=c_connect,
                c_cmap=c_cmap, c_linewidth=4., c_colval = c_colval,
                c_dynamic=(.1, 1.))

    vb.show()

def visu_graph_modules(net_file, lol_file, coords_file, labels_file,inter_modules = True, modality_type = ""):
    

    ########## coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)
    
    if modality_type == "MEG":
        coords = 1000*coords

    print "coords: ",
    print coords

    ########## labels 
    #label_file = os.path.abspath("data/MEG/label_names.txt")


    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print npLabels

    ##########  net file

    node_corres,sparse_matrix = read_Pajek_corres_nodes_and_sparse_matrix(net_file)

    print sparse_matrix

    print node_corres

    print node_corres.shape

    ############# lol file
    community_vect = read_lol_file(lol_file)

    print community_vect

    c_connect = np.array(compute_modular_network(sparse_matrix,community_vect),dtype = 'float64')
    print c_connect.shape

    #data = np.load('RealDataExample.npz')

    #s_data = data['beta']
    #s_xyz = data['xyz']

    umin = 0.0

    umax = 50

    c_connect = np.ma.masked_array(c_connect, mask=True)
    c_connect.mask[np.where((c_connect > umin) & (c_connect < umax))] = False

    print c_connect

    # Colormap properties (for connectivity) :
    c_cmap = 'inferno'		# Matplotlib colormap

    corres_coords = coords[node_corres,:]
    newLabels = npLabels[node_corres]

    c_colval = {0:"red",1:"orange",2:"blue",3:"green",4:"yellow",5:"darksalmon"}
    #c_colval = {-1:"grey",0:"red",1:"orange",2:"blue",3:"green",4:"yellow",5:"purple"}

    if inter_modules:
        c_colval[-1] = "grey"
        
    vb = Brain(s_xyz=corres_coords, s_text=newLabels, s_textsize = 2,s_textcolor="white", c_map=c_cmap, c_connect = c_connect, c_colval = c_colval)
    vb.show()


def visu_graph(conmat_file,coords_file, labels_file):
    
    ########## coords
    #coord_file = os.path.abspath("data/MEG/label_coords.txt")

    coords = np.loadtxt(coords_file)

    print "coords: ",
    print coords

    ########## labels
    labels = [line.strip() for line in open(labels_file)]
    npLabels = np.array(labels)
    print npLabels

    ##########  net file
    signif_mat = np.load(conmat_file)
    print signif_mat
    
    c_connect = np.ma.masked_array(signif_mat, mask=True)
    #c_connect.mask[np.where((c_connect > umin) & (c_connect < umax))] = False


    print c_connect

    c_colval = {4:"darkred",3:"red",2:"orange",1:"yellow",-1:"cyan",-2:"cornflowerblue",-3:"blue",-4:"navy"}
    
    
    #c_colval = {4:"darkred",3:"red",2:"orange",1:"yellow",-1:"cyan",-2:"cornflowerblue",-3:"blue",-4:"navy"}
    
    #c_colval = {-1:"grey",0:"red",1:"orange",2:"blue",3:"green",4:"yellow",5:"purple"}

    vb = Brain(s_xyz=coords, s_text=npLabels, s_textsize = 1,s_textcolor="white", c_connect = c_connect, c_colval = c_colval)
    vb.show()
