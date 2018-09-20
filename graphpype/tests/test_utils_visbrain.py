#######################################################################################################################
### test over visbrain plug-ins  ###
#######################################################################################################################

import os

import numpy as np

try:
    import neuropycon_data
except ImportError:
    print ("neuropycon_data is not installed")
    

import neuropycon_data as nd

from graphpype.utils_visbrain import (visu_graph,visu_graph_modules)
    

######### data

data_path = os.path.join(nd.__path__[0], "data", "data_con")

print(data_path)

coords_file = os.path.join(data_path, "ROI_MNI_coords-Atlas.txt")

labels_file = os.path.join(data_path, "ROI_labels-Atlas.txt")

lol_file = os.path.join(data_path, "data_graph", "Z_List.lol")

Pajek_net_file = os.path.join(data_path, "data_graph", "Z_List.net")

######### tests

def test_visu_graph():
    """
    testing if visualisation of graph is available
    """
    visu_graph(net_file = Pajek_net_file, coords_file = coords_file, labels_file = labels_file)
    
def test_visu_graph_modules():
    """
    testing if visualisation of graph + module  is available
    """
    visu_graph_modules(net_file = Pajek_net_file, 
                             lol_file = lol_file, coords_file = coords_file, labels_file = labels_file)
    
############# signif matrices (from .npy files)

if __name__ == '__main__':
    
    test_visu_graph()
    test_visu_graph_modules()
    
