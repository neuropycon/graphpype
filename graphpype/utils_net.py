# -*- coding: utf-8 -*-
"""
Support function for net handling
"""

import time

import numpy as np
import scipy.sparse as sp


# return cor list (raw) with integer values ( = float * 1000)

def return_net_list(Z_cor_mat):

    t2 = time.time()

    print(Z_cor_mat.shape)

    x_sig, y_sig = np.where(Z_cor_mat != 0.0)

    net_list = np.array(np.column_stack(
        (x_sig + 1, y_sig + 1, Z_cor_mat[x_sig, y_sig]*1000)), dtype=int)

    print(net_list.shape)

    t3 = time.time()

    print("Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s")

    return net_list


def return_int_net_list(int_mat, min_int=0):

    t2 = time.time()

    x_sig, y_sig = np.where(int_mat > min_int)

    net_list = np.array(np.column_stack(
        (x_sig + 1, y_sig + 1, int_mat[x_sig, y_sig])), dtype='int64')

    print(net_list.shape)

    t3 = time.time()

    print("Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s")

    return net_list

##################################### Formatting data for external community detection algorithm (radatools) ##############################


def export_List_net_from_list(Z_List_file, Z_list):

    print(Z_list.shape)

    # print "saving file " + Z_List_file

    np.savetxt(Z_List_file, Z_list, fmt='%d %d %d')


# def read_List_net_file(Z_List_file):

    #Z_list = np.loadtxt(Z_List_file,dtype = 'int64')

    # return Z_list

# def read_mod_file(mod_file):

    #community_vect = np.loadtxt(mod_file,dtype = 'int', delimiter = '\t')

    # return community_vect[:,1]

def read_lol_file(lol_file):

    with open(lol_file, 'r') as f:

        lines = f.readlines()[4:]

        line_nb_elements = lines[0]

        nb_elements = int(line_nb_elements.split(': ')[1])

        # print nb_elements

        community_vect = np.empty((nb_elements), dtype=int)

        lines = lines[3:]

        # print lines

        for i, line in enumerate(lines):

            try:
                nb_nodes, index_nodes = line.split(': ')
                print(nb_nodes, index_nodes)

                if int(nb_nodes) > 1:
                    index_nodes = np.array(
                        list(map(int, index_nodes.split(' '))), dtype=int) - 1

                    # print i,index_nodes
                    community_vect[index_nodes] = i

                    
                else :
                    community_vect[int(index_nodes) -1] = i
               
            except ValueError:
                print("Warning, reading lol file ")

        f.close()

    return community_vect


def read_Pajek_corres_nodes(Pajek_net_file):

    with open(Pajek_net_file, 'r') as f:

        lines = f.readlines()

        line_nb_elements = lines[0]

        nb_elements = int(line_nb_elements.split(' ')[1])

        # print nb_elements

        node_corres = np.empty((nb_elements), dtype='int')

        lines = lines[1:(nb_elements+1)]

        # print lines

        for i, line in enumerate(lines):
            # print line

            new_index, old_index = line.split(' ')

            # print i+1, new_index, old_index

            #node_corres[i] = old_index

            if (i+1) == int(new_index):
                node_corres[i] = int(old_index)-1
            else:
                print("Warning, incompatible indexes {} {}".format(new_index, i+1))

        f.close()

    return node_corres

# from modified Pajek file, read coords

# def read_Pajek_rel_coords(Pajek_net_file):

    # with open(Pajek_net_file,'r') as f :

    #lines = f.readlines()

    #line_nb_elements = lines[0]

    #nb_elements = int(line_nb_elements.split(' ')[1])

    # print nb_elements

    #node_rel_coords = np.empty((nb_elements,3),dtype = 'float')

    #node_lines = lines[1:(nb_elements+1)]

    # print lines

    # for i,line in enumerate(node_lines):
    # print line

    #node_line = line.split(' ')

    # print node_line

    #node_rel_coords[i,0] = node_line[2]
    #node_rel_coords[i,1] = node_line[3]
    #node_rel_coords[i,2] = node_line[4]

    # print node_rel_coords[i,:]

    # f.close()

    # return node_rel_coords

    # " return corres_nodes and sparse matrix from pajek file


def read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file):

    with open(Pajek_net_file, 'r') as f:

        lines = f.readlines()

        line_nb_elements = lines[0]

        nb_elements = int(line_nb_elements.split(' ')[1])

        print(nb_elements)

        node_corres = np.empty((nb_elements), dtype='int')

        node_lines = lines[1:(nb_elements+1)]

        # print lines

        for i, line in enumerate(node_lines):
            # print line

            new_index, old_index = line.split(' ')

            # print i+1, new_index, old_index

            #node_corres[i] = old_index

            if (i+1) == int(new_index):
                node_corres[i] = int(old_index)-1
            else:
                print("Warning, incompatible indexes {} {}".format(new_index, i+1))

        list_sparse_matrix = [line.strip().split(' ')
                              for line in lines[(nb_elements+2):]]

        np_list_sparse_matrix = np.transpose(
            np.array(list_sparse_matrix, dtype='int64'))

        sparse_matrix = sp.coo_matrix((np_list_sparse_matrix[2, :], (
            np_list_sparse_matrix[0, :]-1, np_list_sparse_matrix[1, :]-1)), shape=(nb_elements, nb_elements))

        f.close()

    return node_corres, sparse_matrix

# " compute modular Network


def compute_modular_network(sparse_matrix, community_vect):

    mod_mat = np.empty(sparse_matrix.todense().shape)

    mod_mat[:] = np.NAN

    for u, v, w in zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):

        if (community_vect[u] == community_vect[v]):

            mod_mat[u, v] = community_vect[u]
        else:

            mod_mat[u, v] = -1

    return mod_mat

# read strength from Network_Properties node results


def get_strength_values_from_info_nodes_file(info_nodes_file):

    from pandas.io.parsers import read_csv

    info_nodes = read_csv(info_nodes_file, sep="\t")

    # print info_nodes

    return info_nodes['Strength'].values


def get_strength_pos_values_from_info_nodes_file(info_nodes_file):

    from pandas.io.parsers import read_csv

    info_nodes = read_csv(info_nodes_file, sep="\t")

    print(info_nodes)

    return info_nodes['Strength_Pos']


def get_strength_neg_values_from_info_nodes_file(info_nodes_file):

    from pandas.io.parsers import read_csv

    info_nodes = read_csv(info_nodes_file, sep="\t")

    print(info_nodes)

    return info_nodes['Strength_Neg']


def get_degree_pos_values_from_info_nodes_file(info_nodes_file):

    from pandas.io.parsers import read_csv

    info_nodes = read_csv(info_nodes_file, sep="\t")

    print(info_nodes)

    return info_nodes['Degree_Pos']


def get_degree_neg_values_from_info_nodes_file(info_nodes_file):

    from pandas.io.parsers import read_csv

    info_nodes = read_csv(info_nodes_file, sep="\t")

    print(info_nodes)

    return info_nodes['Degree_Neg']

##################################### Formatting data for external community detection algorithm (Louvain_Traag) ##############################


def export_Louvain_net_from_list(Z_Louvain_file, Z_list, coords):

    print(np.array(Z_list).shape)

    # print sig_x,sig_y
    print("column_stack")
    tab_edges = np.column_stack(
        (np.array(Z_list), np.repeat(1, repeats=len(Z_list))))

    print(tab_edges.shape)

    print("file")

    with open(Z_Louvain_file, 'w') as f:

        # write node list
        nb_nodes = coords.shape[0]

        print("Nb nodes: " + str(nb_nodes))

        coords_list = coords.tolist()

        f.write('>\n')

        for node in range(nb_nodes):

            # print node + coord label
            f.write(str(node+1) + ' ' +
                    '_'.join(map(str, coords_list[node])) + '\n')

        # write slice list
        f.write('>\n')
        f.write('1 1\n')

        # write edge list
        f.write('>\n')
        np.savetxt(f, tab_edges, fmt='%d %d %d %d')