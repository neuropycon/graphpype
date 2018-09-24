# -*- coding: utf-8 -*-
"""
Support function for net handling
"""

import time

import numpy as np
import scipy.sparse as sp


# return cor list (raw) with integer values ( = float * 1000)
def return_net_list(Z_cor_mat, int_factor=1000):

    print(Z_cor_mat.shape)

    x_sig, y_sig = np.where(Z_cor_mat != 0.0)

    net_list = np.array(np.column_stack(
        (x_sig + 1, y_sig + 1, Z_cor_mat[x_sig, y_sig]*int_factor)), dtype=int)

    print(net_list.shape)

    return net_list


def return_int_net_list(int_mat, min_int=0):

    x_sig, y_sig = np.where(int_mat > min_int)

    net_list = np.array(np.column_stack(
        (x_sig + 1, y_sig + 1, int_mat[x_sig, y_sig])), dtype='int64')

    print(net_list.shape)

    return net_list

################################### reading file
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
