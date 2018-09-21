# -*- coding: utf-8 -*-
"""
Support function for net handling
"""
import sys
import time

import pandas as pd
import numpy as np
import scipy.sparse as sp

# from lol_file


def get_modularity_value_from_lol_file(lol_file):

    with open(lol_file, 'r') as f:

        for line in f.readlines():

            split_line = line.strip().split(' ')

            print(split_line)

            if split_line[0] == 'Q':

                print("Found modularity value line")

                return split_line[2]

        print("Unable to find modularity line in file, returning -1")

        return -1.0

###################################### reading info files #################################################################
# from info-nodes


def get_max_degree_from_node_info_file(info_nodes_file):
    """
    Return max degree AND index and name of max degree (radatools based)  
    """
    df = pd.read_table(info_nodes_file)

    print(df)

    print(df['Degree'].max())

    print(df[df['Degree'] == df['Degree'].max()])

    return df['Degree'].max(), df[df['Degree'] == df['Degree'].max()].Index.values[0], df[df['Degree'] == df['Degree'].max()].Name.values[0]


# ead strength from Network_Properties node results
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

# from info_global


def get_values_from_global_info_file(global_info_file):

    global_values = {}

    with open(global_info_file, 'r') as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            split_line = line.strip().split(' ')

            # print split_line

            if split_line[0] == 'Vertices':

                print("Found Vertices value line")

                print(line.strip().split('\t')[-1])

                global_values['Vertices'] = line.strip().split('\t')[-1]

            elif split_line[0] == 'Edges':

                print("Found Edges value line")

                print(line.strip().split('\t')[-1])

                global_values['Edges'] = line.strip().split('\t')[-1]

            elif split_line[0] == 'Total':

                if split_line[1] == 'degree':

                    print("Found Total degree value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Total_degree'] = line.strip().split(
                        '\t')[-1]

                elif split_line[1] == 'strength':

                    print("Found Total strength value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Total_strength'] = line.strip().split(
                        '\t')[-1]

            elif split_line[0] == 'Average':

                if split_line[1] == 'degree':

                    print("Found Average degree value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Average_degree'] = line.strip().split(
                        '\t')[-1]

                elif split_line[1] == 'strength':

                    print("Found Average strength value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Average_strength'] = line.strip().split(
                        '\t')[-1]

                elif split_line[1] == 'clustering' and split_line[2] == 'coefficient':

                    print("Found Clustering_coeff value line")

                    global_values['Clustering_coeff'] = line.strip().split(
                        '\t')[-1]

                    global_values['Clustering_coeff_weighted'] = lines[i +
                                                                       1].strip().split('\t')[-1]

                    print(line.strip().split('\t')[-1])
                    print(lines[i+1].strip().split('\t')[-1])

            elif split_line[0] == 'Minimum':

                if split_line[1] == 'degree':

                    print("Found Minimum degree value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Minimum_degree'] = line.strip().split(
                        '\t')[-1]

                elif split_line[1] == 'strength':

                    print("Found Minimum strength value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Minimum_strength'] = line.strip().split(
                        '\t')[-1]

            elif split_line[0] == 'Maximum':

                if split_line[1] == 'degree':

                    print("Found Maximum degree value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Maximum_degree'] = line.strip().split(
                        '\t')[-1]

                elif split_line[1] == 'strength':

                    print("Found Maximum strength value line")

                    print(line.strip().split('\t')[-1])

                    global_values['Maximum_strength'] = line.strip().split(
                        '\t')[-1]

            elif split_line[0] == 'Assortativity':

                print("Found Assortativity value line")

                print(line.strip().split('\t')[-1])

                global_values['Assortativity'] = line.strip().split('\t')[-1]
                global_values['Assortativity_weighted'] = lines[i +
                                                                1].strip().split('\t')[-1]

        return global_values


def get_values_from_signed_global_info_file(global_info_file):

    global_values = {}

    with open(global_info_file, 'r') as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            split_line = line.strip().split(' ')

            if split_line[0] == 'Vertices':

                print("Found Vertices value line")

                print(split_line)

                print(line.strip().split('\t')[-1])

                global_values['Vertices'] = line.strip().split('\t')[-1]

            elif split_line[0] == 'Edges':

                print("Found Edges value line")

                print(line.strip().split('\t')[-1])

                global_values['Edges'] = line.strip().split('\t')[-1]

            elif split_line[0] == 'Total' and split_line[1] == 'degree':

                print("Found Total degree value line")

                global_values['Total_degree'] = line.strip().split('\t')[-1]
                global_values['Total_pos_degree'] = lines[i +
                                                          1].strip().split('\t')[-1]
                global_values['Total_neg_degree'] = lines[i +
                                                          2].strip().split('\t')[-1]

            elif split_line[0] == 'Total' and split_line[1] == 'strength':

                print("Found Total strength value line")

                print(line.strip().split('\t')[-1])

                global_values['Total_strength'] = line.strip().split('\t')[-1]
                global_values['Total_pos_strength'] = lines[i +
                                                            1].strip().split('\t')[-1]
                global_values['Total_neg_strength'] = lines[i +
                                                            2].strip().split('\t')[-1]

            elif split_line[0] == 'Average' and split_line[1] == 'degree':

                print("Found Average degree value line")

                print(line.strip().split('\t')[-1])

                global_values['Average_degree'] = line.strip().split('\t')[-1]
                global_values['Average_pos_degree'] = lines[i +
                                                            1].strip().split('\t')[-1]
                global_values['Average_neg_degree'] = lines[i +
                                                            2].strip().split('\t')[-1]

            elif split_line[0] == 'Average' and split_line[1] == 'strength':

                print("Found Average strength value line")

                print(line.strip().split('\t')[-1])

                global_values['Average_strength'] = line.strip().split(
                    '\t')[-1]

                global_values['Average_pos_strength'] = lines[i +
                                                              1].strip().split('\t')[-1]
                global_values['Average_neg_strength'] = lines[i +
                                                              2].strip().split('\t')[-1]

            elif split_line[0] == 'Average' and split_line[1] == 'clustering'\
                    and split_line[2] == 'coefficient':

                print("Found Clustering_coeff value line")

                global_values['Clustering_coeff'] = line.strip().split(
                    '\t')[-1]
                global_values['Clustering_coeff_pos'] = lines[i +
                                                              1].strip().split('\t')[-1]
                global_values['Clustering_coeff_neg'] = lines[i +
                                                              2].strip().split('\t')[-1]

                #global_values['Clustering_coeff_weighted'] = line[i+3].strip().split('\t')[-1]
                #global_values['Clustering_coeff_weighted_pos'] = lines[i+4].strip().split('\t')[-1]
                #global_values['Clustering_coeff_weighted_neg'] = lines[i+5].strip().split('\t')[-1]

            elif split_line[0] == 'Minimum' and split_line[1] == 'degree':

                print("Found Minimum degree value line")

                print(line.strip().split('\t')[-1])

                global_values['Minimum_degree'] = line.strip().split('\t')[-1]
                global_values['Minimum_pos_degree'] = lines[i +
                                                            1].strip().split('\t')[-1]
                global_values['Minimum_neg_degree'] = lines[i +
                                                            2].strip().split('\t')[-1]

            elif split_line[0] == 'Minimum' and split_line[1] == 'strength':

                print("Found Minimum strength value line")

                print(line.strip().split('\t')[-1])

                global_values['Minimum_strength'] = line.strip().split(
                    '\t')[-1]
                global_values['Minimum_pos_strength'] = lines[i +
                                                              1].strip().split('\t')[-1]
                global_values['Minimum_neg_strength'] = lines[i +
                                                              2].strip().split('\t')[-1]

            elif split_line[0] == 'Maximum' and split_line[1] == 'degree':

                print("Found Maximum degree value line")

                print(line.strip().split('\t')[-1])

                global_values['Maximum_degree'] = line.strip().split('\t')[-1]
                global_values['Maximum_pos_degree'] = lines[i +
                                                            1].strip().split('\t')[-1]
                global_values['Maximum_neg_degree'] = lines[i +
                                                            2].strip().split('\t')[-1]

            elif split_line[0] == 'Maximum' and split_line[1] == 'strength':

                print("Found Maximum strength value line")

                print(line.strip().split('\t')[-1])

                global_values['Maximum_strength'] = line.strip().split(
                    '\t')[-1]
                global_values['Maximum_pos_strength'] = lines[i +
                                                              1].strip().split('\t')[-1]
                global_values['Maximum_neg_strength'] = lines[i +
                                                              2].strip().split('\t')[-1]

            elif split_line[0] == 'Assortativity':

                print("Found Assortativity value line")

                print(line.strip().split('\t')[-1])

                global_values['Assortativity'] = line.strip().split('\t')[-1]
                global_values['Assortativity_pos'] = lines[i +
                                                           1].strip().split('\t')[-1]
                global_values['Assortativity_neg'] = lines[i +
                                                           2].strip().split('\t')[-1]

                # print(lines[i+3].strip().split('\t'))

                #global_values['Assortativity_weighted'] = lines[i+3].strip().split('\t')[-1]
                #global_values['Assortativity_weighted_pos'] = lines[i+4].strip().split('\t')[-1]
                #global_values['Assortativity_weighted_neg'] = lines[i+5].strip().split('\t')[-1]

        return global_values


def get_path_length_from_info_dists_file(info_dists_file):

    dist_mat = np.loadtxt(info_dists_file)

    print(dist_mat.shape)

    if len(dist_mat.shape) != 2:
        print("warning, only works with 2d arrays (matrices), now array has {} dimensions".format(
            len(dist_mat.shape)))
        sys.exit()

    if dist_mat.shape[0] != dist_mat.shape[1]:
        print("warning, only works with squred matricesm now array  dimensions {} != {}".format(
            dist_mat.shape[0], dist_mat.shape[1]))
        sys.exit()

    triu_dist_mat = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]

    inv_triu_dist_map = 1.0/triu_dist_mat

    print(np.mean(inv_triu_dist_map))

    return np.mean(triu_dist_mat), np.max(triu_dist_mat), np.mean(inv_triu_dist_map)

# modularity

##################################### Formatting data for external community detection algorithm (radatools) ##############################


def read_lol_file(lol_file):

    with open(lol_file, 'r') as f:

        lines = f.readlines()[4:]

        nb_elements = int(lines[0].split(': ')[1])

        community_vect = np.empty((nb_elements), dtype=int)

        for i, line in enumerate(lines[3:]):

            try:
                nb_nodes, index_nodes = line.split(': ')
                print(nb_nodes, index_nodes)

                if int(nb_nodes) > 1:
                    index_nodes = np.array(
                        list(map(int, index_nodes.split(' '))), dtype=int) - 1

                    # print i,index_nodes
                    community_vect[index_nodes] = i

                else:
                    community_vect[int(index_nodes) - 1] = i

            except ValueError:
                print("Warning, error reading lol file ")

        f.close()

    return community_vect

# compute modular matrix from sparse matrix and community vect


def compute_modular_matrix(sparse_matrix, community_vect):

    mod_mat = np.empty(sparse_matrix.todense().shape)

    mod_mat[:] = np.NAN

    for u, v, w in zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):

        if (community_vect[u] == community_vect[v]):

            mod_mat[u, v] = community_vect[u]
        else:

            mod_mat[u, v] = -1

    return mod_mat

# Node roles


def _return_all_Z_com_degree(community_vect, dense_mat):

    degree_vect = np.sum(dense_mat != 0, axis=1)

    community_indexes = np.unique(community_vect)

    all_Z_com_degree = np.zeros(shape=(community_vect.shape[0]))

    for com_index in community_indexes:

        com_degree = degree_vect[com_index == community_vect]

        if com_degree.shape[0] > 1:

            Z_com_degree = (com_degree - np.mean(com_degree)) / \
                np.std(com_degree)

            all_Z_com_degree[com_index == community_vect] = Z_com_degree

        else:
            all_Z_com_degree[com_index == community_vect] = 0

    return all_Z_com_degree


def _return_all_participation_coeff(community_vect, dense_mat):

    degree_vect = np.array(np.sum(dense_mat != 0, axis=1), dtype='float')

    community_indexes = np.unique(community_vect)

    all_participation_coeff = np.ones(
        shape=(community_vect.shape[0]), dtype='float')

    for com_index in community_indexes:

        degree_com_vect = np.sum(
            dense_mat[:, com_index == community_vect], axis=1, dtype='float')

        rel_com_degree = np.square(degree_com_vect/degree_vect)

        all_participation_coeff = all_participation_coeff - rel_com_degree

    return all_participation_coeff


def _return_amaral_roles(all_Z_com_degree, all_participation_coeff):

    if (all_Z_com_degree.shape[0] != all_participation_coeff.shape[0]):
        print("Warning, all_Z_com_degree %d should have same length as all_participation_coeff %d " % (
            all_Z_com_degree.shape[0], all_participation_coeff.shape[0]))
        return 0

    nod_roles = np.zeros(shape=(all_Z_com_degree.shape[0], 2), dtype='int')

    # hubs are at 2,non-hubs are at 1
    hubs = all_Z_com_degree > 2.5
    non_hubs = all_Z_com_degree <= 2.5

    nod_roles[hubs, 0] = 2
    nod_roles[non_hubs, 0] = 1

    # for non-hubs
    # ultraperipheral nodes
    ultraperi_non_hubs = np.logical_and(
        all_participation_coeff < 0.05, non_hubs == True)

    print(np.sum(ultraperi_non_hubs, axis=0))
    nod_roles[ultraperi_non_hubs, 1] = 1

    # ultraperipheral nodes
    peri_non_hubs = np.logical_and(np.logical_and(
        0.05 <= all_participation_coeff, all_participation_coeff < 0.62), non_hubs == True)

    print(np.sum(peri_non_hubs, axis=0))
    nod_roles[peri_non_hubs, 1] = 2

    # non-hub connectors
    non_hub_connectors = np.logical_and(np.logical_and(
        0.62 <= all_participation_coeff, all_participation_coeff < 0.8), non_hubs == True)

    print(np.sum(non_hub_connectors, axis=0))
    nod_roles[non_hub_connectors, 1] = 3

    # kinless non-hubs
    kin_less_non_hubs = np.logical_and(
        0.8 <= all_participation_coeff, non_hubs == True)

    print(np.sum(kin_less_non_hubs, axis=0))
    nod_roles[kin_less_non_hubs, 1] = 4

    # for hubs
    # provincial hubs
    prov_hubs = np.logical_and(all_participation_coeff < 0.3, hubs == True)

    nod_roles[prov_hubs, 1] = 5

    # hub connectors
    hub_connectors = np.logical_and(np.logical_and(
        0.3 <= all_participation_coeff, all_participation_coeff < 0.75), hubs == True)

    print(np.sum(hub_connectors, axis=0))
    nod_roles[hub_connectors, 1] = 6

    # kinless hubs
    kin_less_hubs = np.logical_and(
        0.75 <= all_participation_coeff, hubs == True)

    nod_roles[kin_less_hubs, 1] = 7

    return nod_roles


def _return_4roles(all_Z_com_degree, all_participation_coeff):

    if (all_Z_com_degree.shape[0] != all_participation_coeff.shape[0]):
        print("Warning, all_Z_com_degree %d should have same length as all_participation_coeff %d " % (
            all_Z_com_degree.shape[0], all_participation_coeff.shape[0]))
        return 0

    nod_roles = np.zeros(shape=(all_Z_com_degree.shape[0], 2), dtype='int')

    # hubs are at 2,non-hubs are at 1
    hubs = all_Z_com_degree > 1.0
    non_hubs = all_Z_com_degree <= 1.0

    nod_roles[hubs, 0] = 2
    nod_roles[non_hubs, 0] = 1

    # provincial nodes
    provincial_nodes = all_participation_coeff < 0.3

    nod_roles[provincial_nodes, 1] = 1

    # connector nodes
    connector_nodes = 0.3 <= all_participation_coeff

    nod_roles[connector_nodes, 1] = 2

    print("\nNode roles:")
    print("*Hubs/non-hubs: ", np.sum(hubs, axis=0), np.sum(non_hubs, axis=0))
    print("*provincials/connectors : ", np.sum(provincial_nodes, axis=0),
          np.sum(connector_nodes, axis=0))
    return nod_roles


def compute_roles(community_vect, sparse_mat, role_type="Amaral_roles"):

    import numpy as np

    dense_mat = sparse_mat.todense()

    undir_dense_mat = dense_mat + np.transpose(dense_mat)

    bin_dense_mat = np.array(undir_dense_mat != 0, dtype=int)

    ##################################### within community Z-degree #########################

    all_Z_com_degree = _return_all_Z_com_degree(community_vect, bin_dense_mat)

    ##################################### participation_coeff ###############################

    all_participation_coeff = _return_all_participation_coeff(
        community_vect, bin_dense_mat)

    if role_type == "Amaral_roles":

        node_roles = _return_amaral_roles(
            all_Z_com_degree, all_participation_coeff)

    elif role_type == "4roles":

        node_roles = _return_4roles(all_Z_com_degree, all_participation_coeff)

    return node_roles, all_Z_com_degree, all_participation_coeff
