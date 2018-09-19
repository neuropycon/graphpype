"""
Support function for mod handling
Author:
    David Meunier <david_meunier_79@hotmail.fr>
"""
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
import itertools as iter


# from lol_file
def get_modularity_value_from_lol_file(lol_file):
    """get_modularity_value_from_lol_file"""
    with open(lol_file, 'r') as f:
        for line in f.readlines():
            split_line = line.strip().split(' ')
            print(split_line)
            if split_line[0] == 'Q':
                print("Found modularity value line")
                return split_line[2]
        print("Unable to find modularity line in file, returning -1")
        return -1.0


# reading info files
# from info-nodes
def get_max_degree_from_node_info_file(info_nodes_file):
    """Return max degree AND index and name of max degree (radatools based)"""
    df = pd.read_table(info_nodes_file)

    max_degree_value = df['Degree'].max()
    md_indexes = df[df['Degree'] == max_degree_value].Index.values[0]
    md_names = df[df['Degree'] == max_degree_value].Name.values[0]

    return max_degree_value, md_indexes, md_names


def get_strength_values_from_info_nodes_file(info_nodes_file):
    """Read strength from Network_Properties node results"""
    info_nodes = read_csv(info_nodes_file, sep="\t")
    return info_nodes['Strength'].values


def get_strength_pos_values_from_info_nodes_file(info_nodes_file):
    """Read positive strength from Network_Properties node results"""
    info_nodes = read_csv(info_nodes_file, sep="\t")
    return info_nodes['Strength_Pos']  # not values?


def get_strength_neg_values_from_info_nodes_file(info_nodes_file):
    """Read negative strength from Network_Properties node results"""
    info_nodes = read_csv(info_nodes_file, sep="\t")
    return info_nodes['Strength_Neg']  # not values?


def get_degree_pos_values_from_info_nodes_file(info_nodes_file):
    """Read positive degree from Network_Properties node results"""
    info_nodes = read_csv(info_nodes_file, sep="\t")
    return info_nodes['Degree_Pos']


def get_degree_neg_values_from_info_nodes_file(info_nodes_file):
    """Read negative degree from Network_Properties node results"""
    info_nodes = read_csv(info_nodes_file, sep="\t")
    return info_nodes['Degree_Neg']


# from info_global


def get_values_from_global_info_file(global_info_file):
    """Read all values from global_info file"""
    global_values = {}

    with open(global_info_file, 'r') as f:

        lines = f.readlines()
        for i, line in enumerate(lines):

            split_line = line.strip().split(' ')

            cur_last_val = line.strip().split('\t')[-1]
            if i < (len(lines)-1):
                next_last_val = lines[i+1].strip().split('\t')[-1]

            if split_line[0] == 'Vertices':
                global_values['Vertices'] = cur_last_val

            elif split_line[0] == 'Edges':
                global_values['Edges'] = cur_last_val

            elif split_line[0] == 'Total':
                if split_line[1] == 'degree':
                    global_values['Total_degree'] = cur_last_val

                elif split_line[1] == 'strength':
                    global_values['Total_strength'] = cur_last_val

            elif split_line[0] == 'Average':
                if split_line[1] == 'degree':
                    global_values['Average_degree'] = cur_last_val

                elif split_line[1] == 'strength':
                    global_values['Average_strength'] = cur_last_val

                elif split_line[1] == 'clustering':
                    if split_line[2] == 'coefficient':
                        global_values['Clustering_coeff'] = cur_last_val
                        a = 'Clustering_coeff_weighted'
                        global_values[a] = next_last_val  # not very nice

            elif split_line[0] == 'Minimum':
                if split_line[1] == 'degree':
                    global_values['Minimum_degree'] = cur_last_val

                elif split_line[1] == 'strength':
                    global_values['Minimum_strength'] = cur_last_val

            elif split_line[0] == 'Maximum':
                if split_line[1] == 'degree':
                    global_values['Maximum_degree'] = cur_last_val

                elif split_line[1] == 'strength':
                    global_values['Maximum_strength'] = cur_last_val

            elif split_line[0] == 'Assortativity':
                global_values['Assortativity'] = cur_last_val
                global_values['Assortativity_weighted'] = next_last_val

        return global_values


def get_values_from_signed_global_info_file(global_info_file):

    global_values = {}

    with open(global_info_file, 'r') as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            split_line = line.strip().split(' ')

            cur_last_val = line.strip().split('\t')[-1]

            if i < (len(lines)-1):
                next_last_val = lines[i+1].strip().split('\t')[-1]
            if i < (len(lines)-2):
                fol_next_last_val = lines[i+2].strip().split('\t')[-1]

            if split_line[0] == 'Vertices':

                global_values['Vertices'] = cur_last_val

            elif split_line[0] == 'Edges':

                global_values['Edges'] = cur_last_val

            elif split_line[0] == 'Total' and split_line[1] == 'degree':

                global_values['Total_degree'] = cur_last_val
                global_values['Total_pos_degree'] = next_last_val
                global_values['Total_neg_degree'] = fol_next_last_val

            elif split_line[0] == 'Total' and split_line[1] == 'strength':

                global_values['Total_strength'] = cur_last_val
                global_values['Total_pos_strength'] = next_last_val
                global_values['Total_neg_strength'] = fol_next_last_val

            elif split_line[0] == 'Average' and split_line[1] == 'degree':

                global_values['Average_degree'] = cur_last_val
                global_values['Average_pos_degree'] = next_last_val
                global_values['Average_neg_degree'] = fol_next_last_val

            elif split_line[0] == 'Average' and split_line[1] == 'strength':

                global_values['Average_strength'] = cur_last_val
                global_values['Average_pos_strength'] = next_last_val
                global_values['Average_neg_strength'] = fol_next_last_val

            elif split_line[0] == 'Average' and split_line[1] == 'clustering'\
                    and split_line[2] == 'coefficient':

                global_values['Clustering_coeff'] = cur_last_val
                global_values['Clustering_coeff_pos'] = next_last_val
                global_values['Clustering_coeff_neg'] = fol_next_last_val

            elif split_line[0] == 'Minimum' and split_line[1] == 'degree':

                global_values['Minimum_degree'] = cur_last_val
                global_values['Minimum_pos_degree'] = next_last_val
                global_values['Minimum_neg_degree'] = fol_next_last_val

            elif split_line[0] == 'Minimum' and split_line[1] == 'strength':

                global_values['Minimum_strength'] = cur_last_val
                global_values['Minimum_pos_strength'] = next_last_val
                global_values['Minimum_neg_strength'] = fol_next_last_val

            elif split_line[0] == 'Maximum' and split_line[1] == 'degree':

                global_values['Maximum_degree'] = cur_last_val
                global_values['Maximum_pos_degree'] = next_last_val
                global_values['Maximum_neg_degree'] = fol_next_last_val

            elif split_line[0] == 'Maximum' and split_line[1] == 'strength':

                global_values['Maximum_strength'] = cur_last_val
                global_values['Maximum_pos_strength'] = next_last_val
                global_values['Maximum_neg_strength'] = fol_next_last_val

            elif split_line[0] == 'Assortativity':

                global_values['Assortativity'] = cur_last_val
                global_values['Assortativity_pos'] = next_last_val
                global_values['Assortativity_neg'] = fol_next_last_val

        return global_values


def get_path_length_from_info_dists_file(info_dists_file):

    dist_mat = np.loadtxt(info_dists_file)

    print(dist_mat.shape)

    assert len(dist_mat.shape) == 2, ("Error, only works with 2d arrays \
        (matrices), now array has {} dimensions".format(len(dist_mat.shape)))

    assert dist_mat.shape[0] == dist_mat.shape[1], ("Error, only works with \
        squred matrices, now array  dimensions {} != \
        {}".format(dist_mat.shape[0], dist_mat.shape[1]))

    triu_dist_mat = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]

    inv_d_map = 1.0/triu_dist_mat

    return np.mean(triu_dist_mat), np.max(triu_dist_mat), np.mean(inv_d_map)


# modularity


def read_lol_file(lol_file):
    """Formatting data for community detection algorithm radatools"""
    with open(lol_file, 'r') as f:
        lines = f.readlines()[4:]
        nb_elements = int(lines[0].split(': ')[1])
        community_vect = np.empty((nb_elements), dtype=int)

        for i, line in enumerate(lines[3:]):
            try:
                nb_nodes, index_nodes = line.split(': ')
                if int(nb_nodes) > 1:
                    index_nodes = np.array(
                        list(map(int, index_nodes.split(' '))), dtype=int) - 1
                    community_vect[index_nodes] = i

                else:
                    community_vect[int(index_nodes) - 1] = i

            except ValueError:
                print("Warning, error reading lol file ")

    return community_vect


# compute modular matrix from sparse matrix and community vect
def compute_modular_matrix(sp_mat, community_vect):

    mod_mat = np.empty(sp_mat.todense().shape)

    mod_mat[:] = np.NAN

    for u, v, w in zip(sp_mat.row, sp_mat.col, sp_mat.data):

        if (community_vect[u] == community_vect[v]):
            mod_mat[u, v] = community_vect[u]
        else:
            mod_mat[u, v] = -1

    return mod_mat

# Node roles


def _return_Z_com_deg(community_vect, dense_mat):

    degree_vect = np.sum(dense_mat != 0, axis=1)

    community_indexes = np.unique(community_vect)

    Z_com_deg = np.zeros(shape=(community_vect.shape[0]))

    for com_index in community_indexes:

        com_degree = degree_vect[com_index == community_vect]

        if com_degree.shape[0] > 1:

            Z_com_degree = (com_degree - np.mean(com_degree)) / \
                np.std(com_degree)

            Z_com_deg[com_index == community_vect] = Z_com_degree

        else:
            Z_com_deg[com_index == community_vect] = 0

    return Z_com_deg


def _return_parti_coef(community_vect, dense_mat):

    degree_vect = np.array(np.sum(dense_mat != 0, axis=1), dtype='float')

    community_indexes = np.unique(community_vect)

    parti_coef = np.ones(
        shape=(community_vect.shape[0]), dtype='float')

    for com_index in community_indexes:

        degree_com_vect = np.sum(
            dense_mat[:, com_index == community_vect], axis=1, dtype='float')

        rel_com_degree = np.square(degree_com_vect/degree_vect)

        parti_coef = parti_coef - rel_com_degree

    return parti_coef


def _return_amaral_roles(Z_com_deg, parti_coef):
    """compute Amaral roles (7 node categories)"""
    assert Z_com_deg.shape[0] == parti_coef.shape[0], ("Error, Z_com_deg {} \
        should have same length as parti_coef {}".format(Z_com_deg.shape[0],
                                                         parti_coef.shape[0]))

    nod_roles = np.zeros(shape=(Z_com_deg.shape[0], 2), dtype='int')

    # hubs are at 2, non-hubs are at 1
    non_hubs = Z_com_deg <= 2.5
    nod_roles[Z_com_deg > 2.5, 0] = 2
    nod_roles[Z_com_deg <= 2.5, 0] = 1

    # for non-hubs
    # ultraperipheral nodes
    ultraperi_non_hubs = (parti_coef < 0.05) & (Z_com_deg <= 2.5)

    print(np.sum(ultraperi_non_hubs, axis=0))
    nod_roles[ultraperi_non_hubs, 1] = 1

    # ultraperipheral nodes
    peri_non_hubs = (0.05 <= parti_coef) & (parti_coef < 0.62) & non_hubs

    nod_roles[peri_non_hubs, 1] = 2

    # non-hub connectors
    non_hub_connectors = (0.62 <= parti_coef) & (parti_coef < 0.8) & non_hubs

    print(np.sum(non_hub_connectors, axis=0))
    nod_roles[non_hub_connectors, 1] = 3

    # kinless non-hubs
    kin_less_non_hubs = (0.8 <= parti_coef) & (Z_com_deg <= 2.5)

    print(np.sum(kin_less_non_hubs, axis=0))
    nod_roles[kin_less_non_hubs, 1] = 4

    # for hubs
    # provincial hubs
    prov_hubs = (parti_coef < 0.3) & (Z_com_deg <= 2.5)

    nod_roles[prov_hubs, 1] = 5

    # hub connectors
    hub_connectors = (0.3 <= parti_coef) & (parti_coef < 0.75) & non_hubs

    print(np.sum(hub_connectors, axis=0))
    nod_roles[hub_connectors, 1] = 6

    # kinless hubs
    kin_less_hubs = (0.75 <= parti_coef) & non_hubs

    nod_roles[kin_less_hubs, 1] = 7

    return nod_roles


def _return_4roles(Z_com_deg, parti_coef):
    """private function for computing 4 roles"""
    assert Z_com_deg.shape[0] == parti_coef.shape[0], ("Error, Z_com_deg {} \
        should have same length as parti_coef {} ".format(Z_com_deg.shape[0],
                                                          parti_coef.shape[0]))

    nod_roles = np.zeros(shape=(Z_com_deg.shape[0], 2), dtype='int')

    # hubs are at 2,non-hubs are at 1
    nod_roles[Z_com_deg > 1.0, 0] = 2
    nod_roles[Z_com_deg <= 1.0, 0] = 1

    # provincial nodes
    nod_roles[parti_coef < 0.3, 1] = 1

    # connector nodes
    nod_roles[0.3 <= parti_coef, 1] = 2

    print("\nNode roles:")
    print("*Hubs {}/non-hubs {}".format(np.sum(Z_com_deg > 1.0, axis=0),
                                        np.sum(Z_com_deg <= 1.0, axis=0)))
    print("*provincials {}/connectors {}".format(
        np.sum(parti_coef < 0.3, axis=0), np.sum(0.3 <= parti_coef, axis=0)))
    return nod_roles


def compute_roles(community_vect, sparse_mat, role_type="Amaral_roles"):
    """compute node roles from modular partition and graph"""

    dense_mat = sparse_mat.todense()
    undir_dense_mat = dense_mat + np.transpose(dense_mat)
    bin_dense_mat = np.array(undir_dense_mat != 0, dtype=int)

    # within community Z-degree
    Z_com_deg = _return_Z_com_deg(community_vect, bin_dense_mat)

    # participation_coeff
    parti_coef = _return_parti_coef(community_vect, bin_dense_mat)

    if role_type == "Amaral_roles":
        node_roles = _return_amaral_roles(Z_com_deg, parti_coef)

    elif role_type == "4roles":
        node_roles = _return_4roles(Z_com_deg, parti_coef)

    return node_roles, Z_com_deg, parti_coef


# modules and intermodules computation


def _inter_module_avgmat(con_mat, community_vect):
    """
    intermodules computation
    """
    assert con_mat.shape[0] == community_vect.shape[0], \
        ("Error, mat {}!= community_vect {}".format(
            con_mat.shape[0], community_vect.shape[0]))

    index_mod = np.unique(community_vect)
    nb_mod = index_mod.shape[0]

    avgmat = np.zeros(shape=(nb_mod, nb_mod))

    for i, j in iter.product(index_mod, repeat=2):
        ind_i = np.where(community_vect == i)
        ind_j = np.where(community_vect == j)
        mod_mat = con_mat[ind_i[0], :][:, ind_j[0]]

        if i == j:
            triu = np.triu_indices(mod_mat.shape[0], k=1)
            tri_mod_mat = mod_mat[triu[0], triu[1]]
            if tri_mod_mat.shape[0]:
                avgmat[i, j] = np.mean(tri_mod_mat)
        else:

            avgmat[i, j] = np.mean(mod_mat)

    mod_labels = ["module_"+str(i) for i in np.unique(community_vect)]
    df_avgmat = pd.DataFrame(avgmat, columns=mod_labels)
    return df_avgmat
