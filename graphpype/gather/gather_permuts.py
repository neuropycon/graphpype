# -*- coding: utf-8 -*-

import os

import glob

import pandas as pd
import numpy as np

from collections import Counter

from graphpype.utils_net import read_lol_file, read_Pajek_corres_nodes
from graphpype.utils_dtype_coord import where_in_coords

from graphpype.utils_mod import get_modularity_value_from_lol_file
from graphpype.utils_mod import get_values_from_global_info_file
from graphpype.utils_mod import get_path_length_from_info_dists_file


def glob_natural_sorted(reg_exp):
    # TODO -> utils.py
    """sort reg_exp filenames in natural way (for numbers)"""

    print(reg_exp)

    files = glob.glob(reg_exp)

    print(len(files))

    natural_sorted_files = [reg_exp.replace(
        '*', str(i), -1) for i in range(len(files))]

    return natural_sorted_files, list(range(len(files)))


def compute_rada_df(iter_path, df, radatools_version="3.2", mapflow=[],
                    mapflow_name=""):
    """gather rada """
    if radatools_version == "3.2":

        net_prop_dir = "net_prop"

    elif radatools_version == "4.0":

        net_prop_dir = "prep_rada"
    else:
        print("Warning, could not find radatools_version {}"
              .format(radatools_version))
        return

    # modularity
    if len(mapflow) == 0:

        modularity_file = os.path.join(
            iter_path, "community_rada", "Z_List.lol")

        print(modularity_file)

        if os.path.exists(modularity_file):

            mod_val = get_modularity_value_from_lol_file(modularity_file)
            df['Modularity'] = mod_val

        print(df)

        # info_global
        global_info_file = os.path.join(
            iter_path, net_prop_dir, "Z_List-info_global.txt")

        if os.path.exists(global_info_file):

            global_info_values = get_values_from_global_info_file(
                global_info_file)

            df.update(global_info_values)

        # info_dists
        path_length_file = os.path.join(
            iter_path, net_prop_dir, "Z_List-info_dists.txt")

        if os.path.exists(path_length_file):

            mean_path_length, diameter, global_efficiency = \
                get_path_length_from_info_dists_file(path_length_file)

            df['Mean_path_length'] = str(mean_path_length)
            df['Diameter'] = str(diameter)
            df['Global_efficiency'] = str(global_efficiency)

    else:

        df['Modularity'] = []
        df[mapflow_name] = []

        df['Mean_path_length'] = []
        df['Diameter'] = []
        df['Global_efficiency'] = []

        for i, cond in enumerate(mapflow):

            df[mapflow_name].append(cond)

            modularity_file = os.path.join(
                iter_path, "community_rada", "mapflow",
                "_community_rada"+str(i), "Z_List.lol")

            if os.path.exists(modularity_file):

                mod_val = get_modularity_value_from_lol_file(modularity_file)

                df['Modularity'].append(mod_val)

            else:
                df['Modularity'].append(np.nan)

            # info_global
            global_info_file = os.path.join(
                iter_path, net_prop_dir, "mapflow", "_" + net_prop_dir+str(i),
                "Z_List-info_global.txt")

            if os.path.exists(global_info_file):

                global_info_values = get_values_from_global_info_file(
                    global_info_file)

                for key, value in global_info_values.items():

                    if key not in list(df.keys()):
                        df[key] = []

                    df[key].append(value)

            # info_dists
            path_length_file = os.path.join(
                iter_path, net_prop_dir, "mapflow", "_" + net_prop_dir+str(i),
                "Z_List-info_dists.txt")

            if os.path.exists(path_length_file):

                mean_path_length, diameter, global_efficiency = \
                    get_path_length_from_info_dists_file(path_length_file)

                df['Mean_path_length'].append(str(mean_path_length))
                df['Diameter'].append(str(diameter))
                df['Global_efficiency'].append(str(global_efficiency))
            else:

                df['Mean_path_length'].append(str(np.nan))
                df['Diameter'].append(str(np.nan))
                df['Global_efficiency'].append(str(np.nan))


def compute_nodes_rada_df(local_dir, gm_coords, coords_file, labels_file,
                          radatools_version="3.2"):
    """node properties df"""
    if radatools_version == "3.2":

        net_prop_dir = "net_prop"

    elif radatools_version == "4.0":

        net_prop_dir = "prep_rada"

    list_df = []

    Pajek_file = os.path.join(local_dir, "prep_rada", "Z_List.net")

    if os.path.exists(coords_file) and os.path.exists(Pajek_file) and \
            os.path.exists(labels_file):

        # labels
        labels = np.array([line.strip()
                           for line in open(labels_file)], dtype=str)

        # MNI coordinates
        coords = np.array(np.loadtxt(coords_file), dtype=int)

        # nodes in the connected graph
        node_corres = read_Pajek_corres_nodes(Pajek_file)

        # node_coords
        node_coords = coords[node_corres, :]
        node_labels = labels[node_corres].reshape(-1, 1)

        # where_in_gm_mask
        where_in_gm_mask = where_in_coords(node_coords, gm_coords)

        where_in_gm_mask = where_in_gm_mask.reshape(
            where_in_gm_mask.shape[0], 1)

        # print where_in_gm_mask
        print(where_in_gm_mask.shape)

        list_df.append(pd.DataFrame(
            np.concatenate((where_in_gm_mask, node_labels, node_coords),
                           axis=1),
            columns=['Where_in_GM_mask', 'labels', 'MNI_x', 'MNI_y', 'MNI_z']))
    else:
        if not os.path.exists(coords_file):
            print("Missing {}".format(coords_file))

        if not os.path.exists(Pajek_file):
            print("Missing {}".format(Pajek_file))

        if not os.path.exists(labels_file):
            print("Missing {}".format(labels_file))

    info_nodes_file = os.path.join(
        local_dir, net_prop_dir, "Z_List-info_nodes.txt")

    print(info_nodes_file)

    if os.path.exists(info_nodes_file):

        # loading info_nodes
        df_node_info = pd.read_table(info_nodes_file)
        list_df.append(df_node_info)

    # modules /community_vect
    partition_file = os.path.join(local_dir, "community_rada", "Z_List.lol")

    if os.path.exists(partition_file):

        # loading partition_file
        community_vect = read_lol_file(partition_file)
        list_df.append(pd.DataFrame(community_vect, columns=['Module']))

    # node roles
    roles_file = os.path.join(local_dir, "node_roles", "node_roles.txt")

    part_coeff_file = os.path.join(
        local_dir, "node_roles", "all_participation_coeff.txt")

    Z_com_degree_file = os.path.join(
        local_dir, "node_roles", "all_Z_com_degree.txt")

    if os.path.exists(roles_file) and os.path.exists(part_coeff_file) and \
            os.path.exists(Z_com_degree_file):

        # loding node roles
        node_roles = np.array(np.loadtxt(roles_file), dtype=int)

        part_coeff = np.loadtxt(part_coeff_file)
        part_coeff = part_coeff.reshape(part_coeff.shape[0], 1)

        Z_com_degree = np.loadtxt(Z_com_degree_file)
        Z_com_degree = Z_com_degree.reshape(Z_com_degree.shape[0], 1)

        list_df.append(pd.DataFrame(
            np.concatenate((node_roles, part_coeff, Z_com_degree), axis=1),
            columns=['Role_quality', 'Role_quantity',
                     'Participation_coefficient', 'Z_community_degree']))

    return list_df


def compute_signif_permuts(permut_df, permut_col="Seed",
                           session_col="Session", start_col=0, stop_col=0,
                           columns=[]):
    """
    Computing permutation-based stats per nodes, over several sheetnames

    args:
    compute significance of permutation over a df generated by gather_permuts
    permut_df: original permutation results (pandas Dataframe)
    stop_col: last column to be included
    (in fact, excluded except if value is 0, in this case goes to the last
    column of the df)

    return:
    all_p_higher, all_p_lower: "vector of p_values obtained for 1 tail t-test
    in both direction, first session - second session"

    """

    seed_index = np.unique(permut_df[permut_col].values)

    print(seed_index)

    # should start with -1
    assert seed_index[0] == -1, \
        ("Error, permut_col {} should start with -1".format(permut_col))

    expected_permut_indexes = list(range(len(seed_index)-1))

    print(expected_permut_indexes)

    # should start at 0 and have all values in between
    assert all(x in seed_index[1:] for x in expected_permut_indexes), \
        ("Error, permut indexes should be consecutive and start with \
              0: {} ".format(expected_permut_indexes))

    nb_permuts = len(expected_permut_indexes)

    print(nb_permuts)

    # selecting columns

    if len(columns) != 0:
        data_cols = columns

    else:

        if stop_col == 0:
            data_cols = permut_df.columns[start_col:]

        else:
            data_cols = permut_df.columns[start_col:stop_col]

    print(data_cols)

    # looping over selected columns
    all_p_higher = np.zeros(shape=(len(data_cols)), dtype='float64') - 1
    all_p_lower = np.zeros(shape=(len(data_cols)), dtype='float64') - 1

    cols = []

    if session_col == -1 or len(permut_df[session_col].unique()) == 1:

        for index_col, col in enumerate(data_cols):

            print(index_col, col)

            sum_higher = np.sum(
                (permut_df[col].iloc[1:] > permut_df[col].iloc[0])
                .values.astype(int))
            all_p_higher[index_col] = (
                sum_higher+1)/float(permut_df[col].shape[0])

            sum_lower = np.sum(
                (permut_df[col].iloc[1:] < permut_df[col].iloc[0])
                .values.astype(int))
            all_p_lower[index_col] = (sum_lower+1) / \
                float(permut_df[col].shape[0])

            cols.append(str(col))
    else:
        # all unique values should have 2 different samples
        count_elements = Counter(permut_df[permut_col].values)

        # -1 should be represented Two times:
        if not count_elements[-1] == 2:
            print("-1 should be represented Two times")

            return pd.DataFrame()

        if not all(val == 2 for val in list(count_elements.values())):
            print("Error, all permut indexes should have 2 lines: {}"
                  .format(count_elements))

        # computing diff df

        for index_col, col in enumerate(data_cols):

            df_col = permut_df.pivot(
                index=permut_col, columns=session_col, values=col)

            df_col["Diff"] = pd.to_numeric(
                df_col.iloc[:, 0]) - pd.to_numeric(df_col.iloc[:, 1])

            # asked by flake8 but not sure relevant for pandas Series...
            non_nan_indexes, = np.where(np.isnan(df_col["Diff"]) is False)

            print(non_nan_indexes)

            diff_col = df_col["Diff"].values[non_nan_indexes]

            if not all(val == 2 for val in list(count_elements.values())):
                print("Error, all permut indexes should have 2 lines: {}"
                      .format(count_elements))

                print(diff_col)
                print(diff_col.shape)

            if diff_col.shape[0] == 0:

                all_p_higher[index_col] = np.nan
                all_p_lower[index_col] = np.nan
                cols.append(col)

                continue

            if diff_col[0] > 0:
                sum_higher = np.sum(
                    np.array(diff_col[1:] > diff_col[0], dtype=int))
                print("sum_higher:", sum_higher)
                all_p_higher[index_col] = (sum_higher+1)/float(df_col.shape[0])

            elif diff_col[0] < 0:
                sum_lower = np.sum(
                    np.array(diff_col[1:] < diff_col[0], dtype=int))
                print("sum_lower:", sum_lower)
                all_p_lower[index_col] = (sum_lower+1)/float(df_col.shape[0])

            else:
                print("not able to do diff")

            cols.append(col)

    df_res = pd.DataFrame([all_p_higher, all_p_lower], columns=cols)
    df_res.index = ["Higher", "Lower"]

    return df_res


def compute_signif_node_prop(orig_df, list_permut_df, columns):
    """signif node properties"""
    permut_df = pd.concat(list_permut_df, axis=0)

    all_frac_higher = []

    for col in columns:

        assert col in orig_df.columns, \
            "Error, {} not in orig columns {}".format(col, orig_df.columns)
        assert col in permut_df.columns, \
            "Error, {} not in permut columns {}".format(col, permut_df.columns)

        def sum_higher(a, b):

            def func(el):
                return np.sum(el[0] < b.values)

            return np.apply_along_axis(func, 1, a[:, None])

        frac_higher = np.array(
            sum_higher(orig_df[col], permut_df[col])+1,
            dtype=float)/float(len(permut_df.index) + 1)

        all_frac_higher.append(frac_higher)

    df_signif = pd.DataFrame(np.transpose(
        np.array(all_frac_higher)), columns=columns)

    return df_signif


def gather_diff_con_values(res_path, cond, nb_permuts, labels):
    """gather con values"""

    if isinstance(cond, tuple):
        # si plusieurs conditions = IRMf
        df_filename = os.path.join(
            res_path, "permuts_" + ".".join(cond) + '_con_values.csv')

    else:
        # si une seule valeur
        df_filename = os.path.join(
            res_path, "permuts_" + cond + '_con_values.csv')

    if not os.path.exists(df_filename):

        # pair of labels and tri triu_indices

        triu_indices_i, triu_indices_j = np.triu_indices(len(labels), k=1)

        pair_labels = [labels[i] + "_" + labels[j]
                       for i, j in zip(triu_indices_i.tolist(),
                                       triu_indices_j.tolist())]

        print(pair_labels)
        print(len(pair_labels))

        # creating dataframe
        all_vect_cormats = []

        all_global_info_values = []

        for seed in range(-1, nb_permuts):

            print(seed)

            for sess in ['1', '2']:

                print(sess)

                dict_global_info_values = {'Session': sess, 'Seed': seed}

                all_global_info_values.append(dict_global_info_values)

                # avg_cormat

                if isinstance(cond, tuple):
                    iter_dir = "_cond_" + \
                        ".".join(cond) + "_permut_" + str(seed)

                else:
                    iter_dir = "_freq_band_name_" + \
                        cond + "_permut_" + str(seed)

                avg_cormat_file = os.path.join(
                    res_path, iter_dir,
                    "prepare_mean_correl" + sess, "avg_cormat.npy")

                print(avg_cormat_file)

                if os.path.exists(avg_cormat_file):
                    avg_cormat = np.load(avg_cormat_file)
                    vect_avg_cormat = avg_cormat[triu_indices_i,
                                                 triu_indices_j]
                    all_vect_cormats.append(vect_avg_cormat)

        df_info = pd.DataFrame(all_global_info_values)

        df_con = pd.DataFrame(all_vect_cormats, columns=pair_labels)

        df = pd.concat((df_info, df_con), axis=1)

        df.to_csv(df_filename, index_col=0)
    else:

        df = pd.read_csv(df_filename, index_col=0)

    return df


def gather_con_values(res_path, cond, nb_permuts, labels):

    import os

    if isinstance(cond, tuple):
        # si plusieurs conditions = IRMf
        df_filename = os.path.join(
            res_path, "permuts_" + ".".join(cond) + '_con_values.csv')

    else:
        # si une seule valeur
        df_filename = os.path.join(
            res_path, "permuts_" + cond + '_con_values.csv')

    if not os.path.exists(df_filename):

        # pair of labels and tri triu_indices
        triu_indices_i, triu_indices_j = np.triu_indices(len(labels), k=1)

        pair_labels = [labels[i] + "_" + labels[j]
                       for i, j in zip(triu_indices_i.tolist(),
                                       triu_indices_j.tolist())]

        # creating dataframe
        all_vect_cormats = []
        all_global_info_values = []

        for seed in range(-1, nb_permuts):

            dict_global_info_values = {'Seed': seed}

            all_global_info_values.append(dict_global_info_values)

            # avg_cormat
            if isinstance(cond, tuple):
                iter_dir = "_cond_" + ".".join(cond) + "_permut_" + str(seed)

            else:
                iter_dir = "_freq_band_name_" + cond + "_permut_" + str(seed)

            avg_cormat_file = os.path.join(
                res_path, iter_dir, "shuffle_matrix", "shuffled_matrix.npy")

            if os.path.exists(avg_cormat_file):
                avg_cormat = np.load(avg_cormat_file)
                avg_cormat = avg_cormat + np.transpose(avg_cormat)
                vect_avg_cormat = avg_cormat[triu_indices_i, triu_indices_j]
                all_vect_cormats.append(vect_avg_cormat)

            else:
                print("Warning, could not find file \
                    {}".format(avg_cormat_file))

        df_info = pd.DataFrame(all_global_info_values)
        df_con = pd.DataFrame(all_vect_cormats, columns=pair_labels)
        df = pd.concat((df_info, df_con), axis=1)
        df.to_csv(df_filename, index=False)

    else:

        df = pd.read_csv(df_filename, index_col=None)

    return df


def compute_signif_permut_con_values(df, res_path, cond, alpha, labels,
                                     coords=np.array([]), diff_sess=True):

    import os
    import numpy as np
    import pandas as pd

    from graphpype.gather.gather_permuts import compute_signif_permuts
    from graphpype.utils_plot import plot_int_mat

    print(len(cond))

    if isinstance(cond, tuple):

        cond = ".".join(cond)

    print(cond)

    signif_df = os.path.join(res_path, 'signif_' +
                             cond + '_permut_con_values.csv')

    if os.path.exists(signif_df):

        df_res = pd.read_csv(signif_df, index_col=0)
    else:

        if diff_sess:
            df_res = compute_signif_permuts(df)

        else:
            df_res = compute_signif_permuts(df, session_col=-1)

        df_res.to_csv(signif_df)

    if diff_sess:
        res_higher = df_res.values[0, 2:]
        res_lower = df_res.values[1, 2:]

    else:
        res_higher = df_res.values[0, 1:]
        res_lower = df_res.values[1, 1:]

    signif_mat_higher = np.zeros(shape=(len(labels), len(labels)), dtype="int")
    signif_mat_lower = np.zeros(shape=(len(labels), len(labels)), dtype="int")

    signif_vect_higher = (res_higher < alpha) & (res_higher != -1)
    signif_vect_lower = (res_lower < alpha) & (res_lower != -1)

    triu_indices_i, triu_indices_j = np.triu_indices(
        signif_mat_higher.shape[0], k=1)

    signif_mat_higher[triu_indices_i, triu_indices_j] = signif_vect_higher
    signif_mat_lower[triu_indices_i, triu_indices_j] = signif_vect_lower

    # diff_mat
    diff_mat = np.array(signif_mat_higher - signif_mat_lower, dtype=int)

    signif_mat_plot_file = os.path.join(
        res_path, "signif_" + cond + '_permut_con_values.eps')

    plot_int_mat(plot_file=signif_mat_plot_file, cor_mat=diff_mat,
                 list_labels=labels, fix_full_range=[-1, 1], label_size=2)

    diff_mat_file = os.path.join(
        res_path, "signif_" + cond + '_permut_diff_mat.npy')
    np.save(diff_mat_file, diff_mat)
