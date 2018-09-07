
import os

import pandas as pd
import numpy as np
#import statsmodels as stat
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
#import matplotlib.pyplot as plt
#import nibabel.gifti as gio
#from statsmodels.stats.outliers_influence import OLSInfluence

from itertools import product, combinations
from numpy import isnan, nan, logical_not, logical_or

from graphpype.utils_stats import compute_oneway_anova_fwe, compute_pairwise_ttest_fdr
from graphpype.utils_cor import return_corres_correl_mat


def isInAlphabeticalOrder(word):
    return list(word) == sorted(word)
    # return word==''.join(sorted(word))


def return_all_iter_cormats(cormat_path, iterables, iternames, gm_mask_coords_file=0, gm_mask_labels_file=0, mapflow_iterables=0, mapflow_iternames=0, export_df=False):
    """
    gm_mask_coords_file is the coords commun to all analyses before Inteersectmask
    """
    print(product(*iterables))

    all_iter_cormats = []
    all_descriptors = []

    assert isInAlphabeticalOrder(
        iternames), "Warning, iternames are not in alphabetical oroder, check the iterables order as well"

    if gm_mask_coords_file != 0:
        gm_mask_coords = np.loadtxt(gm_mask_coords_file)

        print(gm_mask_coords)

    if export_df:
        writer = pd.ExcelWriter(os.path.join(cormat_path, "all_cormats.xls"))

    for iter_obj in product(*iterables):

        print(iter_obj)

        assert len(iter_obj) == len(
            iternames), "Error, different number of iternames and iterables"

        iter_dir = "".join(["_" + zip_iter[0].strip() + "_" + zip_iter[1].strip()
                            for zip_iter in zip(iternames, iter_obj)])

        print(iter_dir)

        if mapflow_iterables == 0:

            cormat_file = os.path.join(
                cormat_path, iter_dir, "compute_conf_cor_mat", "Z_cor_mat_resid_ts.npy")

            if os.path.exists(cormat_file):
                cormat = np.load(cormat_file)
                print(cormat.shape)

                if gm_mask_coords_file != 0:
                    coords_file = os.path.join(
                        cormat_path, iter_dir, "extract_mean_ROI_ts", "subj_coord_rois.txt")
                    #coords_file = os.path.join(cormat_path,iter_dir,"filter_ROI_mask_with_GM","filtered_coords_rois.txt")
                    coords = np.loadtxt(coords_file)

                    cormat, _ = return_corres_correl_mat(
                        cormat, coords, gm_mask_coords)

                if export_df:
                    if gm_mask_labels_file:
                        labels = [line.strip()
                                  for line in open(gm_mask_labels_file)]
                    else:
                        labels = list(range(cormat.shape[0]))

                    df = pd.DataFrame(cormat, columns=labels, index=labels)
                    print(df)

                    df.to_excel(writer, "_".join(iter_obj))

                print(cormat.shape)

                all_iter_cormats.append(cormat)
                all_descriptors.append(iter_obj)

            else:
                print("Warning, file {}  could not be found".format(cormat_file))

        else:
            for i, map_iter in enumerate(mapflow_iterables):

                print(map_iter)

                cormat_file = os.path.join(cormat_path, iter_dir, "compute_conf_cor_mat",
                                           "mapflow", "_compute_conf_cor_mat"+str(i), "Z_cor_mat_resid_ts.npy")

                new_iter_obj = list(iter_obj)

                print(new_iter_obj)

                new_iter_obj.append(str(map_iter))

                print(new_iter_obj)

                if os.path.exists(cormat_file):
                    cormat = np.load(cormat_file)
                    print(cormat.shape)

                    if gm_mask_coords_file != 0:
                        coords_file = os.path.join(
                            cormat_path, iter_dir, "extract_mean_ROI_ts", "subj_coord_rois.txt")
                        #coords_file = os.path.join(cormat_path,iter_dir,"filter_ROI_mask_with_GM","filtered_coords_rois.txt")
                        coords = np.loadtxt(coords_file)

                        cormat, _ = return_corres_correl_mat(
                            cormat, coords, gm_mask_coords)

                    if export_df:
                        if gm_mask_labels_file:
                            labels = [line.strip()
                                      for line in open(gm_mask_labels_file)]
                        else:
                            labels = list(range(cormat.shape[0]))

                        df = pd.DataFrame(cormat, columns=labels, index=labels)
                        print(df)

                        df.to_excel(writer, "_".join(new_iter_obj))

                    print(cormat.shape)

                    all_iter_cormats.append(cormat)
                    all_descriptors.append(new_iter_obj)

                else:
                    print("Warning, file {}  could not be found".format(cormat_file))

    # if export_df:
        # writer.save()

    print(np.array(all_iter_cormats).shape)

    if mapflow_iternames != 0:
        iternames .append(mapflow_iternames)
    pd_all_descriptors = pd.DataFrame(all_descriptors, columns=iternames)

    if export_df:
        pd_all_descriptors.to_excel(os.path.join(
            cormat_path, "all_descriptors.xls"))

    return np.array(all_iter_cormats), pd_all_descriptors

###################################### stats over cormats, mean and T-Test of F-Test ###############################


def compute_mean_cormats(all_cormats, all_descriptors, descript_columns):

    print("In compute_mean_cormats")

    dict_mean = {}

    if 'all' in descript_columns:

        dict_mean['all'] = np.mean(all_cormats, axis=0)

        descript_columns.remove('all')

    if len(descript_columns) != 0:

        for column in descript_columns:

            assert column in all_descriptors.columns, "Error, {} not in {}".format(
                column, all_descriptors.columns)

        for elem, lines in all_descriptors.groupby(by=descript_columns):

            print(elem)
            print(lines)
            print(lines.index)

            print(all_cormats.shape)

            elem_cormats = all_cormats[lines.index, :, :]

            print(elem_cormats.shape)

            mean_elem = np.mean(elem_cormats, axis=0)

            print(mean_elem.shape)

            dict_mean[elem] = mean_elem

    return dict_mean


def compute_stats_cormats(all_cormats, all_descriptors, descript_columns, groups=[], keep_intracon=False, cor_alpha=0.05, uncor_alpha=0.01):

    print(all_cormats.shape)

    for column in descript_columns:

        assert column in all_descriptors.columns, "Error, {} not in {}".format(
            column, all_descriptors.columns)

    dict_signif = {}
    dict_p_val = {}
    dict_stats = {}

    for column in descript_columns:

        print(column)

        if len(groups) == 0:
            groups = all_descriptors[column].unique().tolist()

        # compute F-test over matrices

        list_of_list_matrices = [
            all_cormats[all_descriptors[all_descriptors[column] == cond_name].index, :, :] for cond_name in groups]

        print(list_of_list_matrices)
        print(len(list_of_list_matrices))
        print(np.array(list_of_list_matrices).shape)

        signif_adj_mat, p_val_mat, F_stat_mat = compute_oneway_anova_fwe(
            list_of_list_matrices, cor_alpha=cor_alpha, uncor_alpha=uncor_alpha, keep_intracon=keep_intracon)

        # print signif_adj_mat
        # print p_val_mat
        # print F_stat_mat

        dict_signif["F-test_" + column] = signif_adj_mat
        dict_p_val["F-test_" + column] = p_val_mat
        dict_stats["F-test_" + column] = F_stat_mat

        for combi_pair in combinations(groups, 2):
            pair_name = "-".join(combi_pair)
            print(pair_name)

            print(combi_pair[0])

            try:
                signif_adj_mat, p_val_mat, T_stat_mat = compute_pairwise_ttest_fdr(X=list_of_list_matrices[groups.index(combi_pair[0])],
                                                                                   Y=list_of_list_matrices[groups.index(
                                                                                       combi_pair[1])],
                                                                                   cor_alpha=cor_alpha, uncor_alpha=uncor_alpha, paired=True, old_order=False, keep_intracon=keep_intracon)

                print(T_stat_mat)

                dict_signif["T-test_" + pair_name] = signif_adj_mat
                dict_p_val["T-test_" + pair_name] = p_val_mat
                dict_stats["T-test_" + pair_name] = T_stat_mat

            except AssertionError:
                print("Stop running after {} was wrong".format(pair_name))

        print(column)
        print(list(dict_signif.keys()))

    return dict_signif, dict_p_val, dict_stats


if __name__ == '__main__':

    test1 = isInAlphabeticalOrder(["a", "b", "c"])
    print(test1)

    test2 = isInAlphabeticalOrder(["ab", "ba", "ca"])
    print(test2)
