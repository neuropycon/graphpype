# -*- coding: utf-8 -*-

import scipy.stats as stat
import numpy as np
import itertools as it


def info_CI(X, Y):
    """ Compute binomial comparaison
"""
    nX = len(X) * 1.
    nY = len(Y) * 1.

    pX = np.sum(X == 1)/nX

    pY = np.sum(Y == 1)/nY

    # print pX,pY,np.absolute(pX-pY)

    SE = np.sqrt(pX * (1-pX)/nX + pY * (1-pY)/nY)

    # print SE

    # if (np.absolute(pX-pY) > norm) == True:
    # print pX,pY,np.absolute(pX-pY),norm

    return np.absolute(pX-pY), SE, np.sign(pX-pY)

    ###################################################################################### pairwise/nodewise stats ###########################################################################################################


def return_signif_code(p_values, uncor_alpha=0.05, fdr_alpha=0.05, bon_alpha=0.05):

    # print p_values

    # print uncor_alpha,fdr_alpha,bon_alpha

    N = p_values.shape[0]

    order = p_values.argsort()

    # print order

    sorted_p_values = p_values[order]

    # print sorted_p_values

    # by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape=N)

    ################ uncor #############################
    # code = 0 for all correlation below uncor_alpha
    signif_code[p_values > uncor_alpha] = 0

    # print p_values[p_values > uncor_alpha]

    # print p_values < uncor_alpha
    # print p_values == uncor_alpha

    # print signif_code

    ################ FPcor #############################
    if 1.0/N < uncor_alpha:

        signif_code[p_values < 1.0/N] = 2

    ################ fdr ###############################
    seq = np.arange(N, 0, -1)

    seq_fdr_p_values = fdr_alpha/seq

    # print seq_fdr_p_values

    signif_sorted = sorted_p_values < seq_fdr_p_values

    signif_code[order[signif_sorted]] = 3

    ################# bonferroni #######################

    signif_code[p_values < bon_alpha/N] = 4

    # print signif_code

    return signif_code


def return_signif_code_Z(Z_values, conf_interval_binom_fdr=0.05):

    print(Z_values)

    N = Z_values.shape[0]

    order = Z_values.argsort()

    #order =  np_list_diff[:,2].argsort()

    # print order

    #sort_np_list_diff = np_list_diff[order[::-1]]

    # print order

    sorted_Z_values = Z_values[order[::-1]]

    print(sorted_Z_values)

    # by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape=N)

    ################ uncor #############################
    # code = 0 for all correlation below uncor_alpha
    Z_uncor = stat.norm.ppf(1-conf_interval_binom_fdr/2)

    print(Z_uncor)

    signif_code[Z_values < Z_uncor] = 0

    print(np.sum(signif_code != 0))

    ################ FPcor #############################

    Z_FPcor = stat.norm.ppf(1-(1.0/(2*N)))

    print(Z_FPcor)

    signif_code[Z_values > Z_FPcor] = 2

    print(np.sum(signif_code == 2))

    ################ fdr ###############################

    seq = np.arange(N, 0, -1)

    seq_fdr_p_values = conf_interval_binom_fdr/seq

    seq_Z_val = stat.norm.ppf(1-seq_fdr_p_values/2)

    # print seq_fdr_p_values

    signif_sorted = sorted_Z_values > seq_Z_val

    signif_code[order[signif_sorted]] = 3

    print(np.sum(signif_code == 3))

    ################# bonferroni #######################

    Z_bon = stat.norm.ppf(1-conf_interval_binom_fdr/(2*N))

    print(Z_bon)

    signif_code[Z_values > Z_bon] = 4

    print(np.sum(signif_code == 4))

    return signif_code

################################################ pairwise stats ##########################################


def compute_pairwise_binom(X, Y, conf_interval_binom):

    # number of nodes
    N = X.shape[0]

    # Perform binomial test at each edge
    ADJ = np.zeros((N, N), dtype='int')

    for i, j in it.combinations(list(range(N)), 2):

        ADJ[i, j] = ADJ[j, i] = binom_CI_test(
            X[i, j, :], Y[i, j, :], conf_interval_binom)

    return ADJ


def compute_pairwise_ttest_fdr(X, Y, cor_alpha, uncor_alpha, paired=True, old_order=True, keep_intracon=False):

    # number of nodes
    if old_order:
        N = X.shape[0]
    else:

        N = X.shape[1]

    if keep_intracon:
        iter_indexes = it.combinations_with_replacement(list(range(N)), 2)
    else:

        iter_indexes = it.combinations(list(range(N)), 2)

    if old_order:

        print(X.shape)

        list_diff = []

        for i, j in iter_indexes:

            X_nonan = X[i, j, np.logical_not(np.isnan(X[i, j, :]))]
            Y_nonan = Y[i, j, np.logical_not(np.isnan(Y[i, j, :]))]

            # print len(X_nonan),len(Y_nonan)

            if len(X_nonan) < 2 or len(Y_nonan) < 2:
                # if len(X_nonan) < 1 or len(Y_nonan) < 1:
                # list_diff.append([i,j,1.0,0.0])
                continue

            if paired == True:

                t_stat, p_val = stat.ttest_rel(X_nonan, Y_nonan)

                if np.isnan(p_val):

                    print("Warning, unable to compute T-test: ")
                    print(t_stat, p_val, X_nonan, Y_nonan)

                # pas encore present (version scipy 0.18)
                #t_stat,p_val = stat.ttest_rel(X[i,j,:],Y[i,j,:],nan_policy = 'omit')

            else:
                t_stat, p_val = stat.ttest_ind(X_nonan, Y_nonan)

            # print t_stat,p_val

            list_diff.append([i, j, p_val, np.sign(
                np.mean(X_nonan)-np.mean(Y_nonan)), t_stat])
    else:
        # number of nodes
        assert X.shape[1] == X.shape[2] and Y.shape[1] == Y.shape[2], "Error, X {}{} and/or Y {}{} are not squared".format(
            X.shape[1], X.shape[2], Y.shape[1], Y.shape[2])

        if paired:
            assert X.shape[0] == Y.shape[0], "Error, X and Y are paired but do not have the same number od samples{}{}".format(
                X.shape[0], Y.shape[0])

        # print X.shape

        list_diff = []

        for i, j in iter_indexes:

            X_nonan = X[np.logical_not(np.isnan(X[:, i, j])), i, j]
            Y_nonan = Y[np.logical_not(np.isnan(Y[:, i, j])), i, j]

            # print len(X_nonan),len(Y_nonan)

            if len(X_nonan) < 2 or len(Y_nonan) < 2:
                # if len(X_nonan) < 1 or len(Y_nonan) < 1:
                # list_diff.append([i,j,1.0,0.0])
                continue

            if paired == True:

                t_stat, p_val = stat.ttest_rel(X_nonan, Y_nonan)

                if np.isnan(p_val):

                    print("Warning, unable to compute T-test: ")
                    # print t_stat,p_val,X_nonan,Y_nonan

                # pas encore present (version scipy 0.18)
                #t_stat,p_val = stat.ttest_rel(X[i,j,:],Y[i,j,:],nan_policy = 'omit')

            else:
                t_stat, p_val = stat.ttest_ind(X_nonan, Y_nonan)

            # print t_stat,p_val

            list_diff.append([i, j, p_val, np.sign(
                np.mean(X_nonan)-np.mean(Y_nonan)), t_stat])

    # print list_diff

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    # print np_list_diff

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    # print np.sum(signif_code == 0.0),np.sum(signif_code == 1.0),np.sum(signif_code == 2.0),np.sum(signif_code == 3.0),np.sum(signif_code == 4.0)

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    # print np.sum(np_list_diff[:,3] == 0.0)
    # print np.sum(np_list_diff[:,3] == 1.0),np.sum(np_list_diff[:,3] == 2.0),np.sum(np_list_diff[:,3] == 3.0),np.sum(np_list_diff[:,3] == 4.0)
    # print np.sum(np_list_diff[:,3] == -1.0),np.sum(np_list_diff[:,3] == -2.0),np.sum(np_list_diff[:,3] == -3.0),np.sum(np_list_diff[:,3] == -4.0)

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    T_stat_mat = np.zeros((N, N), dtype='float')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_signed_adj_mat[signif_i, signif_j] = signif_signed_adj_mat[signif_j,
                                                                      signif_i] = np.array(np_list_diff[:, 3], dtype=int)

    p_val_mat[signif_i, signif_j] = p_val_mat[signif_j,
                                              signif_i] = np_list_diff[:, 2]
    T_stat_mat[signif_i, signif_j] = T_stat_mat[signif_j,
                                                signif_i] = np_list_diff[:, 4]

    # print T_stat_mat

    return signif_signed_adj_mat, p_val_mat, T_stat_mat


def compute_pairwise_oneway_ttest_fdr(X, cor_alpha, uncor_alpha, old_order=True):

    if old_order:

        # number of nodes
        N = X.shape[0]

        # print X.shape

        list_diff = []

        for i, j in it.combinations(list(range(N)), 2):

            X_nonan = X[i, j, np.logical_not(np.isnan(X[i, j, :]))]

            # print len(X_nonan),len(Y_nonan)

            if len(X_nonan) < 2:
                # if len(X_nonan) < 1 or len(Y_nonan) < 1:
                # list_diff.append([i,j,1.0,0.0])
                continue

            t_stat, p_val = stat.ttest_1samp(X_nonan, 0.0)

            if np.isnan(p_val):

                print("Warning, unable to compute T-test: ")
                print(t_stat, p_val, X_nonan)

            list_diff.append([i, j, p_val, np.sign(np.mean(X_nonan)), t_stat])
    else:
        # number of nodes
        assert X.shape[1] == X.shape[2] and Y.shape[1] == Y.shape[2], "Error, X {}{} and/or Y {}{} are not squared".format(
            X.shape[1], X.shape[2], Y.shape[1], Y.shape[2])

        N = X.shape[1]

        if paired:
            assert X.shape[0] == Y.shape[0], "Error, X and Y are paired but do not have the same number od samples{}{}".format(
                X.shape[0], Y.shape[0])

        # print X.shape

        list_diff = []

        for i, j in it.combinations(list(range(N)), 2):

            X_nonan = X[np.logical_not(np.isnan(X[:, i, j])), i, j]

            # print len(X_nonan),len(Y_nonan)

            if len(X_nonan) < 2:
                # if len(X_nonan) < 1 or len(Y_nonan) < 1:
                # list_diff.append([i,j,1.0,0.0])
                continue

            t_stat, p_val = stat.ttest_1samp(X_nonan, 0.0)

            if np.isnan(p_val):

                print("Warning, unable to compute T-test: ")
                print(t_stat, p_val, X_nonan, end=' ')

            list_diff.append([i, j, p_val, np.sign(np.mean(X_nonan)), t_stat])

    print(list_diff)

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    print(np.sum(np_list_diff[:, 3] == 0.0))
    print(np.sum(np_list_diff[:, 3] == 1.0), np.sum(np_list_diff[:, 3] == 2.0), np.sum(
        np_list_diff[:, 3] == 3.0), np.sum(np_list_diff[:, 3] == 4.0))
    print(np.sum(np_list_diff[:, 3] == -1.0), np.sum(np_list_diff[:, 3] == -2.0),
          np.sum(np_list_diff[:, 3] == -3.0), np.sum(np_list_diff[:, 3] == -4.0))

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    T_stat_mat = np.zeros((N, N), dtype='float')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_signed_adj_mat[signif_i, signif_j] = signif_signed_adj_mat[signif_j,
                                                                      signif_i] = np.array(np_list_diff[:, 3], dtype=int)

    p_val_mat[signif_i, signif_j] = p_val_mat[signif_j,
                                              signif_i] = np_list_diff[:, 2]
    T_stat_mat[signif_i, signif_j] = T_stat_mat[signif_j,
                                                signif_i] = np_list_diff[:, 4]

    print(T_stat_mat)

    return signif_signed_adj_mat, p_val_mat, T_stat_mat


def compute_pairwise_mannwhitney_fdr(X, Y, t_test_thresh_fdr, uncor_alpha=0.01):

    # number of nodes
    N = X.shape[0]

    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        #t_stat_zalewski = ttest2(X[i,j,:],Y[i,j,:])

        u_stat, p_val = stat.mannwhitneyu(
            X[i, j, :], Y[i, j, :], use_continuity=False)

        list_diff.append([i, j, p_val, np.sign(
            np.mean(X[i, j, :])-np.mean(Y[i, j, :]))])

    # print list_diff

    np_list_diff = np.array(list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=t_test_thresh_fdr, bon_alpha=0.05)

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    print(np.sum(np_list_diff[:, 3] == 0.0))
    print(np.sum(np_list_diff[:, 3] == 1.0), np.sum(np_list_diff[:, 3] == 2.0), np.sum(
        np_list_diff[:, 3] == 3.0), np.sum(np_list_diff[:, 3] == 4.0))
    print(np.sum(np_list_diff[:, 3] == -1.0), np.sum(np_list_diff[:, 3] == -2.0),
          np.sum(np_list_diff[:, 3] == -3.0), np.sum(np_list_diff[:, 3] == -4.0))

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_sign = np.array(np_list_diff[:, 3], dtype=int)

    signif_signed_adj_mat[signif_i,
                          signif_j] = signif_signed_adj_mat[signif_j, signif_i] = signif_sign

    # print signif_signed_adj_mat

    return signif_signed_adj_mat


def compute_pairwise_binom_fdr(X, Y, conf_interval_binom_fdr):

    # number of nodes
    N = X.shape[0]

    # Perform binomial test at each edge

    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        abs_diff, SE, sign_diff = info_CI(X[i, j, :], Y[i, j, :])

        list_diff.append([i, j, abs_diff/SE, sign_diff])

    print(list_diff)

    np_list_diff = np.array(list_diff)

    signif_code = return_signif_code_Z(
        np_list_diff[:, 2], conf_interval_binom_fdr)

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    print(np.sum(np_list_diff[:, 3] == 0.0))
    print(np.sum(np_list_diff[:, 3] == 1.0), np.sum(np_list_diff[:, 3] == 2.0), np.sum(
        np_list_diff[:, 3] == 3.0), np.sum(np_list_diff[:, 3] == 4.0))
    print(np.sum(np_list_diff[:, 3] == -1.0), np.sum(np_list_diff[:, 3] == -2.0),
          np.sum(np_list_diff[:, 3] == -3.0), np.sum(np_list_diff[:, 3] == -4.0))

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_sign = np.array(np_list_diff[:, 3], dtype=int)

    # print signif_i,signif_j

    # print signif_signed_adj_mat[signif_i,signif_j]

    # print signif_sign

    signif_signed_adj_mat[signif_i,
                          signif_j] = signif_signed_adj_mat[signif_j, signif_i] = signif_sign

    # print signif_signed_adj_mat

    return signif_signed_adj_mat

# OneWay Anova (F-test)


def compute_oneway_anova_fwe(list_of_list_matrices, cor_alpha=0.05, uncor_alpha=0.001, keep_intracon=False):

    for group_mat in list_of_list_matrices:
        assert group_mat.shape[1] == group_mat.shape[2], "warning, matrices are not squared {} {}".format(
            group_mat.shape[1], group_mat.shape[2])

    N = group_mat.shape[2]

    list_diff = []

    if keep_intracon:
        iter_indexes = it.combinations_with_replacement(list(range(N)), 2)

    else:
        iter_indexes = it.combinations(list(range(N)), 2)

    for i, j in iter_indexes:

        # print i,j

        list_val = [group_mat[:, i, j].tolist()
                    for group_mat in list_of_list_matrices]

        # print list_val

        F_stat, p_val = stat.f_oneway(*list_val)

        # print F_stat,p_val

        list_diff.append([i, j, p_val, F_stat])

    ############### computing significance code ################

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    signif_code[np.isnan(np_list_diff[:, 2])] = 0

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    # converting to matrix

    signif_adj_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    F_stat_mat = np.zeros((N, N), dtype='float')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_adj_mat[signif_i, signif_j] = signif_adj_mat[signif_j,
                                                        signif_i] = signif_code
    p_val_mat[signif_i, signif_j] = p_val_mat[signif_i,
                                              signif_j] = np_list_diff[:, 2]
    F_stat_mat[signif_i, signif_j] = F_stat_mat[signif_i,
                                                signif_j] = np_list_diff[:, 3]

    # print signif_adj_mat
    # print p_val_mat
    # print F_stat_mat

    return signif_adj_mat, p_val_mat, F_stat_mat


def compute_correl_behav(X, reg_interest, uncor_alpha=0.001, cor_alpha=0.05, old_order=False, keep_intracon=False):

    import numpy as np
    import itertools as it

    import scipy.stats as stat

    from graphpype.utils_stats import return_signif_code
    if old_order:
        N = X.shape[0]
    else:
        N = X.shape[1]

    print(reg_interest)
    print(reg_interest.dtype)

    if keep_intracon:
        iter_indexes = it.combinations_with_replacement(list(range(N)), 2)
    else:
        iter_indexes = it.combinations(list(range(N)), 2)

    if not old_order:
        # number of nodes
        assert X.shape[1] == X.shape[2] and "Error, X {}{} is not squared".format(
            X.shape[1], X.shape[2])

        assert X.shape[0] == reg_interest.shape[0], "Incompatible number of fields in dataframe and nb matrices"

        list_diff = []

        for i, j in iter_indexes:

            keep_val = (~np.isnan(X[:, i, j])) & (~np.isnan(reg_interest))

            print(keep_val)

            r_stat, p_val = stat.pearsonr(
                X[keep_val, i, j], reg_interest[keep_val])

            print(r_stat, p_val)

            if np.isnan(p_val):

                print("Warning, unable to compute T-test: ")
                print(r_stat, p_val, X_nonan, Y_nonan)

                # pas encore present (version scipy 0.18)
                #t_stat,p_val = stat.ttest_rel(X[i,j,:],Y[i,j,:],nan_policy = 'omit')

            # print t_stat,p_val

            list_diff.append([i, j, p_val, np.sign(r_stat), r_stat])

    print(list_diff)

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    print(np.sum(np_list_diff[:, 3] == 0.0))
    print(np.sum(np_list_diff[:, 3] == 1.0), np.sum(np_list_diff[:, 3] == 2.0), np.sum(
        np_list_diff[:, 3] == 3.0), np.sum(np_list_diff[:, 3] == 4.0))
    print(np.sum(np_list_diff[:, 3] == -1.0), np.sum(np_list_diff[:, 3] == -2.0),
          np.sum(np_list_diff[:, 3] == -3.0), np.sum(np_list_diff[:, 3] == -4.0))

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    r_stat_mat = np.zeros((N, N), dtype='float')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_signed_adj_mat[signif_i, signif_j] = signif_signed_adj_mat[signif_j,
                                                                      signif_i] = np.array(np_list_diff[:, 3], dtype=int)

    p_val_mat[signif_i, signif_j] = p_val_mat[signif_j,
                                              signif_i] = np_list_diff[:, 2]
    r_stat_mat[signif_i, signif_j] = r_stat_mat[signif_j,
                                                signif_i] = np_list_diff[:, 4]

    print(r_stat_mat)

    return signif_signed_adj_mat, p_val_mat, r_stat_mat


############### nodewise stats #########################
def compute_nodewise_t_test_vect(d_stacked, nx, ny):

    print(d_stacked.shape)

    assert d_stacked.shape[1] == nx + ny

    t1 = time.time()

    t_val_vect = compute_nodewise_t_values(
        d_stacked[:, :nx], d_stacked[:, nx:nx+ny])

    t2 = time.time()

    print("computation took %f" % (t2-t1))

    return t_val_vect

######################## correl ######################################


def compute_pairwise_correl_fdr(X, behav_score, correl_thresh_fdr):

    from scipy.stats.stats import pearsonr

    # number of nodes
    N = X.shape[0]

    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        #t_stat_zalewski = ttest2(X[i,j,:],Y[i,j,:])

        r_stat, p_val = pearsonr(X[i, j, :], behav_score)

        # print i,j,p_val,r_stat

        list_diff.append([i, j, p_val, np.sign(r_stat)])

    # print list_diff

    np_list_diff = np.array(list_diff)

    np_list_diff = np.array(list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=0.001, fdr_alpha=correl_thresh_fdr, bon_alpha=0.05)

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    print(np.sum(np_list_diff[:, 3] == 0.0))
    print(np.sum(np_list_diff[:, 3] == 1.0), np.sum(np_list_diff[:, 3] == 2.0), np.sum(
        np_list_diff[:, 3] == 3.0), np.sum(np_list_diff[:, 3] == 4.0))
    print(np.sum(np_list_diff[:, 3] == -1.0), np.sum(np_list_diff[:, 3] == -2.0),
          np.sum(np_list_diff[:, 3] == -3.0), np.sum(np_list_diff[:, 3] == -4.0))

    signif_signed_adj_mat = np.zeros((N, N), dtype='int')

    signif_i = np.array(np_list_diff[:, 0], dtype=int)
    signif_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_sign = np.array(np_list_diff[:, 3], dtype=int)

    print(signif_i, signif_j)

    print(signif_signed_adj_mat[signif_i, signif_j])

    # print signif_sign

    signif_signed_adj_mat[signif_i,
                          signif_j] = signif_signed_adj_mat[signif_j, signif_i] = signif_sign

    print(signif_signed_adj_mat)

    return signif_signed_adj_mat
