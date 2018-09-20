# -*- coding: utf-8 -*-

import scipy.stats as stat
import numpy as np
import itertools as it

"""
Functions for computing statistics over symetrical matrices (pairwise)

Input parameters:

All functions accept two (or one for test vs 0) stack of matrices (3D objects).

Old order (old_order = True) refers to a framework where the last dimension was referring to the stacking of matrices (10*10*20 for 20 samples of matrices with 10 nodes) whereas the new order (i.e. old_order = False

keep_intracon=True means computing the tests of the diagonal as well. It is relevant for the test of number of inter-modular edges (in the case the diagonal will be the number of edges within a module). For fonctional connectivity the inclusion of diagonal is irrelevant.

Returns:
All the functions returns 3 objects as a tuple:
- a matrix of significance level, with the following code:
0 -> not significant
1 -> uncorrected significant (according to the parameter uncor_alpha, setting the uncorrected threshold)
2 -> FP significant (i.e., passing a threshold of 1/N_tests, given it is more stringent than the uncor_alpha)
    This threshold is mildly accepted in the scientific community but may be justified in case of graph connectivity, see Lynall and Bassett, 2013
3 -> FDR significant (as set as the fdr_alpha, = cor_alpha in most cases)
4 -> Bonferroni significant (as set as the bon_alpha  = cor_alpha in most cases)
The sign of the code (+1 or -1 give the direction of the significant: +1: X > Y, -1: Y > X)

- a matrix of p-value associated with the statitical test

- the statistics value (T values for t-test, R^2 for correlation, etc)

"""

###################################################################################### pairwise/nodewise stats ###########################################################################################################


def return_signif_code(p_values, uncor_alpha=0.001, fdr_alpha=0.05, bon_alpha=0.05):

    N = p_values.shape[0]

    ################# by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape=N)

    ################ uncor #############################
    # code = 0 for all correlation below uncor_alpha
    signif_code[p_values >= uncor_alpha] = 0

    ################ FPcor #############################
    if 1.0/N < uncor_alpha:

        signif_code[p_values < 1.0/N] = 2

    ################ fdr ###############################
    seq = np.arange(N, 0, -1)

    seq_fdr_p_values = fdr_alpha/seq

    order = p_values.argsort()

    signif_sorted = p_values[order] < seq_fdr_p_values

    signif_code[order[signif_sorted]] = 3

    ################# bonferroni #######################

    signif_code[p_values < bon_alpha/N] = 4

    return signif_code


def return_signif_code_Z(Z_values, uncor_alpha=0.001, fdr_alpha = 0.05, bon_alpha = 0.05):

    N = Z_values.shape[0]

    ###################### by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape=N)

    ################ uncor #############################
    
    Z_uncor = stat.norm.ppf(1-uncor_alpha/2)

    signif_code[Z_values < Z_uncor] = 0

    print (signif_code)
    ################ FPcor #############################

    Z_FPcor = stat.norm.ppf(1-(1.0/(2*N)))

    signif_code[Z_values > Z_FPcor] = 2

    print (signif_code)
    
    ################ fdr ###############################

    seq = np.arange(N, 0, -1)

    seq_fdr_p_values = fdr_alpha/seq

    seq_Z_val = stat.norm.ppf(1-seq_fdr_p_values/2)

    order = (-Z_values).argsort() ### classé dans l'ordre inverse

    sorted_Z_values = Z_values[order] ### classé dans l'ordre inverse

    signif_sorted = sorted_Z_values > seq_Z_val

    signif_code[order[signif_sorted]] = 3

    ################# bonferroni #######################

    Z_bon = stat.norm.ppf(1-bon_alpha/(2*N))

    signif_code[Z_values > Z_bon] = 4

    return signif_code

################################################ pairwise stats ##########################################

########################### T-test
def compute_pairwise_ttest_fdr(X, Y, cor_alpha, uncor_alpha, paired=True, old_order=True, keep_intracon=False):

    ### old order was n_nodes, n_nodes, sample_size
    ### new order is sample_size, n_nodes, n_nodes,
    
    # number of nodes
    if old_order:
        X = np.moveaxis(X,2,0)
        Y = np.moveaxis(Y,2,0)
    
    # test squared matrices
    assert X.shape[1] == X.shape[2] and Y.shape[1] == Y.shape[2], "Error, X {} {} and/or Y {} {} are not squared".format(
        X.shape[1], X.shape[2], Y.shape[1], Y.shape[2])

    # test same number of nodes between X and Y
    assert X.shape[1] == Y.shape[1] , "Error, X {} and Y {}do not have the same number of nodes".format(
        X.shape[1], Y.shape[1])

    # test if same number of sample (paired t-test only)
    if paired:
        assert X.shape[0] == Y.shape[0], "Error, X and Y are paired but do not have the same number od samples{}{}".format(
            X.shape[0], Y.shape[0])

    ### info nodes
    N = X.shape[1]

    ### test are also done on the diagonal of the matrix
    if keep_intracon:
        iter_indexes = it.combinations_with_replacement(list(range(N)), 2)
    else:
        iter_indexes = it.combinations(list(range(N)), 2)

    ### computing signif t-tests for each relevant pair 
    list_diff = []

    for i, j in iter_indexes:

        print (X[:,i,j])
        
        ### removing the nan
        X_nonan = X[np.logical_not(np.isnan(X[:, i, j])), i, j]
        Y_nonan = Y[np.logical_not(np.isnan(Y[:, i, j])), i, j]

        print (X_nonan)
        
        if len(X_nonan) < 2 or len(Y_nonan) < 2:
            print ("Not enough values for sample {} {}, len = {} and {}, skipping".format(i,j,len(X_nonan),len(Y_nonan)))
            continue

        if paired == True:
            t_stat, p_val = stat.ttest_rel(X_nonan, Y_nonan)
            if np.isnan(p_val):
                print("Warning, unable to compute T-test: ")

            # pas encore present (version scipy 0.18)
            #t_stat,p_val = stat.ttest_rel(X[i,j,:],Y[i,j,:],nan_policy = 'omit')

        else:
            t_stat, p_val = stat.ttest_ind(X_nonan, Y_nonan)

        list_diff.append([i, j, p_val, np.sign(
            np.mean(X_nonan)-np.mean(Y_nonan)), t_stat])

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    np_list_diff[:, 3] *=  signif_code

    #### formatting signif_mat, p_val_mat and T_stat_mat
    signif_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    T_stat_mat = np.zeros((N, N), dtype='float')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_mat[s_i, s_j] = signif_mat[s_j,s_i] = np_list_diff[:, 3].astype(int)
    p_val_mat[s_i, s_j] = p_val_mat[s_j,s_i] = np_list_diff[:, 2]
    T_stat_mat[s_i, s_j] = T_stat_mat[s_j,s_i] = np_list_diff[:, 4]

    return signif_mat, p_val_mat, T_stat_mat


def compute_pairwise_oneway_ttest_fdr(X, cor_alpha, uncor_alpha, old_order=True):

    if old_order:
        X = np.moveaxis(X,2,0)

    # number of nodes
    assert X.shape[1] == X.shape[2],  "Error, X {}{} and/or Y {}{} are not squared".format(
        X.shape[1], X.shape[2])

    N = X.shape[1]

    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        X_nonan = X[np.logical_not(np.isnan(X[:, i, j])), i, j]

        if len(X_nonan) < 2:
            print ("Not enough values for sample {} {}, len = {}, skipping".format(i,j,len(X_nonan)) )   
            continue

        t_stat, p_val = stat.ttest_1samp(X_nonan, 0.0) ## 0.0 ?

        if np.isnan(p_val):

            print("Warning, unable to compute T-test: ")
            print(t_stat, p_val, X_nonan)

        list_diff.append([i, j, p_val, np.sign(np.mean(X_nonan)), t_stat])

    print(list_diff)

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    np_list_diff[:, 3] *= signif_code

    signif_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    T_stat_mat = np.zeros((N, N), dtype='float')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_mat[s_i, s_j] = signif_mat[s_j,s_i] = np_list_diff[:, 3].astype(int)
    p_val_mat[s_i, s_j] = p_val_mat[s_j,s_i] = np_list_diff[:, 2]
    T_stat_mat[s_i, s_j] = T_stat_mat[s_j,s_i] = np_list_diff[:, 4]

    print(T_stat_mat)

    return signif_mat, p_val_mat, T_stat_mat

################################## Mann Whitney
def compute_pairwise_mannwhitney_fdr(X, Y, cor_alpha, uncor_alpha=0.01, old_order = True):
    ### modified to be compatible with old_order = True (was only developed for old order) + assert
    ### TODO : test if OK with moveaxis and 'new order'? 
    ### TODO : return parameter of mannwhitneyu (i.e. "U" values)?
    
    if old_order:
        X = np.moveaxis(X,2,0)
        Y = np.moveaxis(Y,2,0)

    #### Assert
    
    # test squared matrices
    assert X.shape[1] == X.shape[2] and Y.shape[1] == Y.shape[2], "Error, X {} {} and/or Y {} {} are not squared".format(
        X.shape[1], X.shape[2], Y.shape[1], Y.shape[2])

    # test same number of nodes between X and Y
    assert X.shape[1] == Y.shape[1] , "Error, X {} and Y {}do not have the same number of nodes".format(
        X.shape[1], Y.shape[1])

    # number of nodes
    N = X.shape[1]

    #### compute
    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        #### TODO: handles nan correctly??
        X_val = X[:, i, j]
        Y_val = Y[:, i, j]
        u_stat, p_val = stat.mannwhitneyu(X_val, Y_val, use_continuity=False, alternative = "two-sided")

        sign_diff = np.sign(np.mean(X_val)-np.mean(Y_val))
                            
        list_diff.append([i, j, p_val, sign_diff])

    np_list_diff = np.array(list_diff)

    print (np_list_diff)
    signif_code = return_signif_code(np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    signif_mat = np.zeros((N, N), dtype='int')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)
    signif_sign = np.array(np_list_diff[:, 3], dtype=int)

    signif_mat[s_i,s_j] = signif_mat[s_j, s_i] = signif_sign

    return signif_mat


#### private function for compute_pairwise_binom_fdr
def _info_CI(X, Y):
    """ Compute binomial comparaison"""
    nX = len(X) * 1.
    nY = len(Y) * 1.

    pX = np.sum(X == 1)/nX
    pY = np.sum(Y == 1)/nY

    SE = np.sqrt(pX * (1-pX)/nX + pY * (1-pY)/nY)

    return np.absolute(pX-pY), SE, np.sign(pX-pY)

def compute_pairwise_binom_fdr(X, Y, uncor_alpha=0.001, cor_alpha = 0.05 , old_order = True):
    ### modified to be compatible with old_order = True (was only developed for old order) + assert
    ### TODO : test if OK with moveaxis and 'new order'? 
    
    if old_order:
        X = np.moveaxis(X,2,0)
        Y = np.moveaxis(Y,2,0)

    #### Assert
    
    # test squared matrices
    assert X.shape[1] == X.shape[2] and Y.shape[1] == Y.shape[2], "Error, X {} {} and/or Y {} {} are not squared".format(
        X.shape[1], X.shape[2], Y.shape[1], Y.shape[2])

    # test same number of nodes between X and Y
    assert X.shape[1] == Y.shape[1] , "Error, X {} and Y {}do not have the same number of nodes".format(
        X.shape[1], Y.shape[1])

    # number of nodes
    N = X.shape[1]


    # Perform binomial test at each edge
    list_diff = []

    for i, j in it.combinations(list(range(N)), 2):

        abs_diff, SE, sign_diff = _info_CI(X[:, i, j], Y[:, i, j])

        list_diff.append([i, j, abs_diff/SE, sign_diff])

    np_list_diff = np.array(list_diff)

    signif_code = return_signif_code_Z(np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha = cor_alpha, bon_alpha = cor_alpha)

    np_list_diff[:, 3] = np_list_diff[:, 3] * signif_code

    signif_mat = np.zeros((N, N), dtype='int')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_sign = np.array(np_list_diff[:, 3], dtype=int)

    signif_mat[s_i,s_j] = signif_mat[s_j, s_i] = signif_sign

    return signif_mat

### TODO : warning, this is very different than previous functions, needs to be checked where it is called
# OneWay Anova (F-test)
def compute_oneway_anova_fwe(list_of_list_matrices, cor_alpha=0.05, uncor_alpha=0.001, keep_intracon=False):

    assert False, "Warning, very old function, check your call and report it to developer"
    
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

        list_val = [group_mat[:, i, j].tolist()
                    for group_mat in list_of_list_matrices]

        F_stat, p_val = stat.f_oneway(*list_val)

        list_diff.append([i, j, p_val, F_stat])

    ############### computing significance code ################

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    signif_code[np.isnan(np_list_diff[:, 2])] = 0

    print(np.sum(signif_code == 0.0), np.sum(signif_code == 1.0), np.sum(
        signif_code == 2.0), np.sum(signif_code == 3.0), np.sum(signif_code == 4.0))
    # converting to matrix

    signif_adj_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    F_stat_mat = np.zeros((N, N), dtype='float')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_adj_mat[s_i, s_j] = signif_adj_mat[s_j, s_i] = signif_code
    p_val_mat[s_i, s_j] = p_val_mat[s_i,s_j] = np_list_diff[:, 2]
    F_stat_mat[s_i, s_j] = F_stat_mat[s_i,s_j] = np_list_diff[:, 3]

    # print signif_adj_mat
    # print p_val_mat
    # print F_stat_mat

    return signif_adj_mat, p_val_mat, F_stat_mat

######################################### correlation
def compute_correl_behav(X, reg_interest, uncor_alpha=0.001, cor_alpha=0.05, old_order=False, keep_intracon=False):

    import numpy as np
    import itertools as it

    import scipy.stats as stat

    if old_order:
       X = X.moveaxis(X, 0,2)
       
    N = X.shape[1]

    print(reg_interest)
    print(reg_interest.dtype)

    if keep_intracon:
        iter_indexes = it.combinations_with_replacement(list(range(N)), 2)
    else:
        iter_indexes = it.combinations(list(range(N)), 2)

    # number of nodes
    assert X.shape[1] == X.shape[2] and "Error, X {}{} is not squared".format(
        X.shape[1], X.shape[2])

    assert X.shape[0] == reg_interest.shape[0], "Incompatible number of fields in dataframe and nb matrices"

    list_diff = []

    for i, j in iter_indexes:

        keep_val = (~np.isnan(X[:, i, j])) & (~np.isnan(reg_interest))
        print(keep_val)
        
        X_nonan = X[keep_val,i,j]
        reg_nonan = reg_interest[keep_val]

        r_stat, p_val = stat.pearsonr(X_nonan, reg_nonan)

        print(r_stat, p_val)

        if np.isnan(p_val):

            print("Warning, unable to compute T-test: ")
            print(r_stat, p_val, X_nonan, Y_nonan)

        list_diff.append([i, j, p_val, np.sign(r_stat), r_stat])

    print(list_diff)

    assert len(list_diff) != 0, "Error, list_diff is empty"

    np_list_diff = np.array(list_diff)

    print(np_list_diff)

    signif_code = return_signif_code(
        np_list_diff[:, 2], uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)

    np_list_diff[:, 3] *= signif_code

    signif_mat = np.zeros((N, N), dtype='int')
    p_val_mat = np.zeros((N, N), dtype='float')
    r_stat_mat = np.zeros((N, N), dtype='float')

    s_i = np.array(np_list_diff[:, 0], dtype=int)
    s_j = np.array(np_list_diff[:, 1], dtype=int)

    signif_mat[s_i, s_j] = signif_mat[s_j,s_i] = np_list_diff[:, 3].astype(int)

    p_val_mat[s_i, s_j] = p_val_mat[s_j,s_i] = np_list_diff[:, 2]
    r_stat_mat[s_i, s_j] = r_stat_mat[s_j,s_i] = np_list_diff[:, 4]

    print(r_stat_mat)

    return signif_mat, p_val_mat, r_stat_mat
