#######################################################################################################################
### test over file handling (.net, .lol, ect) required for radatools and possibly other graph tools (Louvain-Traag) ###
#######################################################################################################################

import os

import numpy as np
import scipy.stats as stat

from graphpype.utils_stats import (return_signif_code,return_signif_code_Z,
                                   compute_pairwise_ttest_fdr, compute_pairwise_oneway_ttest_fdr,
                                   compute_pairwise_binom_fdr, compute_pairwise_mannwhitney_fdr)


############################################# test signif_code #####################################################

#### building p-values vector
N = 100 # size of vector
cor_alpha = 0.05
uncor_alpha = 0.05
epsilon = 0.0001

### bonferroni p-values
N_bon = 3
p_bon = np.array([cor_alpha/N - epsilon]*N_bon, dtype = float).reshape(-1,1)

### FDR values (significance should be cut in the range)
N_fdr = 10
p_fdr = np.array([cor_alpha/N]*N_fdr, dtype = float)

### modulating between cor_alpha and uncor_alpha
### significance should vary between fdr, FP, and uncor
p_fdr = p_fdr + np.array([uncor_alpha - cor_alpha/N]*N_fdr, dtype = float)*np.arange(N_fdr)/N_fdr 


p_fdr = p_fdr.reshape(-1,1)

### uncorrected values
N_uncor = 50
p_uncor = (np.array([uncor_alpha - epsilon]*N_uncor, dtype = float)).reshape(-1,1)

N_unsignif = N - (N_bon + N_fdr + N_uncor)
p_unsignif = (np.array([0.5]*N_unsignif, dtype = float)).reshape(-1,1)

p_val = np.concatenate((p_bon,p_fdr,p_uncor,p_unsignif),axis = 0).reshape(-1)

#### equivalent Z_val

Z_val = stat.norm.ppf(1-p_val/2)

def test_return_signif_code():

    print (p_val)

    res = return_signif_code(p_val, uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)
    
    #Counting by hand result
    #[1] 10+9+5+10+9+5+10 = 58
    #[0] 9  + 10+9+5 +4 = 37 
    print (res)
    
    assert all(res == np.array([4] * 3 + [3] + [2] + [1]*58 + [0]*37)), "Error in the significance code"
    
def test_return_signif_code_Z():

    print (Z_val)

    res = return_signif_code_Z(Z_val, uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha, bon_alpha=cor_alpha)
    
    #Counting by hand result
    #[1] 10+9+5+10+9+5+10 = 58
    #[0] 9  + 10+9+5 +4 = 37 
    print (res)
    
    assert all(res == np.array([4] * 3 + [3] + [2] + [1]*58 + [0]*37)), "Error in the significance code vector"
    
############################################################ test pairwise testing

#compute two samples of matrices to test

sample_size = 20
array_size = 10

#### new order (sample_size = first dimension)
new_sample_X = np.random.rand(sample_size, array_size, array_size)*np.random.choice([-1,1], size = (sample_size, array_size, array_size))
new_sample_Y = np.random.rand(sample_size, array_size, array_size)*np.random.choice([-1,1], size = (sample_size, array_size, array_size))

#### old order (sample_size = last dimension)
old_sample_X = np.rollaxis(new_sample_X,0,3)
old_sample_Y = np.rollaxis(new_sample_Y,0,3)

print (old_sample_X.shape)
print (new_sample_X.shape)
    
def test_compute_pairwise_ttest_fdr():
    
    ### testing old order
    print (old_sample_X.shape)
    old_res = compute_pairwise_ttest_fdr(old_sample_X,old_sample_Y, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = True)
    
    print (old_res)
    
    ### testing new order
    print (new_sample_X.shape)
    
    new_res = compute_pairwise_ttest_fdr(new_sample_X,new_sample_Y, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = False)
    
    print (new_res)
    
    ### testing if both are equivalent
    assert (new_res[0] == old_res[0]).all(), "old and new order should be equal, but {} != {}".format(new_res[0], old_res[0])


def test_compute_pairwise_ttest_fdr():
    
    ### testing old order
    print (old_sample_X.shape)
    old_res = compute_pairwise_ttest_fdr(old_sample_X,old_sample_Y, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = True)
    
    print (old_res)
    
    ### testing new order
    print (new_sample_X.shape)
    
    new_res = compute_pairwise_ttest_fdr(new_sample_X,new_sample_Y, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = False)
    
    print (new_res)
    
    ### testing if both are equivalent
    assert (new_res[0] == old_res[0]).all(), "old and new order should be equal, but {} != {}".format(new_res[0], old_res[0])

def test_compute_pairwise_oneway_ttest_fdr():

    ### testing old order
    print (old_sample_X.shape)
    old_res = compute_pairwise_oneway_ttest_fdr(old_sample_X, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = True)
    
    print (old_res)
    
    ### testing new order
    print (new_sample_X.shape)
    
    new_res = compute_pairwise_oneway_ttest_fdr(new_sample_X, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = False)
    
    print (new_res)
    
    ### testing if both are equivalent
    assert (new_res[0] == old_res[0]).all(), "old and new order should be equal, but {} != {}".format(new_res[0], old_res[0])

def test_compute_pairwise_mannwhitney_fdr():

    res = compute_pairwise_mannwhitney_fdr(new_sample_X,new_sample_Y, cor_alpha = cor_alpha, uncor_alpha = uncor_alpha, old_order = False)
    
    print (res)
    
#def test_compute_pairwise_binom_fdr():
    
    #compute_pairwise_binom_fdr()

if __name__ == '__main__':

    #### test signif_code
    #test_return_signif_code() OK
    #test_return_signif_code_Z() # OK
    
    ### test pairwise ttest two-way and one_way
    #test_compute_pairwise_ttest_fdr() ## OK
    #test_compute_pairwise_oneway_ttest_fdr() ## OK
    
    ### test Mann Whitney
    test_compute_pairwise_mannwhitney_fdr()
    
    #test_compute_pairwise_binom_fdr
    
    
    
