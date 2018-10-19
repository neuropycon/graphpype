import numpy as np
import scipy.stats as stat

from graphpype.utils_stats import (_return_signif_code, _return_signif_code_Z,
                                   compute_pairwise_ttest_fdr,
                                   compute_pairwise_oneway_ttest_fdr,
                                   compute_pairwise_binom_fdr,
                                   compute_pairwise_mannwhitney_fdr,
                                   compute_correl_behav)


# building objects for testing
# building p-values vector
N = 100  # size of vector
cor_alpha = 0.05
uncor_alpha = 0.05
epsilon = 0.0001

# bonferroni p-values
N_bon = 3
p_bon = np.array([cor_alpha/N - epsilon]*N_bon, dtype=float).reshape(-1, 1)

# FDR values (significance should be cut in the range)
N_fdr = 10
p_fdr = np.array([cor_alpha/N]*N_fdr, dtype=float)

# modulating between cor_alpha and uncor_alpha
# significance should vary between fdr, FP, and uncor
p_fdr = p_fdr + np.array([uncor_alpha - cor_alpha/N]
                         * N_fdr, dtype=float)*np.arange(N_fdr)/N_fdr

p_fdr = p_fdr.reshape(-1, 1)

# uncorrected values
N_uncor = 50
p_uncor = (np.array([uncor_alpha - epsilon] *
                    N_uncor, dtype=float)).reshape(-1, 1)

N_unsignif = N - (N_bon + N_fdr + N_uncor)
p_unsignif = (np.array([0.5]*N_unsignif, dtype=float)).reshape(-1, 1)

# p-values
p_val = np.concatenate((p_bon, p_fdr, p_uncor, p_unsignif), axis=0).reshape(-1)

# equivalent Z_val
Z_val = stat.norm.ppf(1-p_val/2)


def test_return_signif_code():
    """test if signif codes are correct given a vector of p-values"""
    print(p_val)

    res = _return_signif_code(
        p_val, uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha,
        bon_alpha=cor_alpha)

    # Counting by hand result
    # [1] 10+9+5+10+9+5+10 = 58
    # [0] 9  + 10+9+5 +4 = 37
    assert all(res == np.array([4]*3+[3]+[2]+[1]*58+[0]*37)), ("Error in the \
        significance code")


def test_return_signif_code_Z():
    """test if signif codes are correct given a vector of Zscores"""
    res = _return_signif_code_Z(
        Z_val, uncor_alpha=uncor_alpha, fdr_alpha=cor_alpha,
        bon_alpha=cor_alpha)

    assert all(res == np.array([4]*3+[3]+[2]+[1]*58+[0]*37)), ("Error in the \
        significance code vector")


# compute two samples of matrices to test
sample_size = 20
array_size = 10

# new order (sample_size = first dimension)
new_sample_X = np.random.rand(sample_size, array_size, array_size) * \
    np.random.choice([-1, 1], size=(sample_size, array_size, array_size))
new_sample_Y = np.random.rand(sample_size, array_size, array_size) * \
    np.random.choice([-1, 1], size=(sample_size, array_size, array_size))

# old order (sample_size = last dimension)
old_sample_X = np.rollaxis(new_sample_X, 0, 3)
old_sample_Y = np.rollaxis(new_sample_Y, 0, 3)

# compute random regressor
random_reg = np.random.rand(sample_size)


def test_compute_pairwise_ttest_fdr():
    """
    test if pairwise t-test give the same results if old or new order are given
    """
    # testing old order
    print(old_sample_X.shape)
    old_res = compute_pairwise_ttest_fdr(
        old_sample_X, old_sample_Y, cor_alpha=cor_alpha,
        uncor_alpha=uncor_alpha, old_order=True)

    # testing new order
    new_res = compute_pairwise_ttest_fdr(
        new_sample_X, new_sample_Y, cor_alpha=cor_alpha,
        uncor_alpha=uncor_alpha, old_order=False)

    # testing if both are equivalent
    assert (new_res[0] == old_res[0]).all()


def test_compute_pairwise_oneway_ttest_fdr():
    """test if oneway pairwise t-test"""
    # testing old order
    print(old_sample_X.shape)
    old_res = compute_pairwise_oneway_ttest_fdr(
        old_sample_X, cor_alpha=cor_alpha, uncor_alpha=uncor_alpha,
        old_order=True)

    print(old_res)

    # testing new order
    print(new_sample_X.shape)

    new_res = compute_pairwise_oneway_ttest_fdr(
        new_sample_X, cor_alpha=cor_alpha, uncor_alpha=uncor_alpha,
        old_order=False)

    print(new_res)

    # testing if both are equivalent
    assert (new_res[0] == old_res[0]).all()


def test_compute_pairwise_mannwhitney_fdr():
    """
    test if pairwise Mann-Whitney test is correct
    """
    # TODO relevant tests should be added
    res = compute_pairwise_mannwhitney_fdr(
        new_sample_X, new_sample_Y, cor_alpha=cor_alpha,
        uncor_alpha=uncor_alpha, old_order=False)

    assert res.shape == new_sample_X.shape[1:]


def test_compute_correl_behav():
    """
    test if pairwise correlation with a behavioural vector is correct
    """
    res = compute_correl_behav(X=new_sample_X, reg_interest=random_reg)

    # TODO relevant tests should be added
    assert res[0].shape == new_sample_X.shape[1:]
    assert res[1].shape == new_sample_X.shape[1:]
    assert res[2].shape == new_sample_X.shape[1:]


# test binomial
# Generating random binomial distribution
new_binom_X = np.random.choice(
    [0, 1], size=(sample_size, array_size, array_size))
new_binom_Y = np.random.choice(
    [0, 1], size=(sample_size, array_size, array_size))


def test_compute_pairwise_binom_fdr():
    """test if pairwise binomial test is correct"""
    # TODO relevant tests should be added
    res = compute_pairwise_binom_fdr(
        new_binom_X, new_binom_Y, uncor_alpha=uncor_alpha,
        cor_alpha=cor_alpha, old_order=False)

    assert res.shape == new_binom_X.shape[1:]
