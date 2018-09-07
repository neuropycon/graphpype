# -*- coding: utf-8 -*-
"""
Support function for run_wei_sig_correl_mat.py, run_wei_sig_modularity.py and run_gather_partitions
"""

import sys
import os


from scipy import stats
import numpy as np

import pandas as pd

#import pandas as pd

import itertools as it

#import scipy.io
#import scipy.spatial.distance as dist
##import math

import scipy.signal as filt

#import scipy.sparse as sp
#import scipy.cluster.hierarchy as hie


import time

#from collections import Counter

#from nipype.utils.filemanip import split_filename as split_f

#from dipy.align.aniso2iso import resample

#from sets import Set

from .utils_dtype_coord import *

def mean_select_mask_data(data_img, data_mask):

    img_shape = data_img.shape
    mask_shape = data_mask.shape

    if img_shape[:3] == mask_shape:
        print("Image and mask are compatible")

        masked_data_matrix = data_img[data_mask == 1, :]
        print(masked_data_matrix.shape)

        try:
            print("ok nanmean")
            mean_mask_data_matrix = np.nanmean(masked_data_matrix, axis=0)

        except AttributeError:

            print("no nanmean")
            mean_mask_data_matrix = np.mean(masked_data_matrix, axis=0)

        print(mean_mask_data_matrix.shape)

    else:
        print("Warning, Image and mask are incompatible")
        print(img_shape)
        print(mask_shape)
        return

    return np.array(mean_mask_data_matrix)


def mean_select_indexed_mask_data(orig_ts, indexed_mask_rois_data, min_BOLD_intensity=50, percent_signal=0.5, background_val=-1.0):

        # extrating ts by averaging the time series of all voxel with the same index
    sequence_roi_index = np.unique(indexed_mask_rois_data)

    print(sequence_roi_index)

    if sequence_roi_index[0] == background_val:
        sequence_roi_index = sequence_roi_index[1:]

    # print "sequence_roi_index:"
    print(sequence_roi_index)

    mean_masked_ts = []

    keep_rois = np.zeros(shape=(sequence_roi_index.shape[0]), dtype=bool)

    for i, roi_index in enumerate(sequence_roi_index):

        index_roi_x, index_roi_y, index_roi_z = np.where(
            indexed_mask_rois_data == roi_index)

        all_voxel_roi_ts = orig_ts[index_roi_x, index_roi_y, index_roi_z, :]

        print(all_voxel_roi_ts.shape)
        
        # testing if at least 50% of the voxels in the ROIs have values always higher than min bold intensity
        nb_signal_voxels = np.sum(np.sum(
            all_voxel_roi_ts > min_BOLD_intensity, axis=1) == all_voxel_roi_ts.shape[1])

        non_nan_voxels = np.sum(np.sum(np.logical_not(
            np.isnan(all_voxel_roi_ts)), axis=1) == all_voxel_roi_ts.shape[1])

        print(nb_signal_voxels, non_nan_voxels, all_voxel_roi_ts.shape[0])

        if nb_signal_voxels/float(all_voxel_roi_ts.shape[0]) > percent_signal:

            keep_rois[i] = True

            try:
                mean_all_voxel_roi_ts = np.nanmean(all_voxel_roi_ts, axis=0)

            except AttributeError:

                mean_all_voxel_roi_ts = np.mean(all_voxel_roi_ts, axis=0)

            mean_masked_ts.append(mean_all_voxel_roi_ts)
        else:
            print("ROI {} was not selected : {} {} ".format(roi_index, np.sum(np.sum(all_voxel_roi_ts > min_BOLD_intensity, axis=1) == all_voxel_roi_ts.shape[1]), np.sum(
                np.sum(all_voxel_roi_ts > min_BOLD_intensity, axis=1) == all_voxel_roi_ts.shape[1])/float(all_voxel_roi_ts.shape[0])))

    assert len(mean_masked_ts) != 0, "min_BOLD_intensity {} and percent_signal are to restrictive".format(
        min_BOLD_intensity, percent_signal)

    mean_masked_ts = np.array(mean_masked_ts, dtype='f')

    print(mean_masked_ts.shape)

    return mean_masked_ts, keep_rois

############################################# covariate regression ##############################

def regress_parameters(data_matrix, covariates):

    import statsmodels.formula.api as smf

    print(data_matrix.shape)

    print(covariates.shape)

    resid_data = []

    data_names = ['Var_' + str(i) for i in range(data_matrix.shape[0])]

    print(data_names)

    covar_names = ['Cov_' + str(i) for i in range(covariates.shape[1])]

    print(np.transpose(data_matrix).shape)

    all_data = np.concatenate((np.transpose(data_matrix), covariates), axis=1)

    print(all_data.shape)

    col_names = data_names + covar_names

    df = pd.DataFrame(all_data, columns=col_names)

    print(df)

    for var in data_names:

        formula = var + " ~ " + " + ".join(covar_names)

        print(formula)

        est = smf.ols(formula=formula, data=df).fit()

        resid = est.resid.values
        resid_data.append(est.resid.values)

    resid_data_matrix = np.array(resid_data, dtype=float)

    print(resid_data_matrix.shape)

    return resid_data_matrix


def filter_data(data_matrix):

    print(data_matrix.shape)

    filt_data = []

    b, a = filt.iirfilter(N=5, Wn=0.04, btype='highpass')

    filt_data_matrix = filt.filtfilt(b, a, x=data_matrix, axis=1)

    print(filt_data_matrix.shape)

    return filt_data_matrix


def normalize_data(data_matrix):

    print(data_matrix.shape)

    z_score_data_matrix = stats.zscore(data_matrix, axis=1)

    print(z_score_data_matrix.shape)

    return z_score_data_matrix

# def regress_filter_normalize_parameters(data_matrix,covariates):

    #import statsmodels.formula.api as smf

    # print(data_matrix.shape)

    # print(covariates.shape)

    #resid_data = []
    #resid_filt_data = []
    #z_score_data = []

    #b,a = filt.iirfilter(N = 5, Wn = 0.04, btype = 'highpass')

    #data_names = ['Var_' + str(i) for i in range(data_matrix.shape[0])]

    # print(data_names)

    #covar_names = ['Cov_' + str(i) for i in range(covariates.shape[1])]

    # print(np.transpose(data_matrix).shape)

    #all_data = np.concatenate((np.transpose(data_matrix),covariates), axis = 1)

    # print(all_data.shape)

    #col_names = data_names + covar_names

    #df = pd.DataFrame(all_data, columns = col_names)

    # print(df)

    # for var in data_names:

    #formula = var + " ~ " + " + ".join(covar_names)

    # print(formula)

    #est = smf.ols(formula=formula, data=df).fit()

    # print est.summary()

    # print est.resid.values

    #resid = est.resid.values
    # resid_data.append(est.resid.values)

    #resid_filt = filt.filtfilt(b,a,x = resid)
    # resid_filt_data.append(resid_filt)

    #z_score  = stats.zscore(resid_filt)
    # z_score_data.append(z_score)

    #resid_data_matrix = np.array(resid_data,dtype = float)

    # print(resid_data_matrix.shape)

    #resid_filt_data_matrix = np.array(resid_filt_data, dtype = float)

    # print resid_filt_data_matrix

    #z_score_data_matrix = np.array(z_score_data, dtype = float)

    # print z_score_data_matrix

    # return resid_data_matrix,resid_filt_data_matrix,z_score_data_matrix


def regress_parameters_rpy(data_matrix, covariates):

    import rpy

    resid_data_matrix = np.zeros(shape=data_matrix.shape, dtype='float')

    for i in range(data_matrix.shape[0]):

        rpy.r.assign('r_serie', data_matrix[i, :])
        rpy.r.assign('r_rp', covariates)

        r_formula = "r_serie ~"

        print(covariates.shape)

        for cov in range(covariates.shape[1]):

            r_formula += " r_rp[,"+str(cov+1)+"]"

            if cov != list(range(covariates.shape[1]))[-1]:

                r_formula += " +"

        resid_data_matrix[i, ] = rpy.r.lm(rpy.r(r_formula))['residuals']

        print(resid_data_matrix[i, ])

    print(resid_data_matrix)

    return resid_data_matrix


def regress_filter_normalize_parameters_rpy(data_matrix, covariates):

    import rpy

    resid_data_matrix = np.zeros(shape=data_matrix.shape)
    resid_filt_data_matrix = np.zeros(shape=data_matrix.shape)
    z_score_data_matrix = np.zeros(shape=data_matrix.shape)

    b, a = filt.iirfilter(N=5, Wn=0.04, btype='highpass')

    # for i in [0,1]:
    for i in range(data_matrix.shape[0]):

        # print data_matrix[i,:].shape

        # print covariates.shape

        # print i
        rpy.r.assign('r_serie', data_matrix[i, :])
        rpy.r.assign('r_rp', covariates)

        r_formula = "r_serie ~"

        print(covariates.shape)

        for cov in range(covariates.shape[1]):

            r_formula += " r_rp[,"+str(cov+1)+"]"

            if cov != list(range(covariates.shape[1]))[-1]:

                r_formula += " +"

        # print r_formula

        resid_data_matrix[i, ] = rpy.r.lm(rpy.r(r_formula))['residuals']

        resid_filt_data_matrix[i, ] = filt.lfilter(
            b, a, x=resid_data_matrix[i, ])

        z_score_data_matrix[i, ] = stats.zscore(resid_filt_data_matrix[i, ])

    return resid_data_matrix, resid_filt_data_matrix, z_score_data_matrix

def return_conf_cor_mat(ts_mat, regressor_vect, conf_interval_prob):

    t1 = time.time()

    if ts_mat.shape[0] != len(regressor_vect):
        print("Warning, incompatible regressor length {} {}".format(
            ts_mat.shape[0], len(regressor_vect)))
        return

    keep = regressor_vect > 0.0
    w = regressor_vect[keep]
    ts_mat = ts_mat[keep, :]

    # confidence interval for variance computation
    norm = stats.norm.ppf(1-conf_interval_prob/2)
    #deg_freedom = w.sum()-3
    deg_freedom = w.sum()/w.max()-3
    #deg_freedom = w.shape[0]-3

    print(deg_freedom)

    print(norm, norm/np.sqrt(deg_freedom))

    print(regressor_vect.shape[0], w.shape[0], w.sum(), w.sum()/w.max())

    s, n = ts_mat.shape

    Z_cor_mat = np.zeros((n, n), dtype=float)
    Z_conf_cor_mat = np.zeros((n, n), dtype=float)

    cor_mat = np.zeros((n, n), dtype=float)
    conf_cor_mat = np.zeros((n, n), dtype=float)

    ts_mat2 = ts_mat*np.sqrt(w)[:, np.newaxis]

    for i, j in it.combinations(list(range(n)), 2):

        keep_val = np.logical_not(np.logical_or(
            np.isnan(ts_mat2[:, i]), np.isnan(ts_mat2[:, j])))

        # print keep_val

        s1 = ts_mat2[keep_val, i]
        s2 = ts_mat2[keep_val, j]

        cor_mat[i, j] = (s1*s2).sum()/np.sqrt((s1*s1).sum() * (s2*s2).sum())
        Z_cor_mat[i, j] = np.arctanh(cor_mat[i, j])

        assert not np.isnan(Z_cor_mat[i, j]), "Error Z_cor_mat {}{} should not be NAN value".format(i, j)

        assert not np.isinf(Z_cor_mat[i, j]), "Error Z_cor_mat {}{} should not be infinite value".format(i, j)

        if cor_mat[i, j] > 0:
            conf_cor_mat[i, j] = cor_mat[i, j] - \
                np.tanh(Z_cor_mat[i, j] - norm/np.sqrt(deg_freedom))
        else:
            conf_cor_mat[i, j] = - cor_mat[i, j] + \
                np.tanh(Z_cor_mat[i, j] + norm/np.sqrt(deg_freedom))

    Z_conf_cor_mat = np.zeros((n, n), dtype=float)

    signif_pos = np.logical_and(
        Z_cor_mat > norm/np.sqrt(deg_freedom), np.sign(Z_cor_mat) == +1.0)

    print(np.sum(signif_pos))
    # print signif_pos

    signif_neg = np.logical_and(
        Z_cor_mat < -norm/np.sqrt(deg_freedom), np.sign(Z_cor_mat) == -1.0)

    print(np.sum(signif_neg))
    # print signif_neg

    Z_conf_cor_mat[signif_pos] = Z_cor_mat[signif_pos]
    Z_conf_cor_mat[signif_neg] = Z_cor_mat[signif_neg]

    print(np.sum(Z_conf_cor_mat != 0.0))

    t2 = time.time()

    print("Weighted correlation computation took " + str(t2-t1) + "s")

    return cor_mat, Z_cor_mat, conf_cor_mat, Z_conf_cor_mat


def return_coclass_mat(community_vect, corres_coords, gm_mask_coords):

    print(corres_coords.shape[0], community_vect.shape[0])

    if (corres_coords.shape[0] != community_vect.shape[0]):
        print("warning, length of corres_coords and community_vect are imcompatible {} {}".format(
            corres_coords.shape[0], community_vect.shape[0]))

    where_in_gm = where_in_coords(corres_coords, gm_mask_coords)

    print(where_in_gm)

    print(np.min(where_in_gm))
    print(np.max(where_in_gm))
    print(where_in_gm.shape)

    if (where_in_gm.shape[0] != community_vect.shape[0]):
        print("warning, length of where_in_gm and community_vect are imcompatible {} {}".format(
            where_in_gm.shape[0], community_vect.shape[0]))

    coclass_mat = np.zeros(
        (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

    possible_edge_mat = np.zeros(
        (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

    for i, j in it.combinations(list(range(where_in_gm.shape[0])), 2):

        coclass_mat[where_in_gm[i], where_in_gm[j]] = np.int(
            community_vect[i] == community_vect[j])
        coclass_mat[where_in_gm[j], where_in_gm[i]] = np.int(
            community_vect[i] == community_vect[j])

        possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1
        possible_edge_mat[where_in_gm[j], where_in_gm[i]] = 1

    return coclass_mat, possible_edge_mat


def where_in_labels(corres_labels, gm_mask_labels):

    return np.array([gm_mask_labels.index(lab) for lab in corres_labels], dtype='int64')


def return_coclass_mat_labels(community_vect, corres_labels, gm_mask_labels):

    print(corres_labels.shape[0], community_vect.shape[0])

    if (corres_labels.shape[0] != community_vect.shape[0]):
        print("warning, length of corres_labels and community_vect are imcompatible {} {}".format(
            corres_labels.shape[0], community_vect.shape[0]))

    where_in_gm = where_in_labels(
        corres_labels.tolist(), gm_mask_labels.tolist())

    print(where_in_gm)

    print(np.min(where_in_gm))
    print(np.max(where_in_gm))
    print(where_in_gm.shape)

    if (where_in_gm.shape[0] != community_vect.shape[0]):
        print("warning, length of where_in_gm and community_vect are imcompatible {} {}".format(
            where_in_gm.shape[0], community_vect.shape[0]))

    coclass_mat = np.zeros(
        (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

    possible_edge_mat = np.zeros(
        (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

    for i, j in it.combinations(list(range(where_in_gm.shape[0])), 2):

        coclass_mat[where_in_gm[i], where_in_gm[j]] = np.int(
            community_vect[i] == community_vect[j])
        coclass_mat[where_in_gm[j], where_in_gm[i]] = np.int(
            community_vect[i] == community_vect[j])

        possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1
        possible_edge_mat[where_in_gm[j], where_in_gm[i]] = 1

    return coclass_mat, possible_edge_mat



def return_corres_correl_mat(Z_cor_mat, coords, gm_mask_coords):

    # print coords
    # print gm_mask_coords

    where_in_gm = where_in_coords(coords, gm_mask_coords)

    # print where_in_gm

    print(np.min(where_in_gm), np.max(where_in_gm), where_in_gm.shape)

    corres_correl_mat = np.zeros(
        (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=float)
    possible_edge_mat = np.zeros(
        (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

    for i, j in it.combinations(list(range(len(where_in_gm))), 2):

        corres_correl_mat[where_in_gm[i], where_in_gm[j]] = Z_cor_mat[i, j]
        corres_correl_mat[where_in_gm[j], where_in_gm[i]] = Z_cor_mat[i, j]

        possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1
        possible_edge_mat[where_in_gm[j], where_in_gm[i]] = 1

    return corres_correl_mat, possible_edge_mat


def return_corres_correl_mat_labels(Z_cor_mat, corres_labels, gm_mask_labels):

    where_in_gm = where_in_labels(
        corres_labels.tolist(), gm_mask_labels.tolist())

    print(Z_cor_mat.shape)

    # print where_in_gm

    # print np.min(where_in_gm)
    # print np.max(where_in_gm)
    # print where_in_gm.shape

    print(np.min(where_in_gm), np.max(where_in_gm), where_in_gm.shape)

    corres_correl_mat = np.zeros(
        (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=float)
    possible_edge_mat = np.zeros(
        (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

    for i, j in it.product(list(range(len(where_in_gm))), repeat=2):

        # print i,j

        # print where_in_gm[i],where_in_gm[j]
        # print corres_correl_mat[where_in_gm[i],where_in_gm[j]]
        # print Z_cor_mat[i,j]

        corres_correl_mat[where_in_gm[i], where_in_gm[j]] = Z_cor_mat[i, j]

        # print corres_correl_mat[where_in_gm[i],where_in_gm[j]]

        possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1

    print(corres_correl_mat.shape)

    return corres_correl_mat, possible_edge_mat
