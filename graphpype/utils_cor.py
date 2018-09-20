# -*- coding: utf-8 -*-
"""
Support function for correl_mat.py, modularity.py and gather_cormats
"""

import sys
import os


from scipy import stats
import numpy as np

import pandas as pd

import itertools as it

import scipy.signal as filt

from .utils_dtype_coord import *

############################## time series selection

def mean_select_mask_data(data_img, data_mask):

    if np.all(data_img.shape[:3] == data_mask.shape):
        print("Image and mask are compatible")

        masked_data_matrix = data_img[data_mask == 1, :]
        print(masked_data_matrix.shape)

        try:
            print("ok nanmean")
            mean_mask_data_matrix = np.nanmean(masked_data_matrix, axis=0)

        except AttributeError:

            print("no nanmean (version of numpy is too old), using mean only")
            mean_mask_data_matrix = np.mean(masked_data_matrix, axis=0)
    else:
        print("Warning, Image and mask are incompatible {} {}".format(data_img.shape,data_mask.shape))
        return

    return np.array(mean_mask_data_matrix)


def mean_select_indexed_mask_data(orig_ts, indexed_mask_rois_data, min_BOLD_intensity=50, percent_signal=0.5, background_val=-1.0):

    # extrating ts by averaging the time series of all voxel with the same index
    sequence_roi_index = np.unique(indexed_mask_rois_data)

    if sequence_roi_index[0] == background_val:
        sequence_roi_index = sequence_roi_index[1:]

    print ("sequence_roi_index: {}".format(sequence_roi_index))

    mean_masked_ts = []

    keep_rois = np.zeros(shape=(sequence_roi_index.shape[0]), dtype=bool)

    for i, roi_index in enumerate(sequence_roi_index):

        index_roi_x, index_roi_y, index_roi_z = np.where(
            indexed_mask_rois_data == roi_index)

        all_voxel_roi_ts = orig_ts[index_roi_x, index_roi_y, index_roi_z, :]
        nb_voxels, nb_volumes = all_voxel_roi_ts.shape
        
        # testing if at least 50% of the voxels in the ROIs have values always higher than min bold intensity
        nb_signal_voxels = np.sum(np.sum(
            all_voxel_roi_ts > min_BOLD_intensity, axis=1) == nb_volumes)

        if  nb_signal_voxels/float(nb_voxels) > percent_signal:

            keep_rois[i] = True

            try:
                mean_all_voxel_roi_ts = np.nanmean(all_voxel_roi_ts, axis=0)

            except AttributeError:

                print("Warning, no nanmean (version of numpy is too old), using mean only")
                mean_all_voxel_roi_ts = np.mean(all_voxel_roi_ts, axis=0)

            mean_masked_ts.append(mean_all_voxel_roi_ts)
        else:
            print("ROI {} was not selected : {} {} ".format(roi_index, nb_signal_voxels , nb_signal_voxels/float(nb_voxels)))

    assert len(mean_masked_ts) != 0, "min_BOLD_intensity {} and percent_signal {} are to restrictive".format(min_BOLD_intensity, percent_signal)

    mean_masked_ts = np.array(mean_masked_ts, dtype='f')

    print(mean_masked_ts.shape)

    return mean_masked_ts, keep_rois

############################################# covariate regression ##############################
def regress_parameters(data_matrix, covariates):

    import statsmodels.formula.api as smf

    ##### formatting dataframe
    if data_matrix.shape[1] == covariates.shape[0]:
        print ("Transposing data_matrix shape {} -> {}".format(data_matrix.shape, np.transpose(data_matrix).shape))
        data_matrix = np.transpose(data_matrix)
        
    data_names = ['Var_' + str(i) for i in range(data_matrix.shape[1])]

    covar_names = ['Cov_' + str(i) for i in range(covariates.shape[1])]

    df = pd.DataFrame(np.concatenate((data_matrix, covariates), axis=1), columns=data_names + covar_names)

    #### computing regression
    resid_data = [smf.ols(formula = var+" ~ "+" + ".join(covar_names), data=df).fit().resid.values
                  for var in data_names]

    resid_data_matrix = np.array(resid_data, dtype=float)

    print(resid_data_matrix.shape)

    return resid_data_matrix


def filter_data(data_matrix,N=5, Wn=0.04, btype='highpass'):

    b, a = filt.iirfilter(N = N, Wn = Wn, btype = btype)

    return filt.filtfilt(b, a, x=data_matrix, axis=1)


def normalize_data(data_matrix):

    print(data_matrix.shape)

    z_score_data_matrix = stats.zscore(data_matrix, axis=1)

    print(z_score_data_matrix.shape)

    return z_score_data_matrix

#### old fashionned way to compute regression
#def regress_parameters_rpy(data_matrix, covariates):

    #import rpy

    #resid_data_matrix = np.zeros(shape=data_matrix.shape, dtype='float')

    #for i in range(data_matrix.shape[0]):

        #rpy.r.assign('r_serie', data_matrix[i, :])
        #rpy.r.assign('r_rp', covariates)

        #r_formula = "r_serie ~"

        #print(covariates.shape)

        #for cov in range(covariates.shape[1]):

            #r_formula += " r_rp[,"+str(cov+1)+"]"

            #if cov != list(range(covariates.shape[1]))[-1]:

                #r_formula += " +"

        #resid_data_matrix[i, ] = rpy.r.lm(rpy.r(r_formula))['residuals']

        #print(resid_data_matrix[i, ])

    #print(resid_data_matrix)

    #return resid_data_matrix


################################################## Computing correlation matrices
def return_conf_cor_mat(ts_mat, weight_vect, conf_interval_prob = 0.01):

    """
    Compute correlation matrices over a time series and weight vector, either Rsquared-value matrix (cor_mat), or Z (after R-to-Z values) matrix (Z_cor_mat)
    It also return possibly thresholded matrices (conf_cor_mat and Z_conf_cor_mat) according to a confidence interval probability conf_interval_prob
    """
    if ts_mat.shape[1] == len(weight_vect):
        print ("Transposing data_matrix shape {} -> {}".format(ts_mat.shape, np.transpose(ts_mat).shape))
        ts_mat = np.transpose(ts_mat)
     
    assert ts_mat.shape[0] == len(weight_vect), "Error, incompatible regressor length {} {}".format(
            ts_mat.shape[0], len(weight_vect))

    print (weight_vect)
    
    keep = weight_vect > 0.0
    w = weight_vect[keep]
    ts_mat = ts_mat[keep, :]


    # confidence interval for variance computation
    norm = stats.norm.ppf(1-conf_interval_prob/2)
    deg_freedom = w.sum()/w.max()-3

    s, n = ts_mat.shape

    Z_cor_mat = np.zeros((n, n), dtype=float)
    Z_conf_cor_mat = np.zeros((n, n), dtype=float)
    cor_mat = np.zeros((n, n), dtype=float)
    conf_cor_mat = np.zeros((n, n), dtype=float)

    ts_mat2 = ts_mat*np.sqrt(w)[:, np.newaxis]

    for i, j in it.combinations(list(range(n)), 2):

        keep_val = ~(np.isnan(ts_mat2[:, i]) | np.isnan(ts_mat2[:, j]))

        # print keep_val

        s1 = ts_mat2[keep_val, i]
        s2 = ts_mat2[keep_val, j]

        cor_mat[i, j] = (s1*s2).sum()/np.sqrt((s1*s1).sum() * (s2*s2).sum())
        Z_cor_mat[i, j] = np.arctanh(cor_mat[i, j])

        assert not np.isnan(
            Z_cor_mat[i, j]), "Error Z_cor_mat {}{} should not be NAN value".format(i, j)

        assert not np.isinf(
            Z_cor_mat[i, j]), "Error Z_cor_mat {}{} should not be infinite value".format(i, j)

    signif_pos = (Z_cor_mat > norm/np.sqrt(deg_freedom)) & (np.sign(Z_cor_mat) == +1.0)
    signif_neg = (Z_cor_mat < -norm/np.sqrt(deg_freedom)) & (np.sign(Z_cor_mat) == -1.0)

    Z_conf_cor_mat[signif_pos] = Z_cor_mat[signif_pos]
    Z_conf_cor_mat[signif_neg] = Z_cor_mat[signif_neg]

    conf_cor_mat[signif_pos] = cor_mat[signif_pos]
    conf_cor_mat[signif_neg] = cor_mat[signif_neg]

    return cor_mat, Z_cor_mat, conf_cor_mat, Z_conf_cor_mat


################################### computing corres matrix using reference (corres_) and individual labels or coords
def where_in_labels(labels, corres_labels):

    return np.array([corres_labels.index(lab) for lab in labels], dtype='int64')


def return_corres_correl_mat(mat, coords, corres_coords ):

    assert mat.shape[0] == mat.shape[1], "Error, matrix should be square {}".format(mat.shape)
    assert mat.shape[0] == coords.shape[0], "Error, matrix {} and coords {} should have the same length".format(
        mat.shape[0],coords.shape[1])
    
    where_in_corres = where_in_coords(coords, corres_coords)

    print (where_in_corres)
    corres_size = corres_coords.shape[0]
    
    print(np.min(where_in_corres), np.max(where_in_corres), where_in_corres.shape)

    corres_mat = np.zeros((corres_size,corres_size), dtype=float)
    possible_edge_mat = np.zeros((corres_size,corres_size), dtype=int)

    print (corres_mat.shape)
    
    for i, j in it.combinations(range(coords.shape[0]), 2):

        corres_mat[where_in_corres[i], where_in_corres[j]] = mat[i, j]
        corres_mat[where_in_corres[j], where_in_corres[i]] = mat[i, j]

        possible_edge_mat[where_in_corres[i], where_in_corres[j]] = 1
        possible_edge_mat[where_in_corres[j], where_in_corres[i]] = 1

    return corres_mat, possible_edge_mat


def return_corres_correl_mat_labels(mat, labels, corres_labels):

    assert isinstance(labels,list), "Error, labels should be a list"
    assert isinstance(corres_labels,list), "corres_labels, labels should be a list"
    where_in_corres = where_in_labels(labels, corres_labels)

    corres_size = len(corres_labels)
    
    print(np.min(where_in_corres), np.max(where_in_corres), where_in_corres.shape)

    corres_mat = np.zeros((corres_size,corres_size), dtype=float)
    possible_edge_mat = np.zeros((corres_size, corres_size), dtype=int)

    for i, j in it.product(range(len(labels)), repeat=2):
        corres_mat[where_in_corres[i], where_in_corres[j]] = mat[i, j]
        possible_edge_mat[where_in_corres[i], where_in_corres[j]] = 1

    print(corres_mat.shape)

    return corres_mat, possible_edge_mat

############################################################# 
#def return_coclass_mat(community_vect, corres_coords, gm_mask_coords):

    #print(corres_coords.shape[0], community_vect.shape[0])

    #if (corres_coords.shape[0] != community_vect.shape[0]):
        #print("warning, length of corres_coords and community_vect are imcompatible {} {}".format(
            #corres_coords.shape[0], community_vect.shape[0]))

    #where_in_gm = where_in_coords(corres_coords, gm_mask_coords)

    #print(where_in_gm)

    #print(np.min(where_in_gm))
    #print(np.max(where_in_gm))
    #print(where_in_gm.shape)

    #if (where_in_gm.shape[0] != community_vect.shape[0]):
        #print("warning, length of where_in_gm and community_vect are imcompatible {} {}".format(
            #where_in_gm.shape[0], community_vect.shape[0]))

    #coclass_mat = np.zeros(
        #(gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

    #possible_edge_mat = np.zeros(
        #(gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

    #for i, j in it.combinations(list(range(where_in_gm.shape[0])), 2):

        #coclass_mat[where_in_gm[i], where_in_gm[j]] = np.int(
            #community_vect[i] == community_vect[j])
        #coclass_mat[where_in_gm[j], where_in_gm[i]] = np.int(
            #community_vect[i] == community_vect[j])

        #possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1
        #possible_edge_mat[where_in_gm[j], where_in_gm[i]] = 1

    #return coclass_mat, possible_edge_mat



#def return_coclass_mat_labels(community_vect, corres_labels, gm_mask_labels):

    #print(corres_labels.shape[0], community_vect.shape[0])

    #if (corres_labels.shape[0] != community_vect.shape[0]):
        #print("warning, length of corres_labels and community_vect are imcompatible {} {}".format(
            #corres_labels.shape[0], community_vect.shape[0]))

    #where_in_gm = where_in_labels(
        #corres_labels.tolist(), gm_mask_labels.tolist())

    #print(where_in_gm)

    #print(np.min(where_in_gm))
    #print(np.max(where_in_gm))
    #print(where_in_gm.shape)

    #if (where_in_gm.shape[0] != community_vect.shape[0]):
        #print("warning, length of where_in_gm and community_vect are imcompatible {} {}".format(
            #where_in_gm.shape[0], community_vect.shape[0]))

    #coclass_mat = np.zeros(
        #(gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

    #possible_edge_mat = np.zeros(
        #(gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

    #for i, j in it.combinations(list(range(where_in_gm.shape[0])), 2):

        #coclass_mat[where_in_gm[i], where_in_gm[j]] = np.int(
            #community_vect[i] == community_vect[j])
        #coclass_mat[where_in_gm[j], where_in_gm[i]] = np.int(
            #community_vect[i] == community_vect[j])

        #possible_edge_mat[where_in_gm[i], where_in_gm[j]] = 1
        #possible_edge_mat[where_in_gm[j], where_in_gm[i]] = 1

    #return coclass_mat, possible_edge_mat
