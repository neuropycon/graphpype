# -*- coding: utf-8 -*-

"""
Compute ROI labeled mask from spm contrast image or images
"""


import sys
import os
# sys.path.append('../irm_analysis')

#from  define_variables import *


from graphpype.labeled_mask import compute_recombined_HO_template

from graphpype.utils_dtype_coord import *

import glob

from xml.dom import minidom
import os

import numpy as np

from nibabel import load, save

import nipy.labs.spatial_models.mroi as mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image
import nipy.labs.spatial_models.hroi as hroi

import nipy.labs.statistical_mapping as stat_map

import itertools as iter

import scipy.spatial.distance as dist

########################################### Activation peaks ROI template (computed once before the pipeline) ################################################

# scan toutes les possibilités dans le cube, et ne retourne que les ROIs dont le nombre de voxels dans le voisinage appartienant à AAL et au mask est supérieur à min_nb_voxels_in_neigh


def return_indexed_mask_neigh_within_binary_template(peak_position, neighbourhood, resliced_template_template_data, orig_peak_coords_dt, min_nb_voxels_in_neigh):

    peak_x, peak_y, peak_z = np.array(peak_position, dtype='int')

    neigh_range = list(range(-neighbourhood, neighbourhood+1))

    list_neigh_coords = []

    peak_template_roi_index = resliced_template_template_data[peak_x, peak_y, peak_z]

    print(peak_template_roi_index)

    # print "template index = " + str(peak_template_roi_index)

    count_neigh_in_orig_mask = 0

    if peak_template_roi_index != 0:

        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x, neigh_y, neigh_z = peak_position + relative_coord

            neigh_coord_dt = convert_np_coords_to_coords_dt(
                np.array([[neigh_x, neigh_y, neigh_z]]))
            #neigh_coord_dt = np.array([(neigh_x,neigh_y,neigh_z), ], dtype = coord_dt)

            neigh_template_roi_index = resliced_template_template_data[neigh_x,
                                                                       neigh_y, neigh_z]

            # print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape

            # if neigh_template_roi_index == peak_template_roi_index and np.in1d(neigh_coord_dt,orig_peak_coords_dt):
            if neigh_template_roi_index != 0 and neigh_coord_dt in orig_peak_coords_dt:

                list_neigh_coords.append(
                    np.array([neigh_x, neigh_y, neigh_z], dtype='int16'))

                count_neigh_in_orig_mask = count_neigh_in_orig_mask + 1

        print(list_neigh_coords)

        if min_nb_voxels_in_neigh <= len(list_neigh_coords):

            return list_neigh_coords, peak_template_roi_index

    return [], 0


def return_indexed_mask_cube_size_within_binary_template(peak_position, cube_size, resliced_template_template_data, orig_peak_coords_dt, min_nb_voxels_in_neigh):

    peak_x, peak_y, peak_z = np.array(peak_position, dtype='int')

    list_neigh_coords = []

    peak_template_roi_index = resliced_template_template_data[peak_x, peak_y, peak_z]

    print(peak_template_roi_index)

    # print "template index = " + str(peak_template_roi_index)

    count_neigh_in_orig_mask = 0

    if peak_template_roi_index != 0:

        for relative_coord in iter.product(list(range(cube_size)), repeat=3):

            neigh_x, neigh_y, neigh_z = peak_position + relative_coord

            neigh_coord_dt = convert_np_coords_to_coords_dt(
                np.array([[neigh_x, neigh_y, neigh_z]]))
            #neigh_coord_dt = np.array([(neigh_x,neigh_y,neigh_z), ], dtype = coord_dt)

            neigh_template_roi_index = resliced_template_template_data[neigh_x,
                                                                       neigh_y, neigh_z]

            # print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape

            # if neigh_template_roi_index == peak_template_roi_index and np.in1d(neigh_coord_dt,orig_peak_coords_dt):
            if neigh_template_roi_index != 0 and neigh_coord_dt in orig_peak_coords_dt:

                list_neigh_coords.append(
                    np.array([neigh_x, neigh_y, neigh_z], dtype='int16'))

                count_neigh_in_orig_mask = count_neigh_in_orig_mask + 1

        print(list_neigh_coords)

        0/0

        if min_nb_voxels_in_neigh <= len(list_neigh_coords):

            return list_neigh_coords, peak_template_roi_index

    return [], 0


def return_neigh_within_same_region(peak_position, neighbourhood, resliced_template_template_data, min_nb_voxels_in_neigh):

    peak_x, peak_y, peak_z = np.array(peak_position, dtype='int')

    neigh_range = list(range(-neighbourhood, neighbourhood+1))

    list_neigh_coords = []

    peak_template_roi_index = int(
        resliced_template_template_data[peak_x, peak_y, peak_z])

    # print peak_template_roi_index

    # print "template index = " + str(peak_template_roi_index)

    count_neigh_in_orig_mask = 0

    if peak_template_roi_index != 0:

        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x, neigh_y, neigh_z = peak_position + relative_coord

            neigh_coord_dt = convert_np_coords_to_coords_dt(
                np.array([[neigh_x, neigh_y, neigh_z]]))
            #neigh_coord_dt = np.array([(neigh_x,neigh_y,neigh_z), ], dtype = coord_dt)

            neigh_template_roi_index = resliced_template_template_data[neigh_x,
                                                                       neigh_y, neigh_z]

            # print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape

            if neigh_template_roi_index == peak_template_roi_index:

                list_neigh_coords.append(
                    np.array([neigh_x, neigh_y, neigh_z], dtype='int16'))

                count_neigh_in_orig_mask = count_neigh_in_orig_mask + 1

        # print list_neigh_coords

        if min_nb_voxels_in_neigh <= len(list_neigh_coords):

            return list_neigh_coords, peak_template_roi_index

    return [], 0


def return_voxels_within_same_region(peak_position, ROI_cube_size, template_data, min_nb_voxels_in_neigh):

    template_roi_index = int(
        template_data[peak_position[0], peak_position[1], peak_position[2]])

    if template_roi_index != 0:

        list_voxel_coords = []

        for relative_coord in iter.product(list(range(ROI_cube_size)), repeat=3):

            neigh_x, neigh_y, neigh_z = peak_position + relative_coord

            if np.all(peak_position + relative_coord < np.array(template_data.shape)):

                if template_data[neigh_x, neigh_y, neigh_z] == template_roi_index:

                    list_voxel_coords.append(
                        np.array([neigh_x, neigh_y, neigh_z], dtype='int16'))

        #list_voxel_coords = [[peak_position[0] + relative_coord[0],peak_position[1] + relative_coord[1],peak_position[2] + relative_coord[2]] for relative_coord in iter.product(range(ROI_cube_size), repeat=3) if np.all(peak_position + relative_coord < np.array(template_data.shape)) and  template_data[peak_position[0] + relative_coord[0],peak_position[1] + relative_coord[1],peak_position[2] + relative_coord[2]] == template_roi_index]

        if min_nb_voxels_in_neigh <= len(list_voxel_coords):

            return list_voxel_coords, template_roi_index

    return [], 0

#########################################################################################################


def remove_close_peaks(list_orig_peak_coords, min_dist=2.0 * np.sqrt(3)):

    list_selected_peaks_coords = []

    for orig_peak_coord in list_orig_peak_coords:

        orig_peak_coord_np = np.array(orig_peak_coord)

        if len(list_selected_peaks_coords) > 0:

            selected_peaks_coords_np = np.array(list_selected_peaks_coords)

            #orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)

            #selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)

            # print selected_peaks_coords_np.shape

            # print orig_peak_coord_np.shape

            dist_to_selected_peaks = dist.cdist(
                selected_peaks_coords_np, orig_peak_coord_np.reshape(1, 3), 'euclidean')

            # print dist_to_selected_peaks

            min_dist_to_selected_peaks = np.amin(
                dist_to_selected_peaks, axis=0)

            if min_dist < min_dist_to_selected_peaks:

                list_selected_peaks_coords.append(orig_peak_coord_np)

        else:
            list_selected_peaks_coords.append(orig_peak_coord)

        print(len(list_selected_peaks_coords))

    return list_selected_peaks_coords


def remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords, template_data, min_dist):

    # if len(list_orig_peak_coords) != len(list_orig_peak_MNI_coords):
        # print "!!!!!!!!!!!!!!!! Breaking !!!!!!!!!!!!!!!! list_orig_peak_coords %d and list_orig_peak_MNI_coords %d should have similar length" %(len(list_orig_peak_coords),len(list_orig_peak_MNI_coords))
        # return

    img_shape = template_data.shape

    indexed_mask_rois_data = np.zeros(img_shape, dtype='int64') - 1

    print(indexed_mask_rois_data.shape)

    list_selected_peaks_coords = []

    orig_peak_coords_np = np.array(list_orig_peak_coords)

    print(type(orig_peak_coords_np),
          orig_peak_coords_np.dtype, orig_peak_coords_np.shape)

    list_selected_peaks_indexes = []

    orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)

    print(type(orig_peak_coords_dt),
          orig_peak_coords_dt.dtype, orig_peak_coords_dt.shape)

    # for i,orig_peak_coord in enumerate([list_orig_peak_coords[0]]):
    for i, orig_peak_coord in enumerate(list_orig_peak_coords):

        orig_peak_coord_np = np.array(orig_peak_coord)

        if len(list_selected_peaks_coords) > 0:

            selected_peaks_coords_np = np.array(list_selected_peaks_coords)

            dist_to_selected_peaks = dist.cdist(
                selected_peaks_coords_np, orig_peak_coord_np.reshape(1, 3), 'euclidean')

            min_dist_to_selected_peaks = np.amin(
                dist_to_selected_peaks, axis=0)

            if min_dist < min_dist_to_selected_peaks:

                list_neigh_coords, peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(
                    orig_peak_coord_np, ROI_cube_size, template_data, orig_peak_coords_dt)
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)

                if peak_template_roi_index > 0:

                    neigh_coords = np.array(list_neigh_coords, dtype='int16')

                    indexed_mask_rois_data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2]] = len(
                        list_selected_peaks_coords)

                    list_selected_peaks_coords.append(orig_peak_coord_np)

                    list_selected_peaks_indexes.append(i)

                    print(len(list_selected_peaks_coords))

        else:

            list_neigh_coords, peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(
                orig_peak_coord_np, ROI_cube_size, template_data, orig_peak_coords_dt)
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)

            if peak_template_roi_index > 0:

                neigh_coords = np.array(list_neigh_coords, dtype='int16')

                indexed_mask_rois_data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2]] = len(
                    list_selected_peaks_coords)

                list_selected_peaks_coords.append(orig_peak_coord_np)

                list_selected_peaks_indexes.append(i)

                print(len(list_selected_peaks_coords))

    return list_selected_peaks_coords, indexed_mask_rois_data, list_selected_peaks_indexes


def remove_close_peaks_neigh_in_template(list_orig_peak_coords, template_data, template_labels, min_dist=3.0 * np.sqrt(3)):

    img_shape = template_data.shape

    indexed_mask_rois_data = np.zeros(img_shape, dtype='int64') - 1

    print(indexed_mask_rois_data.shape)

    label_rois = []

    list_selected_peaks_coords = []

    orig_peak_coords_np = np.array(list_orig_peak_coords)

    print(type(orig_peak_coords_np),
          orig_peak_coords_np.dtype, orig_peak_coords_np.shape)

    orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)

    print(type(orig_peak_coords_dt),
          orig_peak_coords_dt.dtype, orig_peak_coords_dt.shape)

    for orig_peak_coord in list_orig_peak_coords:

        orig_peak_coord_np = np.array(orig_peak_coord)

        if len(list_selected_peaks_coords) > 0:

            selected_peaks_coords_np = np.array(list_selected_peaks_coords)

            dist_to_selected_peaks = dist.cdist(
                selected_peaks_coords_np, orig_peak_coord_np.reshape(1, 3), 'euclidean')

            min_dist_to_selected_peaks = np.amin(
                dist_to_selected_peaks, axis=0)

            if min_dist < min_dist_to_selected_peaks:

                list_neigh_coords, peak_template_roi_index = return_indexed_mask_neigh_within_template(
                    orig_peak_coord_np, ROI_cube_size, template_data, orig_peak_coords_dt)
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)

                if peak_template_roi_index > 0:

                    neigh_coords = np.array(list_neigh_coords, dtype='int16')

                    indexed_mask_rois_data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2]] = len(
                        list_selected_peaks_coords)

                    label_rois.append(
                        template_labels[peak_template_roi_index-1])

                    list_selected_peaks_coords.append(orig_peak_coord_np)

        else:
            list_neigh_coords, peak_template_roi_index = return_indexed_mask_neigh_within_template(
                orig_peak_coord_np, ROI_cube_size, template_data, orig_peak_coords_dt)
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)

            if peak_template_roi_index > 0:

                neigh_coords = np.array(list_neigh_coords, dtype='int16')

                indexed_mask_rois_data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2]] = len(
                    list_selected_peaks_coords)

                label_rois.append(template_labels[peak_template_roi_index-1])

                list_selected_peaks_coords.append(orig_peak_coord_np)

        print(len(list_selected_peaks_coords))

    return list_selected_peaks_coords, indexed_mask_rois_data, label_rois


def compute_labelled_mask_from_HO_all_signif_contrasts():

    write_dir = os.path.join(nipype_analyses_path,
                             peak_activation_mask_analysis_name)

    print(spm_contrasts_path)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    #spm_mask_files =  glob.glob(os.path.join(spm_contrasts_path,rel_spm_mask_path,"_contrast_index_[1-6]_group_contrast_index_0/spmT_*.img"))
    spm_mask_files = glob.glob(os.path.join(
        spm_contrasts_path, contrast_pattern))

    print(spm_mask_files)

    print(spm_mask_files.sort())

    # prepare the data
    img = nib.load(spm_mask_files[0])

    img_header = img.get_header()
    img_affine = img.get_affine()
    img_shape = img.shape

    img_data = img.get_data()

    # Computing combined HO areas

    resliced_full_HO_data, HO_labels, HO_abbrev_labels = compute_recombined_HO_template(
        img_header, img_affine, img_shape)

    # Creating peak activation mask contrained by HO areas

    # print len(HO_abbrev_labels)
    # print len(HO_labels)

    # 0/0
    np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

    np_HO_labels = np.array(HO_labels, dtype='string')

    template_indexes = np.unique(resliced_full_HO_data)[1:]

    # print template_indexes

    print(np_HO_labels.shape, np_HO_abbrev_labels.shape, template_indexes.shape)

    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_template = np.hstack((template_indexes.reshape(len(HO_labels), 1), np_HO_labels.reshape(
        len(HO_labels), 1), np_HO_abbrev_labels.reshape(len(HO_labels), 1)))
    # ,rois_MNI_coords))

    print(info_template)

    np.savetxt(info_template_file, info_template, fmt='%s %s %s')

    #np.savetxt(info_template_file,info_rois, fmt = '%s %s %s %s %s %s')

    indexed_mask_rois_files = []

    coord_rois_files = []

    for i, spm_mask_file in enumerate(spm_mask_files):

        print(spm_mask_file)

        spm_mask_img = nib.load(spm_mask_file)

        spm_mask_data = spm_mask_img.get_data()

        # get peaks (avec la fonction stat_map.get_3d_peaks)
        peaks = stat_map.get_3d_peaks(
            image=spm_mask_img, mask=None, threshold=threshold, nn=cluster_nbvoxels)

        # print len(peaks)

        list_orig_ROI_spm_index = []

        if peaks != None:

            print(len(peaks))
            list_orig_peak_vals = [peak['val'] for peak in peaks]
            list_orig_peak_coords = [peak['ijk'] for peak in peaks]
            list_orig_peak_MNI_coords = [peak['pos'] for peak in peaks]

            merged_mask_data = spm_mask_data[np.logical_and(
                spm_mask_data != 0.0, np.logical_not(np.isnan(spm_mask_data)))]

            list_orig_ROI_spm_index = list_orig_ROI_spm_index + \
                [i+1] * len(peaks)

            print(len(list_orig_peak_coords))
            print(len(list_orig_ROI_spm_index))

            list_selected_peaks_coords, indexed_mask_rois_data, list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(
                list_orig_peak_coords, resliced_full_HO_data, min_dist_between_ROIs)

            print(list_selected_peaks_indexes)
            print(len(list_selected_peaks_indexes))

            merged_mask_data[indexed_mask_rois_data != 0] += i+1

            template_indexes = np.array([resliced_full_HO_data[coord[0], coord[1], coord[2]]
                                         for coord in list_selected_peaks_coords], dtype='int64')
            print(template_indexes)

            np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

            np_HO_labels = np.array(HO_labels, dtype='string')

            print(template_indexes-1)

            label_rois = np_HO_abbrev_labels[template_indexes-1]
            full_label_rois = np_HO_labels[template_indexes-1]

            # print label_rois2

            print(label_rois)

            # indexed_mask
            indexed_mask_rois_file = os.path.join(nipype_analyses_path, peak_activation_mask_analysis_name,
                                                  "indexed_mask-" + ROI_mask_prefix + "_spm_contrast" + str(i+1) + ".nii")

            # saving ROI coords as textfile
            # ijk coords
            coord_rois_file = os.path.join(nipype_analyses_path, peak_activation_mask_analysis_name,
                                           "coords-" + ROI_mask_prefix + "_spm_contrast" + str(i+1) + ".txt")

            # coords in MNI space
            MNI_coord_rois_file = os.path.join(nipype_analyses_path, peak_activation_mask_analysis_name,
                                               "coords-MNI-" + ROI_mask_prefix + "_spm_contrast" + str(i+1) + ".txt")

            # saving ROI coords as textfile
            label_rois_file = os.path.join(nipype_analyses_path, peak_activation_mask_analysis_name,
                                           "labels-" + ROI_mask_prefix + "_spm_contrast" + str(i+1) + ".txt")
            #label_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + "_jane.txt")

            # all info in a text file
            info_rois_file = os.path.join(nipype_analyses_path, peak_activation_mask_analysis_name,
                                          "info-" + ROI_mask_prefix + "_spm_contrast" + str(i+1) + ".txt")

            # exporting Rois image with different indexes
            print(np.unique(indexed_mask_rois_data)[1:].shape)
            nib.save(nib.Nifti1Image(data=indexed_mask_rois_data,
                                     header=img_header, affine=img_affine), indexed_mask_rois_file)

            # saving ROI coords as textfile
            np.savetxt(coord_rois_file, np.array(
                list_selected_peaks_coords, dtype=int), fmt='%d')

            # saving MNI coords as textfile
            list_rois_MNI_coords = [list_orig_peak_MNI_coords[index]
                                    for index in list_selected_peaks_indexes]

            print(list_rois_MNI_coords)

            rois_MNI_coords = np.array(list_rois_MNI_coords, dtype=int)
            np.savetxt(MNI_coord_rois_file, rois_MNI_coords, fmt='%d')

            # orig index of peaks
            list_rois_orig_indexes = [list_orig_ROI_spm_index[index]
                                      for index in list_selected_peaks_indexes]

            print(list_rois_orig_indexes)

            rois_orig_indexes = np.array(list_rois_orig_indexes, dtype=int).reshape(
                len(list_rois_orig_indexes), 1)

            print(rois_orig_indexes.shape)

            # saving labels
            np.savetxt(label_rois_file, label_rois, fmt='%s')

            # saving all together for infosource
            np_label_rois = np.array(
                label_rois, dtype='string').reshape(len(label_rois), 1)
            np_full_label_rois = np.array(
                full_label_rois, dtype='string').reshape(len(full_label_rois), 1)

            print(np_label_rois.shape)
            print(rois_MNI_coords.shape)

            #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
            #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
            info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(
                label_rois), 1), np_full_label_rois, np_label_rois, rois_MNI_coords, rois_orig_indexes))

            print(info_rois)

            np.savetxt(info_rois_file, info_rois, fmt='%s %s %s %s %s %s %s')

            indexed_mask_rois_files.append(indexed_mask_rois_file)
            coord_rois_files.append(coord_rois_file)

    return indexed_mask_rois_files, coord_rois_files


def compute_labelled_mask_from_HO_and_merged_spm_mask():

    write_dir = os.path.join(nipype_analyses_path,
                             peak_activation_mask_analysis_name)

    print(spm_contrasts_path)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    spm_mask_files = glob.glob(os.path.join(
        spm_contrasts_path, contrast_pattern))

    print(spm_mask_files)

    print(spm_mask_files.sort())

    # prepare the data
    img = nib.load(spm_mask_files[0])

    img_header = img.get_header()
    img_affine = img.get_affine()
    img_shape = img.shape

    img_data = img.get_data()

    # Computing combined HO areas

    resliced_full_HO_data, HO_labels, HO_abbrev_labels = compute_recombined_HO_template(
        img_header, img_affine, img_shape)

    # Creating peak activation mask contrained by HO areas

    # print len(HO_abbrev_labels)
    # print len(HO_labels)

    # 0/0
    np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

    np_HO_labels = np.array(HO_labels, dtype='string')

    template_indexes = np.unique(resliced_full_HO_data)[1:]

    # print template_indexes

    print(np_HO_labels.shape, np_HO_abbrev_labels.shape, template_indexes.shape)

    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_template = np.hstack((template_indexes.reshape(len(HO_labels), 1), np_HO_labels.reshape(
        len(HO_labels), 1), np_HO_abbrev_labels.reshape(len(HO_labels), 1)))
    # ,rois_MNI_coords))

    print(info_template)

    np.savetxt(info_template_file, info_template, fmt='%s %s %s')

    #np.savetxt(info_template_file,info_rois, fmt = '%s %s %s %s %s %s')

    merged_mask_data = np.zeros(shape=img_shape, dtype=float)

    print(merged_mask_data.shape)

    list_orig_ROI_spm_index = []

    # list for all info about peaks after merging between different contrasts
    list_orig_peak_coords = []
    list_orig_peak_MNI_coords = []
    list_orig_peak_vals = []

    for i, spm_mask_file in enumerate(spm_mask_files):

        print(spm_mask_file)

        spm_mask_img = nib.load(spm_mask_file)

        spm_mask_data = spm_mask_img.get_data()

        # get peaks (avec la fonction stat_map.get_3d_peaks)
        peaks = stat_map.get_3d_peaks(
            image=spm_mask_img, mask=None, threshold=threshold, nn=cluster_nbvoxels)

        # print len(peaks)

        if peaks != None:

            print(len(peaks))
            list_orig_peak_vals = list_orig_peak_vals + \
                [peak['val'] for peak in peaks]
            list_orig_peak_coords = list_orig_peak_coords + \
                [peak['ijk'] for peak in peaks]
            list_orig_peak_MNI_coords = list_orig_peak_MNI_coords + \
                [peak['pos'] for peak in peaks]

            # print list_orig_peak_vals

            # print np.where(np.isnan(spm_mask_data))

            # print spm_mask_data[]

            #merged_mask_data[np.logical_and(spm_mask_data != 0.0, np.logical_not(np.isnan(spm_mask_data)))] = 1.0

            merged_mask_data[spm_mask_data > threshold] += i+1

            # print np.sum(np.logical_and(merged_mask_data != 0.0, np.logical_not(np.isnan(merged_mask_data))))

            list_orig_ROI_spm_index = list_orig_ROI_spm_index + \
                [i+1] * len(peaks)

        print(len(list_orig_peak_coords))
        print(len(list_orig_ROI_spm_index))

    # selectionne les pics sur leur distance entre eux et sur leur appatenance au template HO
    list_selected_peaks_coords, indexed_mask_rois_data, list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(
        list_orig_peak_coords, resliced_full_HO_data, min_dist_between_ROIs)

    nib.save(nib.Nifti1Image(data=merged_mask_data, header=img_header,
                             affine=img_affine), merged_mask_img_file)

    print(list_selected_peaks_indexes)
    print(len(list_selected_peaks_indexes))

    template_indexes = np.array([resliced_full_HO_data[coord[0], coord[1], coord[2]]
                                 for coord in list_selected_peaks_coords], dtype='int64')
    print(template_indexes)

    np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

    np_HO_labels = np.array(HO_labels, dtype='string')

    print(template_indexes-1)

    label_rois = np_HO_abbrev_labels[template_indexes-1]
    full_label_rois = np_HO_labels[template_indexes-1]

    # print label_rois2

    print(label_rois)

    # exporting Rois image with different indexes
    print(np.unique(indexed_mask_rois_data)[1:].shape)
    nib.save(nib.Nifti1Image(data=indexed_mask_rois_data,
                             header=img_header, affine=img_affine), indexed_mask_rois_file)

    # saving ROI coords as textfile
    np.savetxt(coord_rois_file, np.array(
        list_selected_peaks_coords, dtype=int), fmt='%d')

    # saving MNI coords as textfile
    list_rois_MNI_coords = [list_orig_peak_MNI_coords[index]
                            for index in list_selected_peaks_indexes]

    print(list_rois_MNI_coords)

    rois_MNI_coords = np.array(list_rois_MNI_coords, dtype=int)
    np.savetxt(MNI_coord_rois_file, rois_MNI_coords, fmt='%d')

    # orig index of peaks
    list_rois_orig_indexes = [list_orig_ROI_spm_index[index]
                              for index in list_selected_peaks_indexes]

    print(list_rois_orig_indexes)

    rois_orig_indexes = np.array(list_rois_orig_indexes, dtype=int).reshape(
        len(list_rois_orig_indexes), 1)

    print(rois_orig_indexes.shape)

    # mask with orig spm index
    orig_spm_index_mask_data = np.zeros(shape=img_shape, dtype=int)

    print(np.unique(indexed_mask_rois_data))

    for i in np.unique(indexed_mask_rois_data)[1:]:

        print(i, np.sum(indexed_mask_rois_data == i), rois_orig_indexes[i])

        orig_spm_index_mask_data[indexed_mask_rois_data ==
                                 i] = rois_orig_indexes[i]

    nib.save(nib.Nifti1Image(data=orig_spm_index_mask_data,
                             header=img_header, affine=img_affine), orig_spm_index_mask_file)

    # saving labels
    np.savetxt(label_rois_file, label_rois, fmt='%s')

    # saving all together for infosource
    np_label_rois = np.array(
        label_rois, dtype='string').reshape(len(label_rois), 1)
    np_full_label_rois = np.array(
        full_label_rois, dtype='string').reshape(len(full_label_rois), 1)

    print(np_label_rois.shape)
    print(rois_MNI_coords.shape)

    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(
        label_rois), 1), np_full_label_rois, np_label_rois, rois_MNI_coords, rois_orig_indexes))

    print(info_rois)

    np.savetxt(info_rois_file, info_rois, fmt='%s %s %s %s %s %s %s')

    return indexed_mask_rois_file, coord_rois_file


def compute_labelled_mask_from_HO_and_merged_thr_spm_mask():

    write_dir = os.path.join(nipype_analyses_path,
                             peak_activation_mask_analysis_name)

    print(spm_contrasts_path)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    spm_contrast_indexes = [3, 4, 5, 8, 9, 10]

    spm_mask_files = [os.path.join(spm_contrasts_path, "_contrast_index_"+str(
        index)+"_group_contrast_index_0/spmT_0001_thr.img") for index in spm_contrast_indexes]

    # spm_mask_files.sort()

    print(len(spm_mask_files))

    # prepare the data
    img = nib.load(spm_mask_files[0])

    img_header = img.get_header()
    img_affine = img.get_affine()
    img_shape = img.shape

    img_data = img.get_data()

    # Computing combined HO areas

    resliced_full_HO_data, HO_labels, HO_abbrev_labels = compute_recombined_HO_template(
        img_header, img_affine, img_shape)

    # Creating peak activation mask contrained by HO areas

    # print len(HO_abbrev_labels)
    # print len(HO_labels)

    # 0/0
    np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

    np_HO_labels = np.array(HO_labels, dtype='string')

    template_indexes = np.unique(resliced_full_HO_data)[1:]

    # print template_indexes

    print(np_HO_labels.shape, np_HO_abbrev_labels.shape, template_indexes.shape)

    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_template = np.hstack((template_indexes.reshape(len(HO_labels), 1), np_HO_labels.reshape(
        len(HO_labels), 1), np_HO_abbrev_labels.reshape(len(HO_labels), 1)))
    # ,rois_MNI_coords))

    print(info_template)

    np.savetxt(info_template_file, info_template, fmt='%s %s %s')

    #np.savetxt(info_template_file,info_rois, fmt = '%s %s %s %s %s %s')

    merged_mask_data = np.zeros(shape=img_shape, dtype=float)

    print(merged_mask_data.shape)

    list_orig_ROI_spm_index = []

    # list for all info about peaks after merging between different contrasts
    list_orig_peak_coords = []
    list_orig_peak_MNI_coords = []
    list_orig_peak_vals = []

    for i, spm_mask_file in enumerate(spm_mask_files):

        print(spm_mask_file)

        spm_mask_img = nib.load(spm_mask_file)

        spm_mask_data = spm_mask_img.get_data()

        # get peaks (avec la fonction stat_map.get_3d_peaks)
        peaks = stat_map.get_3d_peaks(image=spm_mask_img, mask=None)

        # print len(peaks)

        if peaks != None:

            print(len(peaks))
            list_orig_peak_vals = list_orig_peak_vals + \
                [peak['val'] for peak in peaks]
            list_orig_peak_coords = list_orig_peak_coords + \
                [peak['ijk'] for peak in peaks]
            list_orig_peak_MNI_coords = list_orig_peak_MNI_coords + \
                [peak['pos'] for peak in peaks]

            # print list_orig_peak_vals

            # print np.where(np.isnan(spm_mask_data))

            # print spm_mask_data[]

            #merged_mask_data[np.logical_and(spm_mask_data != 0.0, np.logical_not(np.isnan(spm_mask_data)))] = 1.0

            merged_mask_data[np.logical_and(
                spm_mask_data != 0.0, np.logical_not(np.isnan(spm_mask_data)))] += i+1

            # print np.sum(np.logical_and(merged_mask_data != 0.0, np.logical_not(np.isnan(merged_mask_data))))

            list_orig_ROI_spm_index = list_orig_ROI_spm_index + \
                [i+1] * len(peaks)

        print(len(list_orig_peak_coords))
        print(len(list_orig_ROI_spm_index))

    # selectionne les pics sur leur distance entre eux et sur leur appatenance au template HO

    list_selected_peaks_coords, indexed_mask_rois_data, list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(
        list_orig_peak_coords, resliced_full_HO_data, min_dist_between_ROIs)
    #list_selected_peaks_coords,indexed_mask_rois_data,list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(sorded_merged_peaks_coords,resliced_full_HO_data,min_dist_between_ROIs)

    nib.save(nib.Nifti1Image(data=merged_mask_data, header=img_header,
                             affine=img_affine), merged_mask_img_file)

    print(list_selected_peaks_indexes)
    print(len(list_selected_peaks_indexes))

    # for coord in list_selected_peaks_coords:

    # print coord
    # template_indexes =
    # print resliced_full_HO_data[coord[0],coord[1],coord[2]]

    template_indexes = np.array([resliced_full_HO_data[coord[0], coord[1], coord[2]]
                                 for coord in list_selected_peaks_coords], dtype='int64')
    print(template_indexes)

    np_HO_abbrev_labels = np.array(HO_abbrev_labels, dtype='string')

    np_HO_labels = np.array(HO_labels, dtype='string')

    print(template_indexes-1)

    label_rois = np_HO_abbrev_labels[template_indexes-1]
    full_label_rois = np_HO_labels[template_indexes-1]

    # print label_rois2

    print(label_rois)

    # exporting Rois image with different indexes
    print(np.unique(indexed_mask_rois_data)[1:].shape)
    nib.save(nib.Nifti1Image(data=indexed_mask_rois_data,
                             header=img_header, affine=img_affine), indexed_mask_rois_file)

    # saving ROI coords as textfile
    np.savetxt(coord_rois_file, np.array(
        list_selected_peaks_coords, dtype=int), fmt='%d')

    # saving MNI coords as textfile
    list_rois_MNI_coords = [list_orig_peak_MNI_coords[index]
                            for index in list_selected_peaks_indexes]

    print(list_rois_MNI_coords)

    rois_MNI_coords = np.array(list_rois_MNI_coords, dtype=int)
    np.savetxt(MNI_coord_rois_file, rois_MNI_coords, fmt='%d')

    # orig index of peaks
    list_rois_orig_indexes = [list_orig_ROI_spm_index[index]
                              for index in list_selected_peaks_indexes]

    print(list_rois_orig_indexes)

    rois_orig_indexes = np.array(list_rois_orig_indexes, dtype=int).reshape(
        len(list_rois_orig_indexes), 1)

    print(rois_orig_indexes.shape)

    np.savetxt(rois_orig_indexes_file, rois_orig_indexes, fmt='%d')

    # mask with orig spm index
    orig_spm_index_mask_data = np.zeros(shape=img_shape, dtype=int)

    print(np.unique(indexed_mask_rois_data))

    for i in np.unique(indexed_mask_rois_data)[1:]:

        print(i, np.sum(indexed_mask_rois_data == i), rois_orig_indexes[i])

        orig_spm_index_mask_data[indexed_mask_rois_data ==
                                 i] = rois_orig_indexes[i]

    nib.save(nib.Nifti1Image(data=orig_spm_index_mask_data,
                             header=img_header, affine=img_affine), orig_spm_index_mask_file)

    # saving labels
    np.savetxt(label_rois_file, label_rois, fmt='%s')

    # saving all together for infosource
    np_label_rois = np.array(
        label_rois, dtype='string').reshape(len(label_rois), 1)
    np_full_label_rois = np.array(
        full_label_rois, dtype='string').reshape(len(full_label_rois), 1)

    print(np_label_rois.shape)
    print(rois_MNI_coords.shape)

    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(
        label_rois), 1), np_full_label_rois, np_label_rois, rois_MNI_coords, rois_orig_indexes))

    print(info_rois)

    np.savetxt(info_rois_file, info_rois, fmt='%s %s %s %s %s %s %s')

    return indexed_mask_rois_file, coord_rois_file


if __name__ == '__main__':

    # compute_labelled_mask_from_HO()
    # compute_labelled_mask_from_HO_all_signif_contrasts()

    compute_labelled_mask_from_HO_and_merged_thr_spm_mask()
