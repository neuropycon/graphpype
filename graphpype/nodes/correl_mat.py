# -*- coding: utf-8 -*-

"""
Definition of Nodes for computing correlation matrices 
"""

#import nipy.labs.statistical_mapping as stat_map

#import itertools as iter

#import scipy.spatial.distance as dist

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined

from nipype.utils.filemanip import split_filename as split_f

import nibabel as nb
import numpy as np
import os

import nibabel as nib

from graphpype.utils_plot import plot_signals, plot_sep_signals

######################################################################################## ExtractTS ##################################################################################################################

from graphpype.utils_cor import mean_select_indexed_mask_data


class ExtractTSInputSpec(BaseInterfaceInputSpec):
    indexed_rois_file = File(
        exists=True, desc='indexed mask where all voxels belonging to the same ROI have the same value (! starting from 1)', mandatory=True)

    file_4D = File(
        exists=True, desc='4D volume to be extracted', mandatory=True)

    MNI_coord_rois_file = File(desc='ROI MNI_coordinates')

    coord_rois_file = File(desc='ROI coordinates')

    label_rois_file = File(desc='ROI labels')

    min_BOLD_intensity = traits.Float(
        50.0, desc='BOLD signal below this value will be set to zero', usedefault=True)

    percent_signal = traits.Float(
        0.5, desc="Percent of voxels in a ROI with signal higher that min_BOLD_intensity to keep this ROI", usedefault=True)

    plot_fig = traits.Bool(
        False, desc="Plotting mean signal or not", usedefault=True)

    background_val = traits.Float(
        -1.0, desc='value for background (i.e. outside brain)', usedefault=True)


class ExtractTSOutputSpec(TraitedSpec):

    mean_masked_ts_file = File(
        exists=True, desc="mean ts in .npy (pickle format)")

    subj_coord_rois_file = File(
        desc="ROI coordinates retained for the subject")

    subj_MNI_coord_rois_file = File(
        desc="ROI MNI_coordinates retained for the subject")

    subj_label_rois_file = File(desc="ROI labels retained for the subject")


class ExtractTS(BaseInterface):

    """

    Description: Extract time series from a labelled mask in Nifti Format where all ROIs have the same index

    Inputs:

        indexed_rois_file:
            type = File, exists=True, desc='indexed mask where all voxels belonging to the same ROI have the same value (! starting from 1)', mandatory=True

        file_4D:
            type = File, exists=True, desc='4D volume to be extracted', mandatory=True

        MNI_coord_rois_file:
            typr = File, desc='ROI MNI_coordinates'

        coord_rois_file:
            type = File, desc='ROI coordinates'

        label_rois_file:
            type = File, desc='ROI labels')

        min_BOLD_intensity:
            type = Float, default = 50.0, desc='BOLD signal below this value will be set to zero',usedefault = True

        percent_signal:
            type = Float, default = 0.5, desc  = "Percent of voxels in a ROI with signal higher that min_BOLD_intensity to keep this ROI",usedefault = True

        plot_fig:
            type = Bool, defaults = False, desc = "Plotting mean signal or not", usedefault = True)

        background_val:
            type = Float, -1.0, desc='value for background (i.e. outside brain)',usedefault = True

    Outputs:

        mean_masked_ts_file: 
            type = File, exists=True, desc="mean ts in .npy (pickle format)"

        subj_coord_rois_file: 
            type = File, desc="ROI coordinates retained for the subject"

        subj_MNI_coord_rois_file: 
            type = File, desc="ROI MNI_coordinates retained for the subject"


        subj_label_rois_file: 
            type = File, desc="ROI labels retained for the subject"



    """

    input_spec = ExtractTSInputSpec
    output_spec = ExtractTSOutputSpec

    def _run_interface(self, runtime):

        indexed_rois_file = self.inputs.indexed_rois_file
        file_4D = self.inputs.file_4D
        min_BOLD_intensity = self.inputs.min_BOLD_intensity
        percent_signal = self.inputs.percent_signal
        background_val = self.inputs.background_val

        plot_fig = self.inputs.plot_fig

        # loading ROI indexed mask
        indexed_rois_img = nib.load(indexed_rois_file)
        indexed_mask_rois_data = indexed_rois_img.get_data()

        # loading time series
        orig_ts = nib.load(file_4D).get_data()

        print("orig_ts shape:")
        print(orig_ts.shape)

        mean_masked_ts, keep_rois = mean_select_indexed_mask_data(
            orig_ts, indexed_mask_rois_data, min_BOLD_intensity, percent_signal=percent_signal, background_val=background_val)

        # loading ROI coordinates

        if isdefined(self.inputs.MNI_coord_rois_file):

            MNI_coord_rois = np.loadtxt(self.inputs.MNI_coord_rois_file)

            subj_MNI_coord_rois = MNI_coord_rois[keep_rois, :]

            # saving subject ROIs
            subj_MNI_coord_rois_file = os.path.abspath(
                "subj_MNI_coord_rois.txt")
            np.savetxt(subj_MNI_coord_rois_file,
                       subj_MNI_coord_rois, fmt='%.3f')

        if isdefined(self.inputs.coord_rois_file):

            coord_rois = np.loadtxt(self.inputs.coord_rois_file)

            subj_coord_rois = coord_rois[keep_rois, :]

            # saving subject ROIs
            subj_coord_rois_file = os.path.abspath("subj_coord_rois.txt")
            np.savetxt(subj_coord_rois_file, subj_coord_rois, fmt='%.3f')

        if isdefined(self.inputs.label_rois_file):

            labels_rois = np.array([line.strip() for line in open(
                self.inputs.label_rois_file)], dtype='str')

            subj_label_rois = labels_rois[keep_rois]

            # saving subject ROIs
            subj_label_rois_file = os.path.abspath("subj_label_rois.txt")
            np.savetxt(subj_label_rois_file, subj_label_rois, fmt='%s')

        mean_masked_ts = np.array(mean_masked_ts, dtype='f')

        # saving time series
        mean_masked_ts_file = os.path.abspath("mean_masked_ts.txt")
        np.savetxt(mean_masked_ts_file, mean_masked_ts, fmt='%.3f')

        if plot_fig == True:

            print("plotting mean_masked_ts")

            plot_mean_masked_ts_file = os.path.abspath('mean_masked_ts.eps')

            plot_signals(plot_mean_masked_ts_file, mean_masked_ts)

        return runtime

        # return mean_masked_ts_file,subj_coord_rois_file

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["mean_masked_ts_file"] = os.path.abspath("mean_masked_ts.txt")

        if isdefined(self.inputs.MNI_coord_rois_file):

            outputs["subj_MNI_coord_rois_file"] = os.path.abspath(
                "subj_MNI_coord_rois.txt")

        if isdefined(self.inputs.coord_rois_file):

            outputs["subj_coord_rois_file"] = os.path.abspath(
                "subj_coord_rois.txt")

        if isdefined(self.inputs.label_rois_file):

            outputs["subj_label_rois_file"] = os.path.abspath(
                "subj_label_rois.txt")

        return outputs


######################################################################################## IntersectMask ##################################################################################################################

from graphpype.utils_cor import mean_select_indexed_mask_data


class IntersectMaskInputSpec(BaseInterfaceInputSpec):

    indexed_rois_file = File(
        exists=True, desc='nii file with indexed mask where all voxels belonging to the same ROI have the same value (! starting from 0)', mandatory=True)

    filter_mask_file = File(
        exists=True, desc='nii file with (binary) mask - e.g. grey matter mask', mandatory=True)

    coords_rois_file = File(desc='ijk coords txt file')

    labels_rois_file = File(desc='labels txt file')

    MNI_coords_rois_file = File(desc='MNI coords txt file')

    filter_thr = traits.Float(0.99, usedefault=True,
                              desc='Value to threshold filter_mask')

    background_val = traits.Float(
        -1.0, desc='value for background (i.e. outside brain)', usedefault=True)


class IntersectMaskOutputSpec(TraitedSpec):

    filtered_indexed_rois_file = File(
        exists=True, desc='nii file with indexed mask where all voxels belonging to the same ROI have the same value (! starting from 0)')

    filtered_coords_rois_file = File(
        exists=False, desc='filtered ijk coords txt file')

    filtered_labels_rois_file = File(
        exists=False, desc='filtered labels txt file')

    filtered_MNI_coords_rois_file = File(
        exists=False, desc='filtered MNI coords txt file')


class IntersectMask(BaseInterface):

    """
    Description:

    Keep only values of indexed mask where filter_mask is present.
    Optionnally, keep only ijk_coords, MNI_coords and labels that are kept in filtered mask

    Inputs:

        indexed_rois_file:
            type = File, exists=True, desc='nii file with indexed mask where all voxels belonging to the same ROI have the same value (! starting from 0)', mandatory=True

        filter_mask_file:
            type = File, exists=True, desc='nii file with (binary) mask - e.g. grey matter mask', mandatory=True

        coords_rois_file:
            type = File, desc='ijk coords txt file'

        labels_rois_file:
            type = File, desc='labels txt file'

        MNI_coords_rois_file:
            type = File, desc='MNI coords txt file'

        filter_thr:
            type = Float, default = 0.99, usedefault = True, desc='Value to threshold filter_mask'

        background_val:
            type = Float, -1.0, desc='value for background (i.e. outside brain)',usedefault = True

    Outputs:

        filtered_indexed_rois_file:
            type = File, exists=True, desc='nii file with indexed mask where all voxels belonging to the same ROI have the same value (! starting from 0)'

        filtered_coords_rois_file:
            type = File, exists=False, desc='filtered ijk coords txt file' 

        filtered_labels_rois_file:
            type = File, exists=False, desc='filtered labels txt file'

        filtered_MNI_coords_rois_file:
            type = File, exists=False, desc='filtered MNI coords txt file'

    """

    input_spec = IntersectMaskInputSpec
    output_spec = IntersectMaskOutputSpec

    def _run_interface(self, runtime):

        #import os
        #import numpy as np
        import nibabel as nib

        #import nipype.interfaces.spm as spm

        #from graphpype.utils_plot import plot_signals

        indexed_rois_file = self.inputs.indexed_rois_file
        filter_mask_file = self.inputs.filter_mask_file
        coords_rois_file = self.inputs.coords_rois_file
        labels_rois_file = self.inputs.labels_rois_file
        MNI_coords_rois_file = self.inputs.MNI_coords_rois_file
        background_val = self.inputs.background_val

        filter_thr = self.inputs.filter_thr

        # loading ROI indexed mask
        indexed_rois_img = nib.load(indexed_rois_file)

        indexed_rois_data = indexed_rois_img.get_data()

        print(np.where(np.isnan(indexed_rois_data)))

        indexed_rois_data[np.isnan(indexed_rois_data)] = background_val

        print(np.unique(indexed_rois_data))

        # previous version, error when nan
        #indexed_rois_data = np.array(indexed_rois_img.get_data(),dtype = 'int')

        # loading time series
        filter_mask_data = nib.load(filter_mask_file).get_data()

        assert filter_mask_data.shape == indexed_rois_data.shape, "error, filter_mask {} and indexed_rois {} should have the same shape".format(
            filter_mask_data.shape, indexed_rois_data.shape)

        filter_mask_data[filter_mask_data > filter_thr] = 1.0
        filter_mask_data[filter_mask_data <= filter_thr] = 0.0

        # print np.unique(filter_mask_data)
        print("indexed_rois_data:")
        print(np.unique(indexed_rois_data))
        print(len(np.unique(indexed_rois_data)))

        if background_val == -1.0:

            filtered_indexed_rois_data = np.array(
                filter_mask_data * (indexed_rois_data.copy()+1) - 1, dtype='int64')

        elif background_val == 0.0:

            filtered_indexed_rois_data = np.array(
                filter_mask_data * indexed_rois_data, dtype='int64')

        print("filtered_indexed_rois_data:")
        print(np.unique(filtered_indexed_rois_data))
        print(len(np.unique(filtered_indexed_rois_data)))

        #filtered_indexed_rois_img_file = os.path.abspath("filtered_indexed_rois.nii")
        # nib.save(nib.Nifti1Image(filtered_indexed_rois_data,indexed_rois_img.get_affine(),indexed_rois_img.get_header()),filtered_indexed_rois_img_file)

        print("reorder_indexed_rois:")
        reorder_indexed_rois_data = np.zeros(
            shape=filtered_indexed_rois_data.shape, dtype='int64') - 1

        i = 0

        for index in np.unique(filtered_indexed_rois_data)[1:]:

            print(i, index)

            if np.sum(np.array(filtered_indexed_rois_data == index, dtype=int)) != 0:

                #print(np.sum(np.array(filtered_indexed_rois_data == index,dtype = int)))
                reorder_indexed_rois_data[filtered_indexed_rois_data == index] = i
                i = i+1
            else:
                print(
                    "Warning could not find value %d in filtered_indexed_rois_data" % index)

        # print(np.unique(reorder_indexed_rois_data))

        reorder_indexed_rois_img_file = os.path.abspath(
            "reorder_filtered_indexed_rois.nii")
        nib.save(nib.Nifti1Image(reorder_indexed_rois_data, indexed_rois_img.get_affine(
        ), indexed_rois_img.get_header()), reorder_indexed_rois_img_file)

        print('unique np.unique(filtered_indexed_rois_data)')
        print(np.unique(filtered_indexed_rois_data))

        print("index_corres:")

        if background_val == -1.0:
            index_corres = np.unique(filtered_indexed_rois_data)[1:]

        elif background_val == 0.0:
            index_corres = np.unique(filtered_indexed_rois_data)[1:]-1

        print(index_corres)
        print(len(index_corres))

        if isdefined(coords_rois_file):

            # loading ROI coordinates
            coords_rois = np.loadtxt(coords_rois_file)

            filtered_coords_rois = coords_rois[index_corres, :]

            filtered_coords_rois_file = os.path.abspath(
                "filtered_coords_rois.txt")
            np.savetxt(filtered_coords_rois_file,
                       filtered_coords_rois, fmt="%d")

        if isdefined(MNI_coords_rois_file):

            # loading ROI coordinates
            MNI_coords_rois = np.loadtxt(MNI_coords_rois_file)

            filtered_MNI_coords_rois = MNI_coords_rois[index_corres, :]

            filtered_MNI_coords_rois_file = os.path.abspath(
                "filtered_MNI_coords_rois.txt")
            np.savetxt(filtered_MNI_coords_rois_file,
                       filtered_MNI_coords_rois, fmt="%f")

        if isdefined(labels_rois_file):

            print('extracting node labels')
            np_labels_rois = np.array(
                [line.strip() for line in open(labels_rois_file)], dtype='str')

            filtered_labels_rois = np_labels_rois[index_corres]

            filtered_labels_rois_file = os.path.abspath(
                "filtered_labels_rois.txt")
            np.savetxt(filtered_labels_rois_file,
                       filtered_labels_rois, fmt="%s")

        return runtime

        # return mean_masked_ts_file,subj_coord_rois_file

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["filtered_indexed_rois_file"] = os.path.abspath(
            "reorder_filtered_indexed_rois.nii")

        if isdefined(self.inputs.coords_rois_file):
            outputs["filtered_coords_rois_file"] = os.path.abspath(
                "filtered_coords_rois.txt")

        if isdefined(self.inputs.MNI_coords_rois_file):
            outputs["filtered_MNI_coords_rois_file"] = os.path.abspath(
                "filtered_MNI_coords_rois.txt")

        if isdefined(self.inputs.labels_rois_file):
            outputs["filtered_labels_rois_file"] = os.path.abspath(
                "filtered_labels_rois.txt")

        return outputs


############################################################################################### ExtractMeanTS #####################################################################################################
from graphpype.utils_cor import mean_select_mask_data
from graphpype.utils import check_np_dimension


class ExtractMeanTSInputSpec(BaseInterfaceInputSpec):

    file_4D = File(
        exists=True, desc='4D volume to be extracted', mandatory=True)

    ROI_coord = traits.List(traits.Int(exists=True), desc='values for extracting ROI',
                            mandatory=True, xor=['mask_file', 'filter_mask_file'])

    mask_file = File(xor=['filter_mask_file', 'ROI_coord'], exists=True,
                     desc='mask file where all voxels belonging to the selected region have index 1', mandatory=True)

    filter_mask_file = File(xor=['mask_file', 'ROI_coord'], requires=['filter_thr'], exists=True,
                            desc='mask file where all voxels belonging to the selected region have values higher than threshold', mandatory=True)

    filter_thr = traits.Float(0.99, usedefault=True,
                              desc='Value to threshold filter_mask')

    suffix = traits.String(
        "suf", desc='Suffix added to describe the extracted time series', mandatory=False, usedefault=True)

    plot_fig = traits.Bool(
        False, desc="Plotting mean signal or not", usedefault=True)


class ExtractMeanTSOutputSpec(TraitedSpec):

    mean_masked_ts_file = File(exists=True, desc="mean ts in .npy format")


class ExtractMeanTS(BaseInterface):

    """
    Description:

    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1 (mask_file), or from a percent mask (filter_mask_file) with values higher than threshold (filter_thr)

    Inputs:

        file_4D:
            type = File, exists=True, desc='4D volume to be extracted', mandatory=True

        mask_file:
            type = File, xor = ['filter_mask_file'], exists=True, desc='mask file where all voxels belonging to the selected region have index 1', mandatory=True

        filter_mask_file:
            type = File, xor = ['mask_file'],requires = ['filter_thr'], exists=True, desc='mask file where all voxels belonging to the selected region have values higher than threshold', mandatory=True

        filter_thr:
            type = Float, default = 0.99, usedefault = True, desc='Value to threshold filter_mask'

        suffix:
            type = String, default = "suf",desc='Suffix added to describe the extracted time series',mandatory=False,usedefault = True

        plot_fig:
            type = Bool, default = False, desc = "Plotting mean signal or not", usedefault = True

    Outputs:

        mean_masked_ts_file:
            type = File, exists=True, desc="mean ts in .npy format"


    """
    input_spec = ExtractMeanTSInputSpec
    output_spec = ExtractMeanTSOutputSpec

    def _run_interface(self, runtime):

        print('in select_ts_with_mask')

        file_4D = self.inputs.file_4D
        ROI_coord = self.inputs.ROI_coord
        mask_file = self.inputs.mask_file
        filter_mask_file = self.inputs.filter_mask_file
        filter_thr = self.inputs.filter_thr
        plot_fig = self.inputs.plot_fig

        suffix = self.inputs.suffix

        print("loading img data " + file_4D)

        # Reading 4D volume file to extract time series
        img = nib.load(file_4D)
        img_data = img.get_data()

        print(img_data.shape)

        # Reading 3D mask file
        if isdefined(mask_file):

            print("loading mask data " + mask_file)

            mask_data = nib.load(mask_file).get_data()

        elif isdefined(filter_mask_file) and isdefined(filter_thr):
            print("loading filter mask data " + filter_mask_file)

            filter_mask_data = nib.load(filter_mask_file).get_data()

            mask_data = np.zeros(shape=filter_mask_data.shape, dtype='int')

            mask_data[filter_mask_data > filter_thr] = 1

        elif isdefined(ROI_coord):

            print("In ROI coords")

            mask_data = np.zeros(shape=img_data.shape[:3], dtype=int)
            print(mask_data.shape)

            ROI_coord = np.array(ROI_coord, dtype=int)

            assert check_np_dimension(mask_data.shape, ROI_coord), "Error, non compatible indexes {} with shape {}".format(
                ROI_coord, mask_data.shape)

            mask_data[ROI_coord[0], ROI_coord[1], ROI_coord[2]] = 1

            # print np.where(mask_data == 1)

        else:
            print(
                "Error, either mask_file or (filter_mask_file and filter_thr) should be defined")

            return

        print(np.unique(mask_data))
        print(mask_data.shape)

        print("mean_select_mask_data")

        # Retaining only time series who are within the mask + non_zero
        mean_masked_ts = mean_select_mask_data(img_data, mask_data)

        print(mean_masked_ts)
        print(mean_masked_ts.shape)

        print("saving mean_masked_ts")
        mean_masked_ts_file = os.path.abspath('mean_' + suffix + '_ts.txt')
        np.savetxt(mean_masked_ts_file, mean_masked_ts, fmt='%.3f')

        if plot_fig == True:

            print("plotting mean_masked_ts")

            plot_mean_masked_ts_file = os.path.abspath(
                'mean_' + suffix + '_ts.eps')

            plot_signals(plot_mean_masked_ts_file, mean_masked_ts)

        return runtime

        # return mean_masked_ts_file,subj_coord_rois_file

    def _list_outputs(self):

        outputs = self._outputs().get()

        if isdefined(self.inputs.suffix):

            suffix = self.inputs.suffix

        else:

            suffix = "suf"

        outputs["mean_masked_ts_file"] = os.path.abspath(
            'mean_' + suffix + '_ts.txt')

        return outputs


######################################################################################## ConcatTS ##################################################################################################################

class ConcatTSInputSpec(BaseInterfaceInputSpec):

    all_ts_file = File(
        exists=True, desc='npy file containing all ts to be concatenated', mandatory=True)


class ConcatTSOutputSpec(TraitedSpec):

    concatenated_ts_file = File(exists=True, desc="ts after concatenation")


class ConcatTS(BaseInterface):

    """
    Description:

    Concatenate time series 

    Inputs:

        all_ts_file:
            type = File, exists=True, desc='npy file containing all ts to be concatenated', mandatory=True

    Outputs:

        concatenated_ts_file:
            type = File, exists=True, desc="ts after concatenation"

    Comments:

    Not sure where it is used
    """

    input_spec = ConcatTSInputSpec
    output_spec = ConcatTSOutputSpec

    def _run_interface(self, runtime):

        all_ts_file = self.inputs.all_ts_file

        # loading time series
        all_ts = np.load(all_ts_file)

        print("all_ts: ")
        print(all_ts.shape)

        concatenated_ts = all_ts.swapaxes(1, 0).reshape(all_ts.shape[1], -1)

        print(concatenated_ts.shape)

        # saving time series
        concatenated_ts_file = os.path.abspath("concatenated_ts.npy")
        np.save(concatenated_ts_file, concatenated_ts)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["concatenated_ts_file"] = os.path.abspath(
            "concatenated_ts.npy")

        return outputs

######################################################################################## MergeTS ##################################################################################################################


class MergeTSInputSpec(BaseInterfaceInputSpec):

    all_ts_files = traits.List(File(
        exists=True), desc='list of npy files containing all ts to be merged', mandatory=True)


class MergeTSOutputSpec(TraitedSpec):

    merged_ts_file = File(exists=True, desc="ts after merge")


class MergeTS(BaseInterface):

    """
    Description:

    Merges time series from several files 

    Inputs:

        all_ts_files:
            type = List of Files, exists=True, desc='list of npy files containing all ts to be merged', mandatory=True

    Outputs:

        merged_ts_file: 
            type = File, exists=True, desc="ts after merge"

    Comments:

    Used for multiple-session merges

    """

    input_spec = MergeTSInputSpec
    output_spec = MergeTSOutputSpec

    def _run_interface(self, runtime):

        all_ts_files = self.inputs.all_ts_files

        print(all_ts_files)

        for i, all_ts_file in enumerate(all_ts_files):

            all_ts = np.load(all_ts_file)

            concatenated_ts = all_ts.swapaxes(
                1, 0).reshape(all_ts.shape[1], -1)

            print(concatenated_ts.shape)

            if len(concatenated_ts.shape) > 1:

                if i == 0:
                    merged_ts = concatenated_ts.copy()
                    print(merged_ts.shape)
                else:
                    merged_ts = np.concatenate(
                        (merged_ts, concatenated_ts), axis=1)
                    print(merged_ts.shape)

        merged_ts_file = os.path.abspath("merged_ts.npy")
        np.save(merged_ts_file, merged_ts)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["merged_ts_file"] = os.path.abspath("merged_ts.npy")

        return outputs


######################################################################################## SeparateTS ##################################################################################################################

class SeparateTSInputSpec(BaseInterfaceInputSpec):

    all_ts_file = File(
        exists=True, desc='npy file containing all ts to be concatenated', mandatory=True)


class SeparateTSOutputSpec(TraitedSpec):

    separated_ts_files = traits.List(
        File(exists=True), desc="ts files after separation")


class SeparateTS(BaseInterface):

    """
    Description:

    Save all time series from a npy file to several single time series npy files

    Inputs:

        all_ts_file:
            type = File, exists=True, desc='npy file containing all ts to be concatenated', mandatory=True

    Outputs:

        separated_ts_files
            type = List of Files, exists=True, desc="ts files after separation"


    Comments:

    Not sure where it is used...
    """

    input_spec = SeparateTSInputSpec
    output_spec = SeparateTSOutputSpec

    def _run_interface(self, runtime):

        #import os
        #import numpy as np
        #import nibabel as nib

        #from graphpype.utils_plot import plot_signals

        all_ts_file = self.inputs.all_ts_file

        path, fname_ts, ext = split_f(all_ts_file)

        # loading ts shape = (trigs, electrods, time points)

        all_ts = np.load(all_ts_file)

        print("all_ts: ")
        print(all_ts.shape)

        separated_ts_files = []

        for i in range(all_ts.shape[0]):

            sep_ts_file = os.path.abspath(
                fname_ts + '_trig_' + str(i) + '.npy')

            np.save(sep_ts_file, all_ts[i, :, :])

            separated_ts_files.append(sep_ts_file)

        self.separated_ts_files = separated_ts_files

        return runtime

        # return mean_masked_ts_file,subj_coord_rois_file

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["separated_ts_files"] = self.separated_ts_files

        return outputs

################################################################################# RegressCovar ######################################################################################################################


#from graphpype.utils_cor import regress_movement_wm_csf_parameters
from graphpype.utils_cor import regress_parameters, filter_data, normalize_data
# regress_filter_normalize_parameters


class RegressCovarInputSpec(BaseInterfaceInputSpec):
    masked_ts_file = File(
        exists=True, desc='time series in npy format', mandatory=True)

    rp_file = File(exists=True, desc='Movement parameters', mandatory=False)

    mean_wm_ts_file = File(
        exists=True, desc='White matter signal', mandatory=False)

    mean_csf_ts_file = File(
        exists=True, desc='Cerebro-spinal fluid (ventricules) signal', mandatory=False)

    filtered_normalized = traits.Bool(
        True, usedefault=True, desc="Is the signal filtered and normalized after regression?")

    plot_fig = traits.Bool(True, usedefault=True, desc="Plotting signals?")


class RegressCovarOutputSpec(TraitedSpec):

    resid_ts_file = File(
        exists=True, desc="residuals of time series after regression of all paramters")


class RegressCovar(BaseInterface):
    """
    Description:

    Regress parameters of non-interest (i.e. movement parameters, white matter, csf) from signal.
    Optionnally filter and normalize (z-score) the residuals.

    Inputs:

        masked_ts_file:
            type = File, exists=True, desc='Time series in npy format', mandatory=True

        rp_file: 
            type = File, exists=True, desc='Movement parameters in txt format, SPM style', mandatory=False

        mean_wm_ts_file:
            type = File: exists=True, desc='White matter signal', mandatory=False


        mean_csf_ts_file:   
            type = File, exists=True, desc='Cerebro-spinal fluid (ventricules) signal', mandatory=False

        filtered_normalized:
            type= Bool, default = True, usedefault = True , desc = "Filter and Normalize the signal  after regression?"

    Outputs:

        resid_ts_file:
            type = File, exists=True, desc="residuals of time series after regression of all paramters"


    """
    input_spec = RegressCovarInputSpec
    output_spec = RegressCovarOutputSpec

    def _run_interface(self, runtime):

        print("in regress_covariates")

        print("load masked_ts_file")

        data_mask_matrix = np.loadtxt(self.inputs.masked_ts_file)

        print(data_mask_matrix.shape)

        if isdefined(self.inputs.rp_file):

            print("load rp parameters")

            rp_file = self.inputs.rp_file
            print(rp_file)

            rp = np.genfromtxt(rp_file)

            print(rp.shape)

        else:
            rp = None

        if isdefined(self.inputs.mean_csf_ts_file):

            mean_csf_ts_file = self.inputs.mean_csf_ts_file

            print("load mean_csf_ts_file" + str(mean_csf_ts_file))

            mean_csf_ts = np.loadtxt(mean_csf_ts_file)

            mean_csf_ts = mean_csf_ts.reshape(mean_csf_ts.shape[0], -1)

            print(mean_csf_ts.shape)

        else:
            mean_csf_ts = None

        if isdefined(self.inputs.mean_wm_ts_file):

            print("load mean_wm_ts_file")

            mean_wm_ts = np.loadtxt(self.inputs.mean_wm_ts_file)

            mean_wm_ts = mean_csf_ts.reshape(mean_wm_ts.shape[0], -1)

        else:
            mean_wm_ts = None

        if all([a is None for a in (rp, mean_csf_ts, mean_wm_ts)]):

            resid_data_matrix = data_mask_matrix

        else:
            rp = np.concatenate(
                [a for a in (rp, mean_csf_ts, mean_wm_ts) if a is not None], axis=1)

            # regression movement parameters, return the residuals
            #resid_data_matrix = regress_movement_wm_csf_parameters(data_mask_matrix,rp,mean_wm_ts,mean_csf_ts)
            resid_data_matrix = regress_parameters(data_mask_matrix, rp)

        if self.inputs.filtered_normalized:

            # filtering data
            resid_filt_data_matrix = filter_data(resid_data_matrix)

            # normalizing
            z_score_data_matrix = normalize_data(resid_filt_data_matrix)

            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file, z_score_data_matrix)

            resid_ts_txt_file = os.path.abspath('resid_ts.txt')
            np.savetxt(resid_ts_txt_file, z_score_data_matrix, fmt='%0.3f')

            print("plotting resid_ts")

            if self.inputs.plot_fig:

                plot_resid_ts_file = os.path.abspath('resid_ts.eps')

                plot_sep_signals(plot_resid_ts_file, z_score_data_matrix)

                print("plotting diff filtered and non filtered data")

                plot_diff_filt_ts_file = os.path.abspath('diff_filt_ts.eps')

                plot_signals(plot_diff_filt_ts_file, np.array(
                    resid_filt_data_matrix - resid_data_matrix, dtype='float'))

        else:

            print("Using only regression")

            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file, resid_data_matrix)

            resid_ts_txt_file = os.path.abspath('resid_ts.txt')
            np.savetxt(resid_ts_txt_file, resid_data_matrix, fmt='%0.3f')

            print("plotting resid_ts")

            if self.inputs.plot_fig:

                plot_resid_ts_file = os.path.abspath('resid_ts.eps')

                plot_sep_signals(plot_resid_ts_file, resid_data_matrix)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["resid_ts_file"] = os.path.abspath('resid_ts.npy')

        return outputs

        ################################################################################# FindSPMRegressor ######################################################################################################################


class FindSPMRegressorInputSpec(BaseInterfaceInputSpec):

    spm_mat_file = File(
        exists=True, desc='SPM design matrix after generate model', mandatory=True)

    regressor_name = traits.String(
        exists=True, desc='Name of the regressor in SPM design matrix to be looked after', mandatory=True)

    run_index = traits.Int(
        1, usedefault=True, desc="Run (session) index, default is one in SPM")

    only_positive_values = traits.Bool(
        True, usedefault=True, desc="Return only positive values of the regressor (negative values are set to 0)")

    concatenated_runs = traits.Bool(
        False, usedefault=True, desc="If concatenate runs, need to search for the lenghth of the session")


class FindSPMRegressorOutputSpec(TraitedSpec):

    regressor_file = File(
        exists=True, desc="txt file containing the regressor")


class FindSPMRegressor(BaseInterface):
    """

    Description:

    Find regressor in SPM.mat and save it as timeseries txt file

    Inputs:

        spm_mat_file:
            type = File, exists=True, desc='SPM design matrix after generate model', mandatory=True

        regressor_name:
            type = String, exists=True, desc='Name of the regressor in SPM design matrix to be looked after', mandatory=True

        run_index:
            type = Int, default = 1 , usedefault = True , desc = "Run (session) index, default is one in SPM"

        only_positive_values:
            type = Bool, default = True, usedefault = True , desc = "Return only positive values of the regressor (negative values are set to 0); Otherwise return all values"

        concatenated_runs:  
            type = Bool, default = False , usedefault = True , desc = "If concatenate runs, need to search for the length of the session"

            Deprecation: #concatenate_runs = traits.Int(1, usedefault = True , desc = "If concatenate runs, how many runs there is (needed to return the part of the regressors that is active for the session only)")
    Outputs:

        regressor_file:
            type = File,exists=True, desc="txt file containing the regressor"


    """
    input_spec = FindSPMRegressorInputSpec
    output_spec = FindSPMRegressorOutputSpec

    def _run_interface(self, runtime):

        import scipy.io
        import numpy as np
        import os

        spm_mat_file = self.inputs.spm_mat_file
        regressor_name = self.inputs.regressor_name
        run_index = self.inputs.run_index
        only_positive_values = self.inputs.only_positive_values
        concatenated_runs = self.inputs.concatenated_runs

        print(spm_mat_file)

        # Reading spm.mat for regressors extraction:
        d = scipy.io.loadmat(spm_mat_file)

        # print d

        # Choosing the column according to the regressor name
        #_,col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == u'Sn(1) ' + regressor_name)

        cond_name = 'Sn(' + str(run_index) + ') ' + regressor_name + '*bf(1)'

        print(cond_name)

        _, col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == cond_name)

        print(col)

        # reformating matrix (1,len) in vector (len)
        regressor_vect = d['SPM']['xX'][0][0]['X'][0][0][:, col].reshape(-1)

        print(regressor_vect)

        assert np.sum(
            regressor_vect) != 0, "Error, empty regressor {}".format(cond_name)

        if only_positive_values == True:

            regressor_vect[regressor_vect < 0] = 0

        if concatenated_runs:

            print(run_index)
            print(regressor_vect.shape[0])

            samples = d['SPM']['xX'][0][0]['K'][0][0]['row'][0][run_index-1][0]-1

            print(samples)

            regressor_vect = regressor_vect[samples]

            print(regressor_vect)

            print(regressor_vect.shape)

        print("Saving extract_cond")
        regressor_file = os.path.abspath('extract_cond.txt')

        np.savetxt(regressor_file, regressor_vect)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["regressor_file"] = os.path.abspath('extract_cond.txt')

        return outputs

        ################################################################################# MergeRuns ######################################################################################################################


class MergeRunsInputSpec(BaseInterfaceInputSpec):

    ts_files = traits.List(File(
        exists=True), desc='Numpy files with time series from different runs (sessions)', mandatory=True)

    regressor_files = traits.List(File(
        exists=True), desc='Txt files with regressors from different runs (sessions)', mandatory=True)

    coord_rois_files = traits.List(File(
        exists=True), desc='Txt files with coords from different runs (sessions)', mandatory=True)


class MergeRunsOutputSpec(TraitedSpec):

    ts_all_runs_file = File(
        exists=True, desc="npy file containing the merge ts")

    regressor_all_runs_file = File(
        exists=True, desc="txt file containing the merged regressors")

    coord_rois_all_runs_file = File(
        exists=True, desc="txt file containing the merged coords")


class MergeRuns(BaseInterface):
    """
    Description:

    Merge time series,regressor files and coord files
    Could be done with different cases


    """
    input_spec = MergeRunsInputSpec
    output_spec = MergeRunsOutputSpec

    def _run_interface(self, runtime):

        print('in merge_runs')

        ts_files = self.inputs.ts_files
        regressor_files = self.inputs.regressor_files
        coord_rois_files = self.inputs.coord_rois_files

        if len(ts_files) != len(regressor_files):

            print(
                "Warning, time series and regressors have different length (!= number of runs)")
            return 0

        if len(ts_files) != len(coord_rois_files):

            print(
                "Warning, time series and number of coordinates have different length (!= number of runs)")
            return 0

        # concatenate time series
        for i, ts_file in enumerate(ts_files):

            data_matrix = np.load(ts_file)

            print(data_matrix.shape)

            # loading ROI coordinates
            coord_rois = np.loadtxt(coord_rois_files[i])

            print(coord_rois.shape)

            if i == 0:
                data_matrix_all_runs = np.empty(
                    (data_matrix.shape[0], 0), dtype=data_matrix.dtype)

                coord_rois_all_runs = np.array(coord_rois, dtype='float')

            if coord_rois_all_runs.shape[0] != coord_rois.shape[0]:

                print("ROIs do not match for all different sessions ")

                print(os.getcwd())

                print("Warning, not implemented yet.... ")

                # pris de http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
                # à tester....
                # finir également la partie avec data_matrix_all_runs, en supprimant les colonnes qui ne sont pas communes à tous les runs...

                0/0

                A = coord_rois_all_runs
                B = coord_rois

                nrows, ncols = A.shape
                dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                         'formats': ncols * [A.dtype]}

                C = np.intersect1d(A.view(dtype), B.view(dtype))

                # This last bit is optional if you're okay with "C" being a structured array...
                C = C.view(A.dtype).reshape(-1, ncols)

                coord_rois_all_runs = C

            data_matrix_all_runs = np.concatenate(
                (data_matrix_all_runs, data_matrix), axis=1)

            print(data_matrix_all_runs.shape)

        # save times series for all runs
        ts_all_runs_file = os.path.abspath('ts_all_runs.npy')

        np.save(ts_all_runs_file, data_matrix_all_runs)

        # save coords in common for all runs
        coord_rois_all_runs_file = os.path.abspath('coord_rois_all_runs.txt')

        np.savetxt(coord_rois_all_runs_file, coord_rois_all_runs, fmt='%2.3f')

        # compute regressor for all sessions together (need to sum)

        print("compute regressor for all sessions together (need to sum)")

        regressor_all_runs = np.empty(shape=(0), dtype=float)

        # Sum regressors
        for i, regress_file in enumerate(regressor_files):

            regress_data_vector = np.loadtxt(regress_file)

            if regress_data_vector.shape[0] != 0:

                if regressor_all_runs.shape[0] == 0:

                    regressor_all_runs = regress_data_vector
                else:
                    regressor_all_runs = regressor_all_runs + regress_data_vector

            print(np.sum(regressor_all_runs != 0.0))

        regressor_all_runs_file = os.path.abspath('regressor_all_runs.txt')

        np.savetxt(regressor_all_runs_file, regressor_all_runs, fmt='%0.3f')

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        print(outputs)

        outputs["ts_all_runs_file"] = os.path.abspath('ts_all_runs.npy')

        outputs["coord_rois_all_runs_file"] = os.path.abspath(
            'coord_rois_all_runs.txt')

        outputs["regressor_all_runs_file"] = os.path.abspath(
            'regressor_all_runs.txt')

        return outputs

        ################################################################################# ComputeConfCorMat ######################################################################################################################


from graphpype.utils_cor import return_conf_cor_mat

from graphpype.utils_plot import plot_hist, plot_cormat, plot_ranged_cormat


class ComputeConfCorMatInputSpec(BaseInterfaceInputSpec):

    ts_file = File(
        exists=True, desc='Numpy files with time series to be correlated', mandatory=True)

    transpose_ts = traits.Bool(
        True, usedefault=True, desc='whether to transpose timeseries', mandatory=True)

    weight_file = File(
        exists=True, desc='Weight of the correlation (normally, condition regressor file)', mandatory=False)

    conf_interval_prob = traits.Float(
        0.05, usedefault=True, desc='Confidence interval', mandatory=True)

    plot_mat = traits.Bool(True, usedefault=True,
                           desc='Confidence interval', mandatory=False)

    labels_file = File(
        exists=True, desc='Name of the nodes (used only if plot = true)', mandatory=False)

    method = traits.Enum("Pearson", "Spearman", usedefault=True,
                         desc='Method used for computing correlation (default = Pearson)')


class ComputeConfCorMatOutputSpec(TraitedSpec):

    cor_mat_file = File(
        exists=True, desc="npy file containing the R values of correlation")

    Z_cor_mat_file = File(
        exists=True, desc="npy file containing the Z-values (after Fisher's R-to-Z trasformation) of correlation")

    conf_cor_mat_file = File(
        exists=True, desc="npy file containing the confidence interval around R values")

    Z_conf_cor_mat_file = File(
        exists=True, desc="npy file containing the Z-values (after Fisher's R-to-Z trasformation) of correlation")


class ComputeConfCorMat(BaseInterface):

    """

    Description:

    Compute correlation between time series, with a given confidence interval. If weight_file is specified, used for weighted correlation

    Inputs:

        ts_file:
            type = File, exists=True, desc='Numpy files with time series to be correlated',mandatory=True

        transpose_ts:
            type = Bool, default=True,usedefault = True,desc =  'whether to transpose timeseries', mandatory = True

        weight_file:
            type = File, exists=True, desc='Weight of the correlation (normally, condition regressor file)', mandatory=False

        conf_interval_prob:
            type = Float, default = 0.05, usedefault = True, desc='Confidence interval', mandatory=True

        plot_mat:
            type = Bool, default = True, usedefault = True, desc='Confidence interval', mandatory=False

        labels_file:
            type = File, exists=True, desc='Name of the nodes (used only if plot = true)', mandatory=False

    Outputs:

        cor_mat_file:
            type = File, exists=True, desc="npy file containing the R values of correlation"

        Z_cor_mat_file:
            type = File, exists=True, desc="npy file containing the Z-values (after Fisher's R-to-Z trasformation) of correlation"

        conf_cor_mat_file:
            type = File, exists=True, desc="npy file containing the confidence interval around R values"

    """

    input_spec = ComputeConfCorMatInputSpec
    output_spec = ComputeConfCorMatOutputSpec

    def _run_interface(self, runtime):

        print('in compute_conf_correlation_matrix')

        ts_file = self.inputs.ts_file
        weight_file = self.inputs.weight_file
        transpose_ts = self.inputs.transpose_ts
        conf_interval_prob = self.inputs.conf_interval_prob

        plot_mat = self.inputs.plot_mat
        labels_file = self.inputs.labels_file
        method = self.inputs.method

        print('load resid data')

        path, fname, ext = split_f(ts_file)

        data_matrix = np.load(ts_file)

        print(data_matrix.shape)

        if transpose_ts:
            data_matrix = np.transpose(data_matrix)
            print(data_matrix.shape)

        if isdefined(weight_file):

            print('load weight_vect')

            weight_vect = np.loadtxt(weight_file)

            print(weight_vect.shape)

        else:
            weight_vect = np.ones(shape=(data_matrix.shape[0]))

        print("compute return_Z_cor_mat")

        if method == "Pearson":
            cor_mat, Z_cor_mat, conf_cor_mat, Z_conf_cor_mat = return_conf_cor_mat(
                data_matrix, weight_vect, conf_interval_prob)

        elif method == "Spearman":

            import scipy
            print("Computing Spearman")

            rho_mat, pval_mat = scipy.stats.spearmanr(data_matrix)

            print(rho_mat.shape)

        print(Z_cor_mat.shape)

        cor_mat = cor_mat + np.transpose(cor_mat)

        Z_cor_mat = Z_cor_mat + np.transpose(Z_cor_mat)

        print("saving cor_mat as npy")

        cor_mat_file = os.path.abspath('cor_mat_' + fname + '.npy')

        # np.save(cor_mat_file,non_nan_cor_mat)
        np.save(cor_mat_file, cor_mat)

        print("saving conf_cor_mat as npy")

        conf_cor_mat_file = os.path.abspath('conf_cor_mat_' + fname + '.npy')

        # np.save(conf_cor_mat_file,non_nan_conf_cor_mat)
        np.save(conf_cor_mat_file, conf_cor_mat)

        print("saving Z_cor_mat as npy")

        Z_cor_mat_file = os.path.abspath('Z_cor_mat_' + fname + '.npy')

        # np.save(Z_cor_mat_file,non_nan_Z_cor_mat)
        np.save(Z_cor_mat_file, Z_cor_mat)

        print("saving Z_conf_cor_mat as npy")

        Z_conf_cor_mat_file = os.path.abspath(
            'Z_conf_cor_mat_' + fname + '.npy')

        Z_conf_cor_mat = Z_conf_cor_mat + np.transpose(Z_conf_cor_mat)

        np.save(Z_conf_cor_mat_file, Z_conf_cor_mat)

        if plot_mat:

            if isdefined(labels_file):

                print('extracting node labels')

                labels = [line.strip() for line in open(labels_file)]
                print(labels)

            else:
                labels = []

            # cor_mat

            # heatmap

            print('plotting cor_mat heatmap')

            plot_heatmap_cor_mat_file = os.path.abspath(
                'heatmap_cor_mat_' + fname + '.eps')

            print(plot_heatmap_cor_mat_file)
            plot_cormat(plot_heatmap_cor_mat_file, cor_mat, list_labels=labels)

            # histogram

            print('plotting cor_mat histogram')

            plot_hist_cor_mat_file = os.path.abspath(
                'hist_cor_mat_' + fname + '.eps')

            plot_hist(plot_hist_cor_mat_file, cor_mat, nb_bins=100)

            # Z_cor_mat

            # heatmap

            print('plotting Z_cor_mat heatmap')

            plot_heatmap_Z_cor_mat_file = os.path.abspath(
                'heatmap_Z_cor_mat_' + fname + '.eps')

            plot_cormat(plot_heatmap_Z_cor_mat_file,
                        Z_cor_mat, list_labels=labels)

            # heatmap

            # print 'plotting cor_mat heatmap'

            #plot_heatmap_ranged_Z_cor_mat_file =  os.path.abspath('heatmap_ranged_cor_mat_' + fname + '.eps')

            #plot_ranged_cormat(plot_heatmap_ranged_Z_cor_mat_file,Z_cor_mat,fix_full_range = [0.0,3.5],list_labels = labels)

            # histogram

            print('plotting Z_cor_mat histogram')

            plot_hist_Z_cor_mat_file = os.path.abspath(
                'hist_Z_cor_mat_' + fname + '.eps')

            plot_hist(plot_hist_Z_cor_mat_file, Z_cor_mat, nb_bins=100)

            # conf_cor_mat

            # heatmap

            print('plotting conf_cor_mat heatmap')

            plot_heatmap_conf_cor_mat_file = os.path.abspath(
                'heatmap_conf_cor_mat_' + fname + '.eps')

            plot_cormat(plot_heatmap_conf_cor_mat_file,
                        conf_cor_mat, list_labels=labels)

            # histogram

            #print('plotting conf_cor_mat histogram')

            #plot_hist_conf_cor_mat_file = os.path.abspath('hist_conf_cor_mat_' + fname + '.eps')

            #plot_hist(plot_hist_conf_cor_mat_file,conf_cor_mat,nb_bins = 100)

            # Z_conf_cor_mat

            #Z_conf_cor_mat = np.load(Z_conf_cor_mat_file)

            # heatmap

            print('plotting Z_conf_cor_mat heatmap')

            plot_heatmap_Z_conf_cor_mat_file = os.path.abspath(
                'heatmap_Z_conf_cor_mat_' + fname + '.eps')

            plot_cormat(plot_heatmap_Z_conf_cor_mat_file,
                        Z_conf_cor_mat, list_labels=labels)

            # histogram

            # print 'plotting Z_conf_cor_mat histogram'

            #plot_hist_Z_conf_cor_mat_file = os.path.abspath('hist_Z_conf_cor_mat_' + fname + '.eps')

            #plot_hist(plot_hist_Z_conf_cor_mat_file,Z_conf_cor_mat,nb_bins = 100)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        path, fname, ext = split_f(self.inputs.ts_file)

        outputs["cor_mat_file"] = os.path.abspath('cor_mat_' + fname + '.npy')

        outputs["conf_cor_mat_file"] = os.path.abspath(
            'conf_cor_mat_' + fname + '.npy')

        outputs["Z_cor_mat_file"] = os.path.abspath(
            'Z_cor_mat_' + fname + '.npy')

        outputs["Z_conf_cor_mat_file"] = os.path.abspath(
            'Z_conf_cor_mat_' + fname + '.npy')

        print(outputs)

        return outputs

        ################################################################################# ComputeSpearmanMat ######################################################################################################################


from graphpype.utils_cor import return_conf_cor_mat

from graphpype.utils_plot import plot_hist, plot_cormat, plot_ranged_cormat


class ComputeSpearmanMatInputSpec(BaseInterfaceInputSpec):

    ts_file = File(
        exists=True, desc='Numpy files with time series to be correlated', mandatory=True)

    transpose_ts = traits.Bool(
        True, usedefault=True, desc='whether to transpose timeseries', mandatory=True)

    plot_mat = traits.Bool(True, usedefault=True,
                           desc='Using matplotlib to plot', mandatory=False)

    labels_file = File(
        exists=True, desc='Name of the nodes (used only if plot = true)', mandatory=False)

    method = traits.Enum(
        "Pearson", "Spearman", desc='Method used for computing correlation (default = Pearson)')

    export_csv = traits.Bool(True, usedefault=True,
                             desc='save as CSV as well', mandatory=False)


class ComputeSpearmanMatOutputSpec(TraitedSpec):

    rho_mat_file = File(
        exists=True, desc="npy file containing the rho values of correlation")

    pval_mat_file = File(exists=True, desc="npy file containing the p-values")


class ComputeSpearmanMat(BaseInterface):

    """

    Description:

    Compute correlation between time series, with a given confidence interval. If weight_file is specified, used for weighted correlation

    Inputs:

        ts_file:
            type = File, exists=True, desc='Numpy files with time series to be correlated',mandatory=True

        transpose_ts:
            type = Bool, default=True,usedefault = True,desc =  'whether to transpose timeseries', mandatory = True

        plot_mat:
            type = Bool, default = True, usedefault = True, desc='Confidence interval', mandatory=False

        labels_file:
            type = File, exists=True, desc='Name of the nodes (used only if plot = true)', mandatory=False

    Outputs:

        rho_mat_file :
            type = File, exists=True, desc="npy file containing the rho values of correlation"

        pval_mat_file:
            type = File, exists=True, desc="npy file containing the p-values"

    """

    input_spec = ComputeSpearmanMatInputSpec
    output_spec = ComputeSpearmanMatOutputSpec

    def _run_interface(self, runtime):

        print('in compute_conf_correlation_matrix')

        ts_file = self.inputs.ts_file
        transpose_ts = self.inputs.transpose_ts

        plot_mat = self.inputs.plot_mat
        export_csv = self.inputs.export_csv

        labels_file = self.inputs.labels_file

        print('load resid data')

        path, fname, ext = split_f(ts_file)

        data_matrix = np.load(ts_file)

        print(data_matrix.shape)

        if transpose_ts:

            data_matrix = np.transpose(data_matrix)
            print(data_matrix.shape)

        import scipy
        print("Computing Spearman")

        rho_mat, pval_mat = scipy.stats.spearmanr(data_matrix)

        print(rho_mat.shape)

        np.fill_diagonal(rho_mat, 0)

        # 0/0

        # print(Z_cor_mat.shape)

        #cor_mat = cor_mat + np.transpose(cor_mat)

        #Z_cor_mat = Z_cor_mat + np.transpose(Z_cor_mat)

        print("saving rho_mat as npy")

        rho_mat_file = os.path.abspath('rho_mat_' + fname + '.npy')

        # np.save(rho_mat_file,non_nan_rho_mat)
        np.save(rho_mat_file, rho_mat)

        print("saving pval_mat as npy")

        pval_mat_file = os.path.abspath('pval_mat_' + fname + '.npy')

        # np.save(pval_mat_file,non_nan_pval_mat)
        np.save(pval_mat_file, pval_mat)

        if isdefined(labels_file):

            print('extracting node labels')

            labels = [line.strip() for line in open(labels_file)]
            print(labels)

        else:
            labels = []

        if plot_mat:

            # rho_mat

            # heatmap

            print('plotting rho_mat heatmap')

            plot_heatmap_rho_mat_file = os.path.abspath(
                'heatmap_rho_mat_' + fname + '.eps')

            plot_cormat(plot_heatmap_rho_mat_file, rho_mat, list_labels=labels)

        if export_csv:

            import pandas as pd

            if len(labels) == rho_mat.shape[0] and len(labels) == rho_mat.shape[1]:
                df_rho = pd.DataFrame(rho_mat, columns=labels, index=labels)
                df_pval = pd.DataFrame(pval_mat, columns=labels, index=labels)
            else:

                df_rho = pd.DataFrame(rho_mat)
                df_pval = pd.DataFrame(pval_mat)

            df_rho.to_csv(os.path.abspath('rho_mat.csv'))
            
            df_pval.to_csv(os.path.abspath('pval_mat.csv'))

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        path, fname, ext = split_f(self.inputs.ts_file)

        outputs["rho_mat_file"] = os.path.abspath('rho_mat_' + fname + '.npy')

        outputs["pval_mat_file"] = os.path.abspath(
            'pval_mat_' + fname + '.npy')

        print(outputs)

        return outputs

        ################################################################################# SelectNonNAN ######################################################################################################################


class SelectNonNANInputSpec(BaseInterfaceInputSpec):

    sess_ts_files = traits.List(File(
        exists=True), desc='Numpy files with time series to be correlated', mandatory=True)

    sess_labels_files = traits.List(File(
        exists=True), desc='Name of the nodes (used only if plot = true)', mandatory=False)


class SelectNonNANOutputSpec(TraitedSpec):

    select_ts_files = traits.List(
        File(exists=True), desc='Numpy files with selected time series ', mandatory=True)

    select_labels_file = File(
        exists=True, desc="npy file containing the Z-values (after Fisher's R-to-Z trasformation) of correlation")


class SelectNonNAN(BaseInterface):

    """
    Select time series based on NaN
    TODO: finally no used
    """

    input_spec = SelectNonNANInputSpec
    output_spec = SelectNonNANOutputSpec

    def _run_interface(self, runtime):

        print('in compute_conf_correlation_matrix')

        sess_ts_files = self.inputs.sess_ts_files
        labels_files = self.inputs.sess_labels_files

        if len(sess_ts_files) == 0:

            print("Warning, could not find sess_ts_files")

            return runtime

        path, fname, ext = split_f(sess_ts_files[0])

        list_sess_ts = []

        for ts_file in sess_ts_files:

            print('load data')

            data_matrix = np.load(ts_file)

            print(data_matrix.shape)

            list_sess_ts.append(data_matrix)

        subj_ts = np.concatenate(tuple(list_sess_ts), axis=0)

        print(subj_ts.shape)

        print(np.sum(np.isnan(subj_ts) == True, axis=(1, 2)))
        print(np.sum(np.isnan(subj_ts) == True, axis=(0, 2)))
        print(np.sum(np.isnan(subj_ts) == True, axis=(0, 1)))

        good_trigs = np.sum(np.isnan(subj_ts) == True, axis=(1, 2)) == 0

        select_subj_ts = subj_ts[good_trigs, :, :]

        print(select_subj_ts.shape)

        self.select_ts_files = []

        for i_trig in range(select_subj_ts.shape[0]):

            select_ts_file = os.path.abspath(
                'select_' + fname + '_' + str(i_trig) + '.npy')

            np.save(select_ts_file, select_subj_ts[i_trig, :, :])

            self.select_ts_files.append(select_ts_file)

        # check if all labels_files are identical

        if len(labels_files) == 0:

            print("Warning, could not find sess_ts_files")

            return runtime

        select_labels_file = labels_files[0]

        select_labels = np.array(np.loadtxt(select_labels_file), dtype='str')

        print(select_labels)
        0/0
        labels_files

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["select_ts_files"] = self.select_ts_files

        print(outputs)

        return outputs

############################################# used in run_mean_correl ###############################################################################

##################################################### PrepareMeanCorrel ###############################################################################


from graphpype.utils_cor import return_corres_correl_mat, return_corres_correl_mat_labels


class PrepareMeanCorrelInputSpec(BaseInterfaceInputSpec):

    # print "in PrepareMeanCorrelInputSpec"

    #gm_mask_coords_file = File(exists=True, desc='reference coordinates',mandatory=True)

    cor_mat_files = traits.List(File(
        exists=True), desc='Numpy files with correlation matrices gm_mask_coords', mandatory=True)

    #coords_files = traits.List(File(exists=True), desc='Txt files with coordinates (corresponding to the space also described in ', mandatory=True)

    #labels_file = File(exists=True, desc='reference labels',mandatory=False)

    coords_files = traits.List(File(
        exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True, xor=['labels_files'])

    labels_files = traits.List(File(
        exists=True), desc='list of labels (in txt format) for each subject (after removal of non void data)', mandatory=True, xor=['coords_files'])

    gm_mask_coords_file = File(
        exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=False)

    gm_mask_labels_file = File(
        exists=True, desc='Labels for all possible nodes - in case coords are varying from one indiv to the other (source space for example)', mandatory=False)

    plot_mat = traits.Bool(True, usedefault=True, mandatory=False)

    export_csv = traits.Bool(False, usedefault=True, mandatory=False)


class PrepareMeanCorrelOutputSpec(TraitedSpec):

    # print "in PrepareMeanCorrelOutputSpec"

    group_cor_mat_matrix_file = File(
        exists=True, desc="npy file containing all correlation matrices in 3D")

    sum_cor_mat_matrix_file = File(
        exists=True, desc="npy file containing the sum of all correlation matrices")

    sum_possible_edge_matrix_file = File(
        exists=True, desc="npy file containing the number of correlation matrices where both nodes where actually defined in the mask")

    avg_cor_mat_matrix_file = File(
        exists=True, desc="npy file containing the average of all correlation matrices")


class PrepareMeanCorrel(BaseInterface):

    import numpy as np
    import os

    #import nibabel as nib

    """
    Decription:
    
    Return average of correlation values within the same common space (defined in gm_mask_coords), only when the nodes are defined for a given values 
    
    Input:
    
    gm_mask_coords_file
        type = File, exists=True, desc='reference coordinates',mandatory=True
                
    cor_mat_files
        type = List of Files, exists=True, desc='Numpy files with correlation matrices gm_mask_coords',mandatory=True

    coords_files:
        type = List of Files, exists=True, desc='Txt files with coordinates (corresponding to the space also described in ', mandatory=True
    
    gm_mask_labels_file:
        type = File, exists=True, desc='reference labels',mandatory=False
    
    plot_mat:
        type = Bool; default = True, usedefault = True, mandatory = False
        
        
    Outputs: 
    
    group_cor_mat_matrix_file: 
        type = File,exists=True, desc="npy file containing all correlation matrices in 3D"
    
    sum_cor_mat_matrix_file
        type = File,exists=True, desc="npy file containing the sum of all correlation matrices"
    
    sum_possible_edge_matrix_file:
        type = File, exists=True, desc="npy file containing the number of correlation matrices where both nodes where actually defined in the mask"
    
    avg_cor_mat_matrix_file:
        type = File, exists=True, desc="npy file containing the average of all correlation matrices"
    
    """

    input_spec = PrepareMeanCorrelInputSpec
    output_spec = PrepareMeanCorrelOutputSpec

    def _run_interface(self, runtime):

        import pandas as pd

        cor_mat_files = self.inputs.cor_mat_files
        gm_mask_labels_file = self.inputs.gm_mask_labels_file
        plot_mat = self.inputs.plot_mat
        export_csv = self.inputs.export_csv

        if isdefined(gm_mask_labels_file):

            print('extracting node labels')

            labels = [line.strip() for line in open(gm_mask_labels_file)]
            print(labels)

        else:
            labels = []

        if isdefined(self.inputs.gm_mask_coords_file) and isdefined(self.inputs.coords_files):

            coords_files = self.inputs.coords_files

            gm_mask_coords_file = self.inputs.gm_mask_coords_file

            print('loading gm mask corres')

            gm_mask_coords = np.array(
                np.loadtxt(gm_mask_coords_file), dtype=int)

            print(gm_mask_coords.shape)

            sum_cor_mat_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=float)
            print(sum_cor_mat_matrix.shape)

            sum_possible_edge_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)
            print(sum_possible_edge_matrix.shape)

            group_cor_mat_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0], len(cor_mat_files)), dtype=float)
            print(group_cor_mat_matrix.shape)

            if len(cor_mat_files) != len(coords_files):
                print("warning, length of cor_mat_files, coords_files are imcompatible {} {} {}".format(
                    len(cor_mat_files), len(coords_files)))

            for index_file in range(len(cor_mat_files)):

                print(cor_mat_files[index_file])

                if os.path.exists(cor_mat_files[index_file]) and os.path.exists(coords_files[index_file]):

                    Z_cor_mat = np.load(cor_mat_files[index_file])
                    print(Z_cor_mat.shape)

                    coords = np.array(np.loadtxt(
                        coords_files[index_file]), dtype=int)

                    print(coords)
                    print(coords.shape)

                    print(Z_cor_mat)

                    corres_cor_mat, possible_edge_mat = return_corres_correl_mat(
                        Z_cor_mat, coords, gm_mask_coords)

                    print(corres_cor_mat)
                    print(possible_edge_mat)

                    np.fill_diagonal(corres_cor_mat, 0)

                    np.fill_diagonal(possible_edge_mat, 1)

                    sum_cor_mat_matrix += corres_cor_mat

                    sum_possible_edge_matrix += possible_edge_mat

                    group_cor_mat_matrix[:, :, index_file] = corres_cor_mat

                else:
                    print("Warning, one or more files between " +
                          cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists")

        elif isdefined(self.inputs.gm_mask_labels_file) and isdefined(self.inputs.labels_files):

            labels_files = self.inputs.labels_files

            gm_mask_labels_file = self.inputs.gm_mask_labels_file

            print('loading gm mask labels')

            gm_mask_labels = [line.strip()
                              for line in open(gm_mask_labels_file)]

            gm_size = len(gm_mask_labels)

            
            sum_cor_mat_matrix = np.zeros((gm_size, gm_size), dtype=float)
            print(sum_cor_mat_matrix.shape)

            sum_possible_edge_matrix = np.zeros((gm_size, gm_size), dtype=int)
            print(sum_possible_edge_matrix.shape)

            group_cor_mat_matrix = np.zeros(
                (gm_size, gm_size, len(cor_mat_files)), dtype=float)
            print(group_cor_mat_matrix.shape)

            if len(cor_mat_files) != len(labels_files):
                print("warning, length of cor_mat_files, labels_files are imcompatible {} {} {}".format(
                    len(cor_mat_files), len(labels_files)))

            for index_file in range(len(cor_mat_files)):

                print(cor_mat_files[index_file])

                if os.path.exists(cor_mat_files[index_file]) and os.path.exists(labels_files[index_file]):

                    Z_cor_mat = np.load(cor_mat_files[index_file])
                    print(Z_cor_mat.shape)

                    labels = [line.strip() for line in open(labels_files[i])]
                    print(labels)

                    corres_cor_mat, possible_edge_mat = return_corres_correl_mat_labels(
                        Z_cor_mat, labels, gm_mask_labels)

                    np.fill_diagonal(corres_cor_mat, 0)

                    np.fill_diagonal(possible_edge_mat, 1)

                    sum_cor_mat_matrix += corres_cor_mat

                    sum_possible_edge_matrix += possible_edge_mat

                    group_cor_mat_matrix[:, :, index_file] = corres_cor_mat

                else:
                    print("Warning, one or more files between " +
                          cor_mat_files[index_file] + ', ' + labels_files[index_file] + " do not exists")

        self.group_cor_mat_matrix_file = os.path.abspath(
            'group_cor_mat_matrix.npy')

        np.save(self.group_cor_mat_matrix_file, group_cor_mat_matrix)

        print('saving sum cor_mat matrix')

        self.sum_cor_mat_matrix_file = os.path.abspath(
            'sum_cor_mat_matrix.npy')

        np.save(self.sum_cor_mat_matrix_file, sum_cor_mat_matrix)

        print('saving sum_possible_edge matrix')

        self.sum_possible_edge_matrix_file = os.path.abspath(
            'sum_possible_edge_matrix.npy')

        np.save(self.sum_possible_edge_matrix_file, sum_possible_edge_matrix)

        print('saving avg_cor_mat_matrix')

        self.avg_cor_mat_matrix_file = os.path.abspath(
            'avg_cor_mat_matrix.npy')

        #avg_cor_mat_matrix = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = float)

        if np.sum(np.array(sum_possible_edge_matrix == 0)) == 0:

            avg_cor_mat_matrix = np.divide(np.array(sum_cor_mat_matrix, dtype=float), np.array(
                sum_possible_edge_matrix, dtype=float))

            avg_cor_mat_matrix[np.isnan(avg_cor_mat_matrix)] = 0.0

            print(np.amin(avg_cor_mat_matrix), np.amax(avg_cor_mat_matrix))

            np.save(self.avg_cor_mat_matrix_file, avg_cor_mat_matrix)

            if export_csv:

                csv_avg_cor_mat_matrix_file = os.path.abspath(
                    'avg_cor_mat_matrix.csv')

                df = pd.DataFrame(avg_cor_mat_matrix,
                                  index=labels, columns=labels)

                df.to_csv(csv_avg_cor_mat_matrix_file)

        else:

            # print "!!!!!!!!!!!!!!!!!!!!!!Breaking!!!!!!!!!!!!!!!!!!!!!!!!, found 0 elements in sum_cor_mat_matrix"

            # print sum_possible_edge_matrix
            # print np.array(sum_possible_edge_matrix == 0)
            # print np.array(sum_possible_edge_matrix == 0)
            # print np.sum(np.array(sum_possible_edge_matrix == 0))

            # print np.unique(sum_possible_edge_matrix)

            # return

            avg_cor_mat_matrix = np.divide(np.array(sum_cor_mat_matrix, dtype=float), np.array(
                sum_possible_edge_matrix, dtype=float))

            avg_cor_mat_matrix[np.isnan(avg_cor_mat_matrix)] = 0.0

            print(np.amin(avg_cor_mat_matrix), np.amax(avg_cor_mat_matrix))

            np.save(self.avg_cor_mat_matrix_file, avg_cor_mat_matrix)

            if export_csv:

                csv_avg_cor_mat_matrix_file = os.path.abspath(
                    'avg_cor_mat_matrix.csv')

                df = pd.DataFrame(avg_cor_mat_matrix,
                                  index=labels, columns=labels)

                df.to_csv(csv_avg_cor_mat_matrix_file)

        if plot_mat == True:

            # heatmap

            print('plotting Z_cor_mat heatmap')

            plot_heatmap_avg_cor_mat_file = os.path.abspath(
                'heatmap_avg_cor_mat.eps')

            plot_cormat(plot_heatmap_avg_cor_mat_file,
                        avg_cor_mat_matrix, list_labels=labels)

        return runtime
    
    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["group_cor_mat_matrix_file"] = self.group_cor_mat_matrix_file
        outputs["sum_cor_mat_matrix_file"] = self.sum_cor_mat_matrix_file
        outputs["sum_possible_edge_matrix_file"] = self.sum_possible_edge_matrix_file
        outputs["avg_cor_mat_matrix_file"] = self.avg_cor_mat_matrix_file

        # print outputs

        return outputs


##################################################### PreparePermutMeanCorrel ###############################################################################

class PreparePermutMeanCorrelInputSpec(BaseInterfaceInputSpec):

    cor_mat_files = traits.List(File(
        exists=True), desc='Numpy files with correlation matrices gm_mask_coords', mandatory=True)

    permut_group_sizes = traits.List(
        traits.Int, desc='How to split the groups after shuffling', mandatory=True)

    seed = traits.Int(0, usedefault=True, decs='Start of random process')

    #variance_adjst = traits.Bool(False, usedefault = True, desc = "is between-subject variance adjusted taken into account?")


class PreparePermutMeanCorrelOutputSpec(TraitedSpec):

    permut_mean_cormat_files = traits.List(File(
        exists=True), desc="npy files containing the average of permuted correlation matrices")


class PreparePermutMeanCorrel(BaseInterface):

    import numpy as np
    import os

    #import nibabel as nib

    """
    Return average of correlation values after shuffling orig datasets
    """

    input_spec = PreparePermutMeanCorrelInputSpec
    output_spec = PreparePermutMeanCorrelOutputSpec

    def _run_interface(self, runtime):

        print(self.inputs.seed)

        np.random.seed(self.inputs.seed)

        cormats = [np.load(cor_mat_file)
                   for cor_mat_file in self.inputs.cor_mat_files]

        print(cormats)

        assert len(cormats) == sum(self.inputs.permut_group_sizes), "Warning, len(cormats) {0} != sum permut_group_sizes {1}".format(
            len(cormats), sum(self.inputs.permut_group_sizes))

        subj_indexes = np.arange(len(cormats))

        np.random.shuffle(subj_indexes)

        print(subj_indexes)

        subj_indexes_file = os.path.abspath("subj_indexes.txt")

        f = open(subj_indexes_file, "w+")

        # f.write(subj_indexes.tolist())

        np.savetxt(f, subj_indexes, fmt="%d")

        min_index = 0

        cormats = np.array(cormats)

        print(cormats.shape)

        self.permut_mean_cormat_files = []

        for i, cur_nb_subj_by_gender in enumerate(self.inputs.permut_group_sizes):

            print(cur_nb_subj_by_gender)

            cur_range = np.arange(min_index, min_index+cur_nb_subj_by_gender)

            print(cur_range)

            rand_indexes = subj_indexes[cur_range]

            print(rand_indexes)

            # f.write(rand_indexes)

            np.savetxt(f, rand_indexes, fmt="%d")

            permut_cormats = cormats[rand_indexes, :, :]

            permut_mean_cormat = np.mean(permut_cormats, axis=0)

            print(permut_mean_cormat.shape)

            permut_mean_cormat_file = os.path.abspath(
                "permut_mean_cormat_" + str(i) + ".npy")

            np.save(permut_mean_cormat_file, permut_mean_cormat)

            self.permut_mean_cormat_files.append(permut_mean_cormat_file)

            min_index += cur_nb_subj_by_gender
        f.close()

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["permut_mean_cormat_files"] = self.permut_mean_cormat_files

        return outputs
