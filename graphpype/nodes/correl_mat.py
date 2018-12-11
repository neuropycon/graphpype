"""
Definition of Nodes for computing correlation matrices and handling time series
"""

import scipy

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined

from nipype.utils.filemanip import split_filename as split_f

import numpy as np
import os

import nibabel as nib
import pandas as pd

from graphpype.utils_plot import (plot_signals, plot_sep_signals, plot_hist,
                                  plot_cormat)


from graphpype.utils_cor import (return_corres_correl_mat,
                                 return_corres_correl_mat_labels,
                                 return_conf_cor_mat, regress_parameters,
                                 filter_data, normalize_data,
                                 mean_select_mask_data,
                                 mean_select_indexed_mask_data)


from graphpype.utils import check_np_dimension


# ExtractTS
class ExtractTSInputSpec(BaseInterfaceInputSpec):
    indexed_rois_file = File(
        exists=True, desc='indexed mask where all voxels belonging to the same\
            ROI have the same value (! starting from 1)', mandatory=True)

    file_4D = File(
        exists=True, desc='4D volume to be extracted', mandatory=True)

    MNI_coord_rois_file = File(desc='ROI MNI_coordinates')

    coord_rois_file = File(desc='ROI coordinates')

    label_rois_file = File(desc='ROI labels')

    min_BOLD_intensity = traits.Float(
        50.0,
        desc='BOLD signal below this value will be set to zero',
        usedefault=True)

    percent_signal = traits.Float(
        0.5, desc="Percent of voxels in a ROI with signal higher that \
        min_BOLD_intensity to keep this ROI", usedefault=True)

    plot_fig = traits.Bool(
        False, desc="Plotting mean signal or not", usedefault=True)

    background_val = traits.Float(
        -1.0, desc='value for background (i.e. outside brain)',
        usedefault=True)


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

    Description: Extract time series from a labelled mask in Nifti Format
    where all ROIs have the same index

    Inputs:

        indexed_rois_file:
            type = File, exists=True, desc='indexed mask where all voxels
            belonging to the same ROI have the same value,
            mandatory=True

        file_4D:
            type = File, exists=True, desc='4D volume to be extracted',
            mandatory=True

        MNI_coord_rois_file:
            typr = File, desc='ROI MNI_coordinates'

        coord_rois_file:
            type = File, desc='ROI coordinates'

        label_rois_file:
            type = File, desc='ROI labels')

        min_BOLD_intensity:
            type = Float, default = 50.0,
            desc='BOLD signal below this value will be set to zero',
            usedefault = True

        percent_signal:
            type = Float, default = 0.5,
            desc  = "Percent of voxels in a ROI with signal higher that
            min_BOLD_intensity to keep this ROI",
            usedefault = True

        plot_fig:
            type = Bool, defaults = False,
            desc = "Plotting mean signal or not",
            usedefault = True)

        background_val:
            type = Float, -1.0,
            desc='value for background (i.e. outside brain)',
            usedefault = True

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

        mean_masked_ts, keep_rois = mean_select_indexed_mask_data(
            orig_ts, indexed_mask_rois_data, min_BOLD_intensity,
            percent_signal=percent_signal, background_val=background_val)

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

        if plot_fig:
            # plotting mean_masked_ts
            plot_mean_masked_ts_file = os.path.abspath('mean_masked_ts.eps')
            plot_signals(plot_mean_masked_ts_file, mean_masked_ts)

        return runtime

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


# IntersectMask
class IntersectMaskInputSpec(BaseInterfaceInputSpec):

    indexed_rois_file = File(
        exists=True,
        desc='nii file with indexed mask where all voxels belonging to the\
            same ROI have the same value (! starting from 0)',
        mandatory=True)

    filter_mask_file = File(
        exists=True,
        desc='nii file with (binary) mask - e.g. grey matter mask',
        mandatory=True)

    coords_rois_file = File(desc='ijk coords txt file')

    labels_rois_file = File(desc='labels txt file')

    MNI_coords_rois_file = File(desc='MNI coords txt file')

    filter_thr = traits.Float(0.99, usedefault=True,
                              desc='Value to threshold filter_mask')

    background_val = traits.Float(
        -1.0, desc='value for background (i.e. outside brain)',
        usedefault=True)


class IntersectMaskOutputSpec(TraitedSpec):

    filtered_indexed_rois_file = File(
        exists=True,
        desc=('nii file with indexed mask where all voxels belonging to the\
              same ROI have the same value'))

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
    Optionnally, keep only ijk_coords,
    MNI_coords and labels that are kept in filtered mask

    Inputs:

        indexed_rois_file:
            type = File, exists=True,
            desc='nii file with indexed mask where all voxels belonging to the
            same ROI have the same value (! starting from 0)',
            mandatory=True

        filter_mask_file:
            type = File, exists=True,
            desc='nii file with (binary) mask - e.g. grey matter mask',
            mandatory=True

        coords_rois_file:
            type = File, desc='ijk coords txt file'

        labels_rois_file:
            type = File, desc='labels txt file'

        MNI_coords_rois_file:
            type = File, desc='MNI coords txt file'

        filter_thr:
            type = Float, default = 0.99, usedefault = True,
            desc='Value to threshold filter_mask'

        background_val:
            type = Float, -1.0,
            desc='value for background (i.e. outside brain)',
            usedefault = True

    Outputs:

        filtered_indexed_rois_file:
            type = File, exists=True,
            desc='nii file with indexed mask where all voxels belonging to the
            same ROI have the same value (! starting from 0)'

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
        indexed_rois_data[np.isnan(indexed_rois_data)] = background_val

        # loading time series
        filter_mask_data = nib.load(filter_mask_file).get_data()

        assert filter_mask_data.shape == indexed_rois_data.shape, \
            ("error, filter_mask {} and indexed_rois {} should have the \
                same shape".format(filter_mask_data.shape,
                                   indexed_rois_data.shape))

        filter_mask_data[filter_mask_data > filter_thr] = 1.0
        filter_mask_data[filter_mask_data <= filter_thr] = 0.0

        # keep_rois_data
        if background_val == -1.0:

            val = filter_mask_data*(indexed_rois_data.copy()+1) - 1
            keep_rois_data = np.array(val, dtype='int64')

        elif background_val == 0.0:
            keep_rois_data = np.array(filter_mask_data * indexed_rois_data,
                                      dtype='int64')

        # reorder_indexed_rois (starting from -1 (background) and raising by 1
        # for all available ROI
        reorder_indexed_rois_data = np.zeros(
            shape=keep_rois_data.shape, dtype='int64') - 1

        for i, index in enumerate(np.unique(keep_rois_data)[1:]):
            assert np.sum(np.array(keep_rois_data == index, dtype=int)), \
                ("Error, could not find value {} in \
                 keep_rois_data".format(index))
            reorder_indexed_rois_data[keep_rois_data == index] = i

        nib.save(nib.Nifti1Image(
            reorder_indexed_rois_data,
            indexed_rois_img.get_affine(),
            indexed_rois_img.get_header()),
            os.path.abspath("reorder_filtered_indexed_rois.nii"))

        # index_corres
        if background_val == -1.0:
            index_corres = np.unique(keep_rois_data)[1:]

        elif background_val == 0.0:
            index_corres = np.unique(keep_rois_data)[1:]-1

        # if ROI coordinates
        if isdefined(coords_rois_file):
            coords_rois = np.loadtxt(coords_rois_file)
            filtered_coords_rois = coords_rois[index_corres, :]
            filtered_coords_rois_file = os.path.abspath(
                "filtered_coords_rois.txt")
            np.savetxt(filtered_coords_rois_file,
                       filtered_coords_rois, fmt="%d")

        # if ROI MNI coordinates
        if isdefined(MNI_coords_rois_file):
            MNI_coords_rois = np.loadtxt(MNI_coords_rois_file)
            filtered_MNI_coords_rois = MNI_coords_rois[index_corres, :]
            filtered_MNI_coords_rois_file = os.path.abspath(
                "filtered_MNI_coords_rois.txt")
            np.savetxt(filtered_MNI_coords_rois_file,
                       filtered_MNI_coords_rois, fmt="%f")

        # if ROI labels
        if isdefined(labels_rois_file):
            np_labels_rois = np.array(
                [line.strip() for line in open(labels_rois_file)], dtype='str')
            filtered_labels_rois = np_labels_rois[index_corres]
            filtered_labels_rois_file = os.path.abspath(
                "filtered_labels_rois.txt")
            np.savetxt(filtered_labels_rois_file,
                       filtered_labels_rois, fmt="%s")

        return runtime

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


# ExtractMeanTS
class ExtractMeanTSInputSpec(BaseInterfaceInputSpec):

    file_4D = File(
        exists=True, desc='4D volume to be extracted', mandatory=True)

    ROI_coord = traits.List(
        traits.Int(exists=True),
        desc='values for extracting ROI',
        mandatory=True, xor=['mask_file', 'filter_mask_file'])

    mask_file = File(
        xor=['filter_mask_file', 'ROI_coord'], exists=True,
        desc='mask file where all voxels belonging to the selected region have\
            index 1', mandatory=True)

    filter_mask_file = File(
        xor=['mask_file', 'ROI_coord'], requires=['filter_thr'], exists=True,
        desc='mask file where all voxels belonging to the selected region \
            have values higher than threshold', mandatory=True)

    filter_thr = traits.Float(0.99, usedefault=True,
                              desc='Value to threshold filter_mask')

    suffix = traits.String(
        "suf", desc='Suffix added to describe the extracted time series',
        mandatory=False, usedefault=True)

    plot_fig = traits.Bool(
        False, desc="Plotting mean signal or not", usedefault=True)


class ExtractMeanTSOutputSpec(TraitedSpec):

    mean_masked_ts_file = File(exists=True, desc="mean ts in .npy format")


class ExtractMeanTS(BaseInterface):

    """
    Description:

    Extract mean time series from a labelled mask in Nifti Format
    where the voxels of interest have values 1 (mask_file),
    or from a percent mask (filter_mask_file) with values higher than
    threshold (filter_thr)

    Inputs:

        file_4D:
            type = File, exists=True, desc='4D volume to be extracted'
            mandatory=True

        mask_file:
            type = File, xor = ['filter_mask_file'], exists=True,
            desc='mask file where all voxels belonging to the selected region
            have index 1',
            mandatory=True

        filter_mask_file:
            type = File, xor = ['mask_file'],requires = ['filter_thr'],
            exists=True,
            desc='mask file where all voxels belonging to the
            selected region have values higher than threshold',
            mandatory=True

        filter_thr:
            type = Float, default = 0.99, usedefault = True,
            desc='Value to threshold filter_mask'

        suffix:
            type = String, default = "suf",
            desc='Suffix added to describe the extracted time
            series',
            mandatory=False,
            usedefault = True

        plot_fig:
            type = Bool, default = False,
            desc = "Plotting mean signal or not",
            usedefault = True

    Outputs:

        mean_masked_ts_file:
            type = File, exists=True, desc="mean ts in .npy format"


    """
    input_spec = ExtractMeanTSInputSpec
    output_spec = ExtractMeanTSOutputSpec

    def _run_interface(self, runtime):

        file_4D = self.inputs.file_4D
        ROI_coord = self.inputs.ROI_coord
        mask_file = self.inputs.mask_file
        filter_mask_file = self.inputs.filter_mask_file
        filter_thr = self.inputs.filter_thr
        plot_fig = self.inputs.plot_fig
        suffix = self.inputs.suffix

        # Reading 4D volume file to extract time series
        img = nib.load(file_4D)
        img_data = img.get_data()

        # Reading 3D mask file
        if isdefined(mask_file):
            mask_data = nib.load(mask_file).get_data()

        elif isdefined(filter_mask_file) and isdefined(filter_thr):
            filter_mask_data = nib.load(filter_mask_file).get_data()
            mask_data = np.zeros(shape=filter_mask_data.shape, dtype='int')
            mask_data[filter_mask_data > filter_thr] = 1

        elif isdefined(ROI_coord):
            mask_data = np.zeros(shape=img_data.shape[:3], dtype=int)
            ROI_coord = np.array(ROI_coord, dtype=int)
            assert check_np_dimension(mask_data.shape, ROI_coord), \
                ("Error, non compatible indexes {} with shape {}".format(
                    ROI_coord, mask_data.shape))
            mask_data[ROI_coord[0], ROI_coord[1], ROI_coord[2]] = 1

        else:
            raise(ValueError, "Error, either mask_file or (filter_mask_file \
                and filter_thr) or ROI_coord should be defined")

        # Retaining only time series who are within the mask + non_zero
        mean_masked_ts = mean_select_mask_data(img_data, mask_data)

        # saving mean_masked_ts
        mean_masked_ts_file = os.path.abspath('mean_' + suffix + '_ts.txt')
        np.savetxt(mean_masked_ts_file, mean_masked_ts, fmt='%.3f')

        if plot_fig:
            # plotting mean_masked_ts
            plot_signals(
                os.path.abspath('mean_' + suffix + '_ts.eps'),
                mean_masked_ts)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        if isdefined(self.inputs.suffix):
            suffix = self.inputs.suffix

        else:
            suffix = "suf"

        outputs["mean_masked_ts_file"] = os.path.abspath(
            'mean_' + suffix + '_ts.txt')

        return outputs


# ConcatTS
class ConcatTSInputSpec(BaseInterfaceInputSpec):

    all_ts_file = File(
        exists=True, desc='npy file containing all ts to be concatenated',
        mandatory=True)


class ConcatTSOutputSpec(TraitedSpec):

    concatenated_ts_file = File(exists=True, desc="ts after concatenation")


class ConcatTS(BaseInterface):

    """
    Description:

    Concatenate time series

    Inputs:

        all_ts_file:
            type = File, exists=True,
            desc='npy file containing all ts to be concatenated',
            mandatory=True

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


# MergeTS
class MergeTSInputSpec(BaseInterfaceInputSpec):

    all_ts_files = traits.List(File(
        exists=True), desc='list of npy files containing all ts to be merged',
        mandatory=True)


class MergeTSOutputSpec(TraitedSpec):

    merged_ts_file = File(exists=True, desc="ts after merge")


class MergeTS(BaseInterface):

    """
    Description:

    Merges time series from several files

    Inputs:

        all_ts_files:
            type = List of Files, exists=True,
            desc='list of npy files containing all ts to be merged',
            mandatory=True

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

        for i, all_ts_file in enumerate(all_ts_files):

            all_ts = np.load(all_ts_file)

            concatenated_ts = all_ts.swapaxes(
                1, 0).reshape(all_ts.shape[1], -1)

            print(concatenated_ts.shape)

            if len(concatenated_ts.shape) > 1:

                if i == 0:
                    merged_ts = concatenated_ts.copy()
                else:
                    merged_ts = np.concatenate(
                        (merged_ts, concatenated_ts), axis=1)

        merged_ts_file = os.path.abspath("merged_ts.npy")
        np.save(merged_ts_file, merged_ts)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["merged_ts_file"] = os.path.abspath("merged_ts.npy")
        return outputs


# SeparateTS

class SeparateTSInputSpec(BaseInterfaceInputSpec):

    all_ts_file = File(
        exists=True, desc='npy file containing all ts to be concatenated',
        mandatory=True)


class SeparateTSOutputSpec(TraitedSpec):

    separated_ts_files = traits.List(
        File(exists=True), desc="ts files after separation")


class SeparateTS(BaseInterface):

    """
    Description:

    Save all time series from a npy file
    to several single time series npy files

    Inputs:

        all_ts_file:
            type = File, exists=True,
            desc='npy file containing all ts to be concatenated',
            mandatory=True

    Outputs:

        separated_ts_files
            type = List of Files, exists=True, desc="ts files after separation"

    Comments:

    Not sure where it is used...
    """

    input_spec = SeparateTSInputSpec
    output_spec = SeparateTSOutputSpec

    def _run_interface(self, runtime):

        all_ts_file = self.inputs.all_ts_file
        path, fname_ts, ext = split_f(all_ts_file)

        # loading ts shape = (trigs, electrods, time points)
        all_ts = np.load(all_ts_file)

        separated_ts_files = []

        for i in range(all_ts.shape[0]):

            sep_ts_file = os.path.abspath("{}_trig_{}.npy".format(
                fname_ts, str(i)))

            np.save(sep_ts_file, all_ts[i, :, :])
            separated_ts_files.append(sep_ts_file)
        self.separated_ts_files = separated_ts_files

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["separated_ts_files"] = self.separated_ts_files
        return outputs


# RegressCovar
class RegressCovarInputSpec(BaseInterfaceInputSpec):
    masked_ts_file = File(
        exists=True, desc='time series in npy format', mandatory=True)

    rp_file = File(exists=True, desc='Movement parameters', mandatory=False)

    mean_wm_ts_file = File(
        exists=True, desc='White matter signal', mandatory=False)

    mean_csf_ts_file = File(
        exists=True, desc='Cerebro-spinal fluid (ventricules) signal',
        mandatory=False)

    filtered_normalized = traits.Bool(
        True, usedefault=True,
        desc="Is the signal filtered and normalized after regression?")

    plot_fig = traits.Bool(True, usedefault=True, desc="Plotting signals?")


class RegressCovarOutputSpec(TraitedSpec):

    resid_ts_file = File(
        exists=True,
        desc="residuals of time series after regression of all paramters")


class RegressCovar(BaseInterface):
    """
    Description:

    Regress parameters of non-interest
    (i.e. movement parameters, white matter, csf) from signal.
    Optionnally filter and normalize (z-score) the residuals.

    Inputs:

        masked_ts_file:
            type = File, exists=True,
            desc='Time series in npy format',
            mandatory=True

        rp_file:
            type = File, exists=True,
            desc='Movement parameters in txt format, SPM style',
            mandatory=False

        mean_wm_ts_file:
            type = File: exists=True, desc='White matter signal',
            mandatory=False


        mean_csf_ts_file:
            type = File, exists=True,
            desc='Cerebro-spinal fluid (ventricules) signal', mandatory=False

        filtered_normalized:
            type= Bool, default = True, usedefault = True ,
            desc = "Filter and Normalize the signal  after regression?"

    Outputs:

        resid_ts_file:
            type = File, exists=True,
            desc="residuals of time series after regression of all paramters"


    """
    input_spec = RegressCovarInputSpec
    output_spec = RegressCovarOutputSpec

    def _run_interface(self, runtime):

        # load masked_ts_file
        data_mask_matrix = np.loadtxt(self.inputs.masked_ts_file)

        if isdefined(self.inputs.rp_file):

            # load rp parameters
            rp = np.genfromtxt(self.inputs.rp_file)

        else:
            rp = None

        if isdefined(self.inputs.mean_csf_ts_file):

            # load mean_csf_ts_file
            mean_csf_ts = np.loadtxt(self.inputs.mean_csf_ts_file)
            mean_csf_ts = mean_csf_ts.reshape(mean_csf_ts.shape[0], -1)

        else:
            mean_csf_ts = None

        if isdefined(self.inputs.mean_wm_ts_file):

            # load mean_wm_ts_file
            mean_wm_ts = np.loadtxt(self.inputs.mean_wm_ts_file)
            mean_wm_ts = mean_csf_ts.reshape(mean_wm_ts.shape[0], -1)

        else:
            mean_wm_ts = None

        regs = (rp, mean_csf_ts, mean_wm_ts)

        if all([a is None for a in regs]):
            resid_data_matrix = data_mask_matrix

        else:
            keep_regs = [a for a in regs if a is not None]
            rp = np.concatenate(keep_regs, axis=1)

            # regression movement parameters, return the residuals
            resid_data_matrix = regress_parameters(data_mask_matrix, rp)

        if self.inputs.filtered_normalized:

            # filtering data
            resid_filt_data_matrix = filter_data(resid_data_matrix)

            # normalizing
            z_score_data_matrix = normalize_data(resid_filt_data_matrix)

            #  saving resid_ts
            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file, z_score_data_matrix)

            resid_ts_txt_file = os.path.abspath('resid_ts.txt')
            np.savetxt(resid_ts_txt_file, z_score_data_matrix, fmt='%0.3f')

            if self.inputs.plot_fig:

                # plotting resid_ts
                plot_resid_ts_file = os.path.abspath('resid_ts.eps')
                plot_sep_signals(plot_resid_ts_file, z_score_data_matrix)

                # plotting diff filtered and non filtered data
                plot_diff_filt_ts_file = os.path.abspath('diff_filt_ts.eps')
                diff_resid = resid_filt_data_matrix - resid_data_matrix
                plot_signals(plot_diff_filt_ts_file,
                             np.array(diff_resid, dtype='float'))

        else:

            # Using only regression
            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file, resid_data_matrix)

            resid_ts_txt_file = os.path.abspath('resid_ts.txt')
            np.savetxt(resid_ts_txt_file, resid_data_matrix, fmt='%0.3f')

            if self.inputs.plot_fig:
                # plotting resid_ts
                plot_resid_ts_file = os.path.abspath('resid_ts.eps')
                plot_sep_signals(plot_resid_ts_file, resid_data_matrix)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["resid_ts_file"] = os.path.abspath('resid_ts.npy')
        return outputs


# FindSPMRegressor
class FindSPMRegressorInputSpec(BaseInterfaceInputSpec):

    spm_mat_file = File(
        exists=True,
        desc='SPM design matrix after generate model', mandatory=True)

    regressor_name = traits.String(
        exists=True,
        desc='Name of the regressor in SPM design matrix to be looked after',
        mandatory=True)

    run_index = traits.Int(
        1,
        usedefault=True,
        desc="Run (session) index, default is one in SPM")

    only_positive_values = traits.Bool(
        True, usedefault=True,
        desc="Return only positive values of the regressor")

    concatenated_runs = traits.Bool(
        False, usedefault=True,
        desc="If concatenate runs, need to search for the lenghth of the \
            session")


class FindSPMRegressorOutputSpec(TraitedSpec):

    regressor_file = File(
        exists=True, desc="txt file containing the regressor")


class FindSPMRegressor(BaseInterface):
    """

    Description:

    Find regressor in SPM.mat and save it as timeseries txt file

    Inputs:

        spm_mat_file:
            type = File, exists=True,
            desc='SPM design matrix after generate model', mandatory=True

        regressor_name:
            type = String, exists=True,
            desc='Name of the regressor in SPM design matrix to be looked
            after', mandatory=True

        run_index:
            type = Int, default = 1 , usedefault = True ,
            desc = "Run (session) index, default is one in SPM"

        only_positive_values:
            type = Bool, default = True, usedefault = True ,
            desc = "Return only positive values of the regressor (negative
            values are set to 0); Otherwise return all values"

        concatenated_runs:
            type = Bool, default = False , usedefault = True,
            desc = "If concatenate runs, need to search for the length of the
            session"

            Deprecation: #concatenate_runs = traits.Int(1, usedefault = True,
            desc = "If concatenate runs, how many runs there is (needed to
            return the part of the regressors that is active for the session
            only)")
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

        # Choosing the column according to the regressor name
        cond_name = 'Sn(' + str(run_index) + ') ' + regressor_name + '*bf(1)'
        _, col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == cond_name)

        # reformating matrix (1,len) in vector (len)
        regressor_vect = d['SPM']['xX'][0][0]['X'][0][0][:, col].reshape(-1)

        assert np.sum(
            regressor_vect) != 0, "Error, empty regressor {}".format(cond_name)

        if only_positive_values:
            regressor_vect[regressor_vect < 0] = 0

        if concatenated_runs:
            samples = \
                d['SPM']['xX'][0][0]['K'][0][0]['row'][0][run_index-1][0]-1

            regressor_vect = regressor_vect[samples]

        # Saving extract_cond
        regressor_file = os.path.abspath('extract_cond.txt')

        np.savetxt(regressor_file, regressor_vect)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["regressor_file"] = os.path.abspath('extract_cond.txt')
        return outputs


# MergeRuns
class MergeRunsInputSpec(BaseInterfaceInputSpec):

    ts_files = traits.List(File(
        exists=True),
        desc='Numpy files with time series from different runs (sessions)',
        mandatory=True)

    regressor_files = traits.List(File(
        exists=True),
        desc='Txt files with regressors from different runs (sessions)',
        mandatory=True)

    coord_rois_files = traits.List(File(
        exists=True),
        desc='Txt files with coords from different runs (sessions)',
        mandatory=True)


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

        assert len(ts_files) == len(regressor_files), \
            ("Error, time series and regressors have different lengths")

        assert len(ts_files) == len(coord_rois_files), \
            ("Error, time series and  coordinates with different lengths")

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

                # TODO Finish case
                assert False, "Error, not implemented yet.... "

                # Suite a tester....
                # finir egalement la partie avec data_matrix_all_runs,
                # en supprimant les colonnes qui ne sont pas communes a tous
                # les runs...

                A = coord_rois_all_runs
                B = coord_rois

                nrows, ncols = A.shape
                dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                         'formats': ncols * [A.dtype]}

                C = np.intersect1d(A.view(dtype), B.view(dtype))

                # This last bit is optional
                # if you're okay with "C" being a structured array...
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
        regressor_all_runs = np.empty(shape=(0), dtype=float)

        # Sum regressors
        for i, regress_file in enumerate(regressor_files):

            regress_data_vector = np.loadtxt(regress_file)

            if regress_data_vector.shape[0] != 0:

                if regressor_all_runs.shape[0] == 0:

                    regressor_all_runs = regress_data_vector
                else:
                    regressor_all_runs += regress_data_vector

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


# ComputeConfCorMat
class ComputeConfCorMatInputSpec(BaseInterfaceInputSpec):

    ts_file = File(
        exists=True, desc='Numpy files with time series to be correlated',
        mandatory=True)

    transpose_ts = traits.Bool(
        True, usedefault=True, desc='whether to transpose timeseries',
        mandatory=True)

    weight_file = File(
        exists=True,
        desc='Weight of the correlation (normally, condition regressor file)',
        mandatory=False)

    conf_interval_prob = traits.Float(
        0.05, usedefault=True, desc='Confidence interval', mandatory=True)

    plot_mat = traits.Bool(True, usedefault=True,
                           desc='Confidence interval', mandatory=False)

    labels_file = File(
        exists=True, desc='Name of the nodes (used only if plot = true)',
        mandatory=False)

    method = traits.Enum(
        "Pearson",
        "Spearman",
        usedefault=True,
        desc='Method used for computing correlation -default Pearson')


class ComputeConfCorMatOutputSpec(TraitedSpec):

    cor_mat_file = File(
        exists=True, desc="npy file containing the R values of correlation")

    Z_cor_mat_file = File(
        exists=True,
        desc="npy file containing the Z-values (after Fisher's R-to-Z \
            trasformation) of correlation")

    conf_cor_mat_file = File(
        exists=True,
        desc="npy file containing the confidence interval around R values")

    Z_conf_cor_mat_file = File(
        exists=True,
        desc="npy file containing the Z-values (after Fisher's R-to-Z \
            trasformation) of correlation")


class ComputeConfCorMat(BaseInterface):
    """
    Description:

    Compute correlation between time series, with a given confidence interval.
    If weight_file is specified, used for weighted correlation

    Inputs:

        ts_file:
            type = File, exists=True,
            desc='Numpy files with time series to be correlated',mandatory=True

        transpose_ts:
            type = Bool, default=True,usedefault = True,
            desc = 'whether to transpose timeseries', mandatory = True

        weight_file:
            type = File, exists=True,
            desc='Weight of the correlation (normally, condition
            regressor file)',
            mandatory=False

        conf_interval_prob:
            type = Float, default = 0.05, usedefault = True,
            desc='Confidence interval', mandatory=True

        plot_mat:
            type = Bool, default = True, usedefault = True,
            desc='Confidence interval', mandatory=False

        labels_file:
            type = File, exists=True,
            desc='Name of the nodes (used only if plot = true)',
            mandatory=False

    Outputs:

        cor_mat_file:
            type = File, exists=True,
            desc="npy file containing the R values of correlation"

        Z_cor_mat_file:
            type = File, exists=True,
            desc="npy file containing the Z-values (after Fisher's R-to-Z
            trasformation) of correlation"

        conf_cor_mat_file:
            type = File, exists=True,
            desc="npy file containing the confidence interval around R values"

    """

    input_spec = ComputeConfCorMatInputSpec
    output_spec = ComputeConfCorMatOutputSpec

    def _run_interface(self, runtime):

        ts_file = self.inputs.ts_file
        weight_file = self.inputs.weight_file
        transpose_ts = self.inputs.transpose_ts
        conf_interval_prob = self.inputs.conf_interval_prob
        plot_mat = self.inputs.plot_mat
        labels_file = self.inputs.labels_file
        method = self.inputs.method

        # load time series

        path, fname, ext = split_f(ts_file)

        data_matrix = np.load(ts_file)

        if transpose_ts:
            print("Transposing data")
            data_matrix = np.transpose(data_matrix)

        if isdefined(weight_file):
            weight_vect = np.loadtxt(weight_file)

        else:
            weight_vect = np.ones(shape=(data_matrix.shape[0]))

        if method == "Pearson":
            print("Transposing data")
            cor_mat, Z_cor_mat, conf_cor_mat, Z_conf_cor_mat = \
                return_conf_cor_mat(data_matrix, weight_vect,
                                    conf_interval_prob)

            # Z_cor_mat
            cor_mat = cor_mat + np.transpose(cor_mat)
            Z_cor_mat = Z_cor_mat + np.transpose(Z_cor_mat)

            # saving cor_mat as npy
            cor_mat_file = os.path.abspath('cor_mat_' + fname + '.npy')
            np.save(cor_mat_file, cor_mat)

            # saving conf_cor_mat as npy
            conf_cor_mat_file = os.path.abspath(
                'conf_cor_mat_' + fname + '.npy')
            np.save(conf_cor_mat_file, conf_cor_mat)

            # saving Z_cor_mat as npy")
            Z_cor_mat_file = os.path.abspath('Z_cor_mat_' + fname + '.npy')
            np.save(Z_cor_mat_file, Z_cor_mat)

            # saving Z_conf_cor_mat as npy
            Z_conf_cor_mat_file = os.path.abspath(
                'Z_conf_cor_mat_' + fname + '.npy')
            Z_conf_cor_mat = Z_conf_cor_mat + np.transpose(Z_conf_cor_mat)
            np.save(Z_conf_cor_mat_file, Z_conf_cor_mat)

        elif method == "Spearman":

            rho_mat, pval_mat = scipy.stats.spearmanr(data_matrix)

        if plot_mat:

            if isdefined(labels_file):
                labels = [line.strip() for line in open(labels_file)]

            else:
                labels = []

            if method == "Spearman":
                # rho_mat
                plot_heatmap_rho_mat_file = os.path.abspath(
                    'heatmap_rho_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_rho_mat_file, rho_mat,
                            list_labels=labels)

                # rho_mat histogram
                plot_hist_rho_mat_file = os.path.abspath(
                    'hist_rho_mat_' + fname + '.eps')

                plot_hist(plot_hist_rho_mat_file, rho_mat, nb_bins=100)

                # pval_mat
                plot_heatmap_pval_mat_file = os.path.abspath(
                    'heatmap_pval_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_pval_mat_file, pval_mat,
                            list_labels=labels)

                # pval_mat histogram
                plot_hist_pval_mat_file = os.path.abspath(
                    'hist_pval_mat_' + fname + '.eps')

                plot_hist(plot_hist_pval_mat_file, pval_mat, nb_bins=100)

            elif method == "Pearson":
                # cor_mat heatmap
                plot_heatmap_cor_mat_file = os.path.abspath(
                    'heatmap_cor_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_cor_mat_file, cor_mat,
                            list_labels=labels)

                # cor_mat histogram
                plot_hist_cor_mat_file = os.path.abspath(
                    'hist_cor_mat_' + fname + '.eps')

                plot_hist(plot_hist_cor_mat_file, cor_mat, nb_bins=100)

                # Z_cor_mat heatmap
                plot_heatmap_Z_cor_mat_file = os.path.abspath(
                    'heatmap_Z_cor_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_Z_cor_mat_file,
                            Z_cor_mat, list_labels=labels)

                # Z_cor_mat histogram
                plot_hist_Z_cor_mat_file = os.path.abspath(
                    'hist_Z_cor_mat_' + fname + '.eps')

                plot_hist(plot_hist_Z_cor_mat_file, Z_cor_mat, nb_bins=100)

                # conf_cor_mat heatmap
                plot_heatmap_conf_cor_mat_file = os.path.abspath(
                    'heatmap_conf_cor_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_conf_cor_mat_file,
                            conf_cor_mat, list_labels=labels)

                # Z_conf_cor_mat heatmap
                plot_heatmap_Z_conf_cor_mat_file = os.path.abspath(
                    'heatmap_Z_conf_cor_mat_' + fname + '.eps')

                plot_cormat(plot_heatmap_Z_conf_cor_mat_file,
                            Z_conf_cor_mat, list_labels=labels)

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

        return outputs


# ComputeSpearmanMat
# TODO suppressed, as is redondant with previous method now...
# not sure which one I used so far...
class ComputeSpearmanMatInputSpec(BaseInterfaceInputSpec):

    ts_file = File(
        exists=True,
        desc='Numpy files with time series to be correlated',
        mandatory=True)

    transpose_ts = traits.Bool(
        True,
        usedefault=True,
        desc='whether to transpose timeseries',
        mandatory=True)

    plot_mat = traits.Bool(
        True,
        usedefault=True,
        desc='Using matplotlib to plot',
        mandatory=False)

    labels_file = File(
        exists=True,
        desc='Name of the nodes (used only if plot = true)',
        mandatory=False)

    export_csv = traits.Bool(
        True,
        usedefault=True,
        desc='save as CSV as well',
        mandatory=False)


class ComputeSpearmanMatOutputSpec(TraitedSpec):

    rho_mat_file = File(
        exists=True, desc="npy file containing the rho values of correlation")

    pval_mat_file = File(exists=True, desc="npy file containing the p-values")


class ComputeSpearmanMat(BaseInterface):
    """
    Description:

    Compute correlation between time series, with a given confidence interval.
    If weight_file is specified, used for weighted correlation

    Inputs:

        ts_file:
            type = File, exists=True,
            desc='Numpy files with time series to be correlated',
            mandatory=True

        transpose_ts:
            type = Bool, default=True,usedefault = True,
            desc =  'whether to transpose timeseries',
            mandatory = True

        plot_mat:
            type = Bool, default = True, usedefault = True,
            desc='Confidence interval', mandatory=False

        labels_file:
            type = File, exists=True,
            desc='Name of the nodes (used only if plot = true)',
            mandatory=False

    Outputs:

        rho_mat_file :
            type = File, exists=True,
            desc="npy file containing the rho values of correlation"

        pval_mat_file:
            type = File, exists=True, desc="npy file containing the p-values"

    """

    input_spec = ComputeSpearmanMatInputSpec
    output_spec = ComputeSpearmanMatOutputSpec

    def _run_interface(self, runtime):

        ts_file = self.inputs.ts_file
        transpose_ts = self.inputs.transpose_ts
        plot_mat = self.inputs.plot_mat
        export_csv = self.inputs.export_csv
        labels_file = self.inputs.labels_file

        # load resid data
        path, fname, ext = split_f(ts_file)
        data_matrix = np.load(ts_file)

        if transpose_ts:

            data_matrix = np.transpose(data_matrix)

        rho_mat, pval_mat = scipy.stats.spearmanr(data_matrix)
        np.fill_diagonal(rho_mat, 0)

        # saving rho_mat as npy
        rho_mat_file = os.path.abspath('rho_mat_' + fname + '.npy')
        np.save(rho_mat_file, rho_mat)

        #  saving pval_mat as npy
        pval_mat_file = os.path.abspath('pval_mat_' + fname + '.npy')
        np.save(pval_mat_file, pval_mat)

        if isdefined(labels_file):

            labels = [line.strip() for line in open(labels_file)]

        else:
            labels = []

        if plot_mat:

            # heatmap rho_mat
            plot_heatmap_rho_mat_file = os.path.abspath(
                'heatmap_rho_mat_' + fname + '.eps')
            plot_cormat(plot_heatmap_rho_mat_file, rho_mat, list_labels=labels)

        if export_csv:

            if len(labels) == rho_mat.shape[0] and \
                    len(labels) == rho_mat.shape[1]:

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

        return outputs


# used in run_mean_correl
# PrepareMeanCorrel
class PrepareMeanCorrelInputSpec(BaseInterfaceInputSpec):

    cor_mat_files = traits.List(
        File(exists=True),
        desc='Numpy files with correlation matrices gm_mask_coords',
        mandatory=True)

    coords_files = traits.List(
        File(exists=True), desc='list of all coordinates in numpy space files \
            for each subject (after removal of non void data)',
        mandatory=False)

    labels_files = traits.List(
        File(exists=True), desc='list of labels (in txt format) for each \
            subject (after removal of non void data)',
        mandatory=False)

    gm_mask_coords_file = File(
        exists=True,
        desc='Coordinates in numpy space, corresponding to all possible nodes \
            in the original space', mandatory=False)

    gm_mask_labels_file = File(
        exists=True,
        desc='Labels for all possible nodes - in case coords are varying from \
            one indiv to the other (source space for example)',
        mandatory=False)

    plot_mat = traits.Bool(True, usedefault=True, mandatory=False)

    export_csv = traits.Bool(False, usedefault=True, mandatory=False)


class PrepareMeanCorrelOutputSpec(TraitedSpec):

    group_cor_mat_matrix_file = File(
        exists=True, desc="npy file containing all correlation matrices in 3D")

    sum_cor_mat_matrix_file = File(
        exists=True,
        desc="npy file containing the sum of all correlation matrices")

    sum_possible_edge_matrix_file = File(
        exists=True, desc="npy file containing the number of correlation \
            matrices where both nodes where actually defined in the mask")

    avg_cor_mat_matrix_file = File(
        exists=True,
        desc="npy file containing the average of all correlation matrices")


class PrepareMeanCorrel(BaseInterface):
    """
    Decription:

    Return average of correlation values within the same common space
    (defined in gm_mask_coords), only when the nodes are defined for a given
    values

    Input:

    gm_mask_coords_file
        type = File, exists=True, desc='reference coordinates',mandatory=True

    cor_mat_files
        type = List of Files, exists=True,
        desc='Numpy files with correlation matrices ', mandatory=True

    coords_files:
        type = List of Files, exists=True,
        desc='Txt files with coordinates (corresponding to the space also
        described in gm_mask_coords)', mandatory=True

    gm_mask_labels_file:
        type = File, exists=True, desc='reference labels',mandatory=False

    plot_mat:
        type = Bool; default = True, usedefault = True, mandatory = False


    Outputs:

    group_cor_mat_matrix_file:
        type = File,exists=True,
        desc="npy file containing all correlation matrices in 3D"

    sum_cor_mat_matrix_file
        type = File,exists=True,
        desc="npy file containing the sum of all correlation matrices"

    sum_possible_edge_matrix_file:
        type = File, exists=True,
        desc="npy file containing the number of correlation matrices where both
        nodes where actually defined in the mask"

    avg_cor_mat_matrix_file:
        type = File, exists=True,
        desc="npy file containing the average of all correlation matrices"
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

        if isdefined(self.inputs.gm_mask_coords_file) and\
                isdefined(self.inputs.coords_files):

            coords_files = self.inputs.coords_files
            gm_mask_coords_file = self.inputs.gm_mask_coords_file

            gm_mask_coords = np.array(
                np.loadtxt(gm_mask_coords_file), dtype=int)

            sum_cor_mat_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]),
                dtype=float)

            sum_possible_edge_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

            group_cor_mat_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0],
                 len(cor_mat_files)), dtype=float)

            assert len(cor_mat_files) == len(coords_files), \
                ("Error, length of cor_mat_files and coords_files are \
                    imcompatible {} {}".format(len(cor_mat_files),
                                               len(coords_files)))

            for index_file in range(len(cor_mat_files)):

                print(cor_mat_files[index_file])

                if os.path.exists(cor_mat_files[index_file]) and\
                        os.path.exists(coords_files[index_file]):

                    Z_cor_mat = np.load(cor_mat_files[index_file])

                    coords = np.array(np.loadtxt(
                        coords_files[index_file]), dtype=int)

                    corres_cor_mat, possible_edge_mat = \
                        return_corres_correl_mat(Z_cor_mat, coords,
                                                 gm_mask_coords)

                    np.fill_diagonal(corres_cor_mat, 0)
                    np.fill_diagonal(possible_edge_mat, 1)
                    sum_cor_mat_matrix += corres_cor_mat
                    sum_possible_edge_matrix += possible_edge_mat
                    group_cor_mat_matrix[:, :, index_file] = corres_cor_mat

                else:
                    print("Warning, one or more files between {} and {} is \
                        missing".format(cor_mat_files[index_file],
                                        coords_files[index_file]))

        elif isdefined(self.inputs.gm_mask_labels_file) and \
                isdefined(self.inputs.labels_files):

            labels_files = self.inputs.labels_files
            gm_mask_labels_file = self.inputs.gm_mask_labels_file
            gm_mask_labels = [line.strip()
                              for line in open(gm_mask_labels_file)]

            gm_size = len(gm_mask_labels)
            sum_cor_mat_matrix = np.zeros((gm_size, gm_size), dtype=float)
            sum_possible_edge_matrix = np.zeros((gm_size, gm_size), dtype=int)

            group_cor_mat_matrix = np.zeros(
                (gm_size, gm_size, len(cor_mat_files)), dtype=float)

            assert len(cor_mat_files) == len(labels_files), \
                ("warning, length of cor_mat_files, labels_files are \
                    imcompatible {} {} {}".format(len(cor_mat_files),
                                                  len(labels_files)))

            for i in range(len(cor_mat_files)):

                if os.path.exists(cor_mat_files[i]) and \
                        os.path.exists(labels_files[i]):

                    Z_cor_mat = np.load(cor_mat_files[i])
                    print(Z_cor_mat.shape)

                    labels = [line.strip() for line in open(labels_files[i])]
                    print(labels)

                    corres_cor_mat, possible_edge_mat = \
                        return_corres_correl_mat_labels(
                            Z_cor_mat, labels, gm_mask_labels)

                    np.fill_diagonal(corres_cor_mat, 0)

                    np.fill_diagonal(possible_edge_mat, 1)

                    sum_cor_mat_matrix += corres_cor_mat

                    sum_possible_edge_matrix += possible_edge_mat

                    group_cor_mat_matrix[:, :, i] = corres_cor_mat

                else:
                    print("Warning, one or more files between {} {} do not \
                        exists".format(cor_mat_files[i], labels_files[i]))

        else:

            group_cor_mat_matrix = np.array(
                [np.load(cor_mat_file) for cor_mat_file in cor_mat_files
                    if os.path.exists(cor_mat_file)])
            sum_cor_mat_matrix = np.sum(group_cor_mat_matrix, axis=0)
            sum_possible_edge_matrix = np.ones(
                shape=sum_cor_mat_matrix.shape)*len(cor_mat_files)

        self.group_cor_mat_matrix_file = os.path.abspath(
            'group_cor_mat_matrix.npy')

        np.save(self.group_cor_mat_matrix_file, group_cor_mat_matrix)

        self.sum_cor_mat_matrix_file = os.path.abspath(
            'sum_cor_mat_matrix.npy')

        np.save(self.sum_cor_mat_matrix_file, sum_cor_mat_matrix)

        self.sum_possible_edge_matrix_file = os.path.abspath(
            'sum_possible_edge_matrix.npy')

        np.save(self.sum_possible_edge_matrix_file, sum_possible_edge_matrix)

        self.avg_cor_mat_matrix_file = os.path.abspath(
            'avg_cor_mat_matrix.npy')

        if np.sum(np.array(sum_possible_edge_matrix == 0)) == 0:

            avg_cor_mat_matrix = np.divide(
                np.array(sum_cor_mat_matrix, dtype=float),
                np.array(sum_possible_edge_matrix, dtype=float))

            avg_cor_mat_matrix[np.isnan(avg_cor_mat_matrix)] = 0.0

            np.save(self.avg_cor_mat_matrix_file, avg_cor_mat_matrix)

            if export_csv:
                csv_avg_cor_mat_matrix_file = os.path.abspath(
                    'avg_cor_mat_matrix.csv')
                df = pd.DataFrame(avg_cor_mat_matrix,
                                  index=labels, columns=labels)
                df.to_csv(csv_avg_cor_mat_matrix_file)

        else:

            avg_cor_mat_matrix = np.divide(
                np.array(sum_cor_mat_matrix, dtype=float),
                np.array(sum_possible_edge_matrix, dtype=float))

            avg_cor_mat_matrix[np.isnan(avg_cor_mat_matrix)] = 0.0

            np.save(self.avg_cor_mat_matrix_file, avg_cor_mat_matrix)

            if export_csv:

                csv_avg_cor_mat_matrix_file = os.path.abspath(
                    'avg_cor_mat_matrix.csv')

                df = pd.DataFrame(avg_cor_mat_matrix,
                                  index=labels, columns=labels)

                df.to_csv(csv_avg_cor_mat_matrix_file)

        if plot_mat:

            # heatmap
            plot_heatmap_avg_cor_mat_file = os.path.abspath(
                'heatmap_avg_cor_mat.eps')
            plot_cormat(plot_heatmap_avg_cor_mat_file,
                        avg_cor_mat_matrix, list_labels=labels)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["group_cor_mat_matrix_file"] = self.group_cor_mat_matrix_file
        outputs["sum_cor_mat_matrix_file"] = self.sum_cor_mat_matrix_file
        outputs["sum_possible_edge_matrix_file"] = \
            self.sum_possible_edge_matrix_file
        outputs["avg_cor_mat_matrix_file"] = self.avg_cor_mat_matrix_file

        return outputs


# PreparePermutMeanCorrel


class PreparePermutMeanCorrelInputSpec(BaseInterfaceInputSpec):

    cor_mat_files = traits.List(
        File(exists=True),
        desc='Numpy files with correlation matrices gm_mask_coords',
        mandatory=True)

    permut_group_sizes = traits.List(
        traits.Int,
        desc='How to split the groups after shuffling',
        mandatory=True)

    seed = traits.Int(0, usedefault=True, decs='Start of random process')


class PreparePermutMeanCorrelOutputSpec(TraitedSpec):

    permut_mean_cormat_files = traits.List(
        File(exists=True),
        desc="npy files with the average of permuted correlation matrices")


class PreparePermutMeanCorrel(BaseInterface):
    """
    Return average of correlation values after shuffling orig datasets
    """
    input_spec = PreparePermutMeanCorrelInputSpec
    output_spec = PreparePermutMeanCorrelOutputSpec

    def _run_interface(self, runtime):
        np.random.seed(self.inputs.seed)
        cormats = [np.load(cor_mat_file)
                   for cor_mat_file in self.inputs.cor_mat_files]

        assert len(cormats) == sum(self.inputs.permut_group_sizes), ("Error,\
            len(cormats) {} != sum permut_group_sizes {1}".format(
            len(cormats), sum(self.inputs.permut_group_sizes)))

        subj_indexes = np.arange(len(cormats))
        np.random.shuffle(subj_indexes)
        subj_indexes_file = os.path.abspath("subj_indexes.txt")
        f = open(subj_indexes_file, "w+")
        np.savetxt(f, subj_indexes, fmt="%d")
        min_index = 0
        cormats = np.array(cormats)

        self.permut_mean_cormat_files = []
        for i, cur_nb_subj in enumerate(self.inputs.permut_group_sizes):
            cur_range = np.arange(min_index, min_index+cur_nb_subj)
            rand_indexes = subj_indexes[cur_range]
            np.savetxt(f, rand_indexes, fmt="%d")
            permut_cormats = cormats[rand_indexes, :, :]
            permut_mean_cormat = np.mean(permut_cormats, axis=0)
            permut_mean_cormat_file = os.path.abspath(
                "permut_mean_cormat_" + str(i) + ".npy")
            np.save(permut_mean_cormat_file, permut_mean_cormat)
            self.permut_mean_cormat_files.append(permut_mean_cormat_file)
            min_index += cur_nb_subj
        f.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["permut_mean_cormat_files"] = self.permut_mean_cormat_files
        return outputs
