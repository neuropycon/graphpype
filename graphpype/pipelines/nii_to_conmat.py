
"""
From nifti file to conmat
"""
import nipype.pipeline.engine as pe

import nipype.interfaces.utility as niu
import nipype.interfaces.spm.utils as spmu

from nipype.interfaces.niftyreg.regutils import RegResample

from graphpype.nodes.correl_mat import (
    IntersectMask, ExtractTS, ExtractMeanTS, RegressCovar, FindSPMRegressor,
    ComputeConfCorMat)

from graphpype.utils import show_files


def create_pipeline_nii_to_conmat_simple(
        main_path, pipeline_name="nii_to_conmat", conf_interval_prob=0.05,
        background_val=-1.0, plot=True):
    """
    Description:

    Pipeline from nifti 4D (after preprocessing) to connectivity matrices,
    no segmentation in tissues given, but coords for wm and csf are available
    and regressed. coords / labels o indexed mask are also available

    Inputs (inputnode):

        * nii_4D_file
        * ROI_mask_file

    Optional inputs (inputnode) :
        * rp_file
        * ROI_coords_file
        * ROI_MNI_coords_file
        * ROI_labels_file

    Comments:

    Typically used after nipype preprocessing pipeline and before
    conmat_to_graph pipeline

    """

    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'nii_4D_file', 'ROI_mask_file', 'rp_file', 'ROI_coords_file',
        'ROI_MNI_coords_file', 'ROI_labels_file']), name='inputnode')

    # Nodes version: use min_BOLD_intensity and
    # return coords where signal is strong enough
    extract_mean_ROI_ts = pe.Node(interface=ExtractTS(
        plot_fig=False), name='extract_mean_ROI_ts')

    extract_mean_ROI_ts.inputs.background_val = background_val

    pipeline.connect(inputnode, 'nii_4D_file', extract_mean_ROI_ts, 'file_4D')
    pipeline.connect(inputnode, 'ROI_mask_file',
                     extract_mean_ROI_ts, 'indexed_rois_file')
    pipeline.connect(inputnode, 'ROI_coords_file',
                     extract_mean_ROI_ts, 'coord_rois_file')
    pipeline.connect(inputnode, 'ROI_MNI_coords_file',
                     extract_mean_ROI_ts, 'MNI_coord_rois_file')
    pipeline.connect(inputnode, 'ROI_labels_file',
                     extract_mean_ROI_ts, 'label_rois_file')

    # regress covariates
    regress_covar = pe.Node(interface=RegressCovar(plot_fig=plot), iterfield=[
                            'masked_ts_file', 'rp_file'], name='regress_covar')

    pipeline.connect(extract_mean_ROI_ts, 'mean_masked_ts_file',
                     regress_covar, 'masked_ts_file')
    pipeline.connect(inputnode, 'rp_file', regress_covar, 'rp_file')

    # compute correlations
    compute_conf_cor_mat = pe.Node(
        interface=ComputeConfCorMat(plot_mat=plot),
        name='compute_conf_cor_mat')
    compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

    pipeline.connect(regress_covar, 'resid_ts_file',
                     compute_conf_cor_mat, 'ts_file')
    pipeline.connect(extract_mean_ROI_ts, 'subj_label_rois_file',
                     compute_conf_cor_mat, 'labels_file')

    return pipeline


def create_pipeline_nii_to_conmat_seg_template(
        main_path, pipeline_name="nii_to_conmat", conf_interval_prob=0.05):
    """
    Description:

    Pipeline from nifti 4D (after preprocessing) to connectivity matrices

    Inputs (inputnode):

        * nii_4D_file
        * rp_file

        * wm_anat_file
        * csf_anat_file

        * ROI_mask_file
        * ROI_coords_file
        * ROI_MNI_coords_file
        * ROI_labels_file

    Comments:

    Typically used after nipype preprocessing pipeline and
    before conmat_to_graph pipeline

    """

    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'nii_4D_file', 'rp_file', 'wm_anat_file', 'csf_anat_file',
        'ROI_mask_file', 'ROI_coords_file', 'ROI_MNI_coords_file',
        'ROI_labels_file']), name='inputnode')

    # Nodes version: use min_BOLD_intensity and
    # return coords where signal is strong enough
    extract_mean_ROI_ts = pe.Node(interface=ExtractTS(
        plot_fig=False), name='extract_mean_ROI_ts')

    pipeline.connect(inputnode, 'nii_4D_file', extract_mean_ROI_ts, 'file_4D')
    pipeline.connect(inputnode, 'ROI_mask_file',
                     extract_mean_ROI_ts, 'indexed_rois_file')
    pipeline.connect(inputnode, 'ROI_coords_file',
                     extract_mean_ROI_ts, 'coord_rois_file')
    pipeline.connect(inputnode, 'ROI_MNI_coords_file',
                     extract_mean_ROI_ts, 'MNI_coord_rois_file')
    pipeline.connect(inputnode, 'ROI_labels_file',
                     extract_mean_ROI_ts, 'label_rois_file')

    # extract white matter signal
    compute_wm_ts = pe.Node(interface=ExtractMeanTS(
        plot_fig=False), name='extract_wm_ts')
    compute_wm_ts.inputs.suffix = 'wm'

    pipeline.connect(inputnode, 'nii_4D_file', compute_wm_ts, 'file_4D')
    pipeline.connect(inputnode, 'wm_anat_file',
                     compute_wm_ts, 'filter_mask_file')

    # extract csf signal
    compute_csf_ts = pe.Node(interface=ExtractMeanTS(
        plot_fig=False), name='extract_csf_ts')
    compute_csf_ts.inputs.suffix = 'csf'

    pipeline.connect(inputnode, 'nii_4D_file', compute_csf_ts, 'file_4D')
    pipeline.connect(inputnode, 'csf_anat_file',
                     compute_csf_ts, 'filter_mask_file')

    regress_covar = pe.Node(interface=RegressCovar(), iterfield=[
                            'masked_ts_file', 'rp_file'], name='regress_covar')

    pipeline.connect(extract_mean_ROI_ts, 'mean_masked_ts_file',
                     regress_covar, 'masked_ts_file')
    pipeline.connect(compute_wm_ts, 'mean_masked_ts_file',
                     regress_covar, 'mean_wm_ts_file')
    pipeline.connect(compute_csf_ts, 'mean_masked_ts_file',
                     regress_covar, 'mean_csf_ts_file')
    pipeline.connect(inputnode, 'rp_file', regress_covar, 'rp_file')

    # compute correlations

    compute_conf_cor_mat = pe.Node(
        interface=ComputeConfCorMat(), name='compute_conf_cor_mat')
    compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

    pipeline.connect(regress_covar, 'resid_ts_file',
                     compute_conf_cor_mat, 'ts_file')
    pipeline.connect(extract_mean_ROI_ts, 'subj_label_rois_file',
                     compute_conf_cor_mat, 'labels_file')

    return pipeline


def create_pipeline_nii_to_subj_ROI(
        main_path, filter_gm_threshold=0.9, pipeline_name="nii_to_subj_ROI",
        background_val=-1.0, plot=True, reslice=False, resample=False,
        min_BOLD_intensity=50, percent_signal=0.5):
    """
    Description:

    Pipeline from nifti 4D (after preprocessing) to connectivity matrices
    Use Grey matter for having a mask for each subject

    Inputs (inputnode):

        * nii_4D_file
        * gm_anat_file
        * ROI_mask_file
        * ROI_coords_file
        * ROI_MNI_coords_file
        * ROI_labels_file

    Comments:

    Typically used after nipype preprocessing pipeline and
    before conmat_to_graph pipeline

    """
    if reslice and resample:
        print("Only reslice OR resample can be true, setting reslice to False")
        reslice = False

    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'nii_4D_file', 'ROI_mask_file', 'gm_anat_file', 'ROI_coords_file',
        'ROI_MNI_coords_file', 'ROI_labels_file']), name='inputnode')

    # reslice gm
    if reslice:

        reslice_gm = pe.Node(interface=spmu.Reslice(), name='reslice_gm')
        pipeline.connect(inputnode, 'ROI_mask_file',
                         reslice_gm, 'space_defining')
        pipeline.connect(inputnode, 'gm_anat_file', reslice_gm, 'in_file')

    if resample:

        resample_gm = pe.Node(interface=RegResample(), name='resample_gm')
        pipeline.connect(inputnode, 'ROI_mask_file', resample_gm, 'ref_file')
        pipeline.connect(inputnode, 'gm_anat_file', resample_gm, 'flo_file')

    # Preprocess pipeline,
    filter_ROI_mask_with_GM = pe.Node(
        interface=IntersectMask(), name='filter_ROI_mask_with_GM')

    filter_ROI_mask_with_GM.inputs.filter_thr = filter_gm_threshold
    filter_ROI_mask_with_GM.inputs.background_val = background_val

    pipeline.connect(inputnode, 'ROI_mask_file',
                     filter_ROI_mask_with_GM, 'indexed_rois_file')
    pipeline.connect(inputnode, 'ROI_coords_file',
                     filter_ROI_mask_with_GM, 'coords_rois_file')
    pipeline.connect(inputnode, 'ROI_MNI_coords_file',
                     filter_ROI_mask_with_GM, 'MNI_coords_rois_file')
    pipeline.connect(inputnode, 'ROI_labels_file',
                     filter_ROI_mask_with_GM, 'labels_rois_file')

    if reslice:
        pipeline.connect(reslice_gm, 'out_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    elif resample:
        pipeline.connect(resample_gm, 'out_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    else:
        pipeline.connect(inputnode, 'gm_anat_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    # Nodes version: use min_BOLD_intensity and
    # return coords where signal is strong enough
    extract_mean_ROI_ts = pe.Node(interface=ExtractTS(
        plot_fig=plot), name='extract_mean_ROI_ts')

    extract_mean_ROI_ts.inputs.percent_signal = percent_signal
    extract_mean_ROI_ts.inputs.min_BOLD_intensity = min_BOLD_intensity

    pipeline.connect(inputnode, 'nii_4D_file', extract_mean_ROI_ts, 'file_4D')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_indexed_rois_file',
                     extract_mean_ROI_ts, 'indexed_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_MNI_coords_rois_file',
                     extract_mean_ROI_ts, 'MNI_coord_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_coords_rois_file',
                     extract_mean_ROI_ts, 'coord_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_labels_rois_file',
                     extract_mean_ROI_ts, 'label_rois_file')

    return pipeline


def create_pipeline_nii_to_conmat(
        main_path, filter_gm_threshold=0.9, pipeline_name="nii_to_conmat",
        conf_interval_prob=0.05, background_val=-1.0, plot=True,
        reslice=False, resample=False, min_BOLD_intensity=50,
        percent_signal=0.5):
    """
    Description:

    Pipeline from nifti 4D (after preprocessing) to connectivity matrices

    Inputs (inputnode):

        * nii_4D_file
        * rp_file
        * ROI_mask_file
        * gm_anat_file
        * wm_anat_file
        * csf_anat_file
        * ROI_coords_file
        * ROI_MNI_coords_file
        * ROI_labels_file

    Comments:

    Typically used after nipype preprocessing pipeline and
    before conmat_to_graph pipeline

    """
    if reslice and resample:
        print("Only reslice OR resample can be true, setting reslice to False")
        reslice = False

    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'nii_4D_file', 'ROI_mask_file', 'rp_file', 'gm_anat_file',
        'wm_anat_file', 'csf_anat_file', 'ROI_coords_file',
        'ROI_MNI_coords_file', 'ROI_labels_file']), name='inputnode')

    # reslice gm
    if reslice:

        reslice_gm = pe.Node(interface=spmu.Reslice(), name='reslice_gm')
        pipeline.connect(inputnode, 'ROI_mask_file', reslice_gm,
                         'space_defining')
        pipeline.connect(inputnode, 'gm_anat_file', reslice_gm, 'in_file')

    if resample:

        resample_gm = pe.Node(interface=RegResample(), name='resample_gm')
        pipeline.connect(inputnode, 'ROI_mask_file', resample_gm,
                         'ref_file')
        pipeline.connect(inputnode, 'gm_anat_file', resample_gm, 'flo_file')

    # reslice wm
    if reslice:

        reslice_wm = pe.Node(interface=spmu.Reslice(), name='reslice_wm')
        pipeline.connect(inputnode, 'ROI_mask_file', reslice_wm,
                         'space_defining')
        pipeline.connect(inputnode, 'wm_anat_file', reslice_wm, 'in_file')

    if resample:

        resample_wm = pe.Node(interface=RegResample(), name='resample_wm')
        pipeline.connect(inputnode, 'ROI_mask_file', resample_wm,
                         'ref_file')
        pipeline.connect(inputnode, 'wm_anat_file', resample_wm, 'flo_file')
    # reslice csf
    if reslice:

        reslice_csf = pe.Node(interface=spmu.Reslice(), name='reslice_csf')
        pipeline.connect(inputnode, 'ROI_mask_file', reslice_csf,
                         'space_defining')

        pipeline.connect(inputnode, 'csf_anat_file', reslice_csf, 'in_file')

    if resample:

        resample_csf = pe.Node(interface=RegResample(), name='resample_csf')
        pipeline.connect(inputnode, 'ROI_mask_file', resample_csf,
                         'ref_file')
        pipeline.connect(inputnode, 'csf_anat_file', resample_csf, 'flo_file')

    # Preprocess pipeline,
    filter_ROI_mask_with_GM = pe.Node(
        interface=IntersectMask(), name='filter_ROI_mask_with_GM')

    filter_ROI_mask_with_GM.inputs.filter_thr = filter_gm_threshold
    filter_ROI_mask_with_GM.inputs.background_val = background_val

    pipeline.connect(inputnode, 'ROI_mask_file',
                     filter_ROI_mask_with_GM, 'indexed_rois_file')
    pipeline.connect(inputnode, 'ROI_coords_file',
                     filter_ROI_mask_with_GM, 'coords_rois_file')
    pipeline.connect(inputnode, 'ROI_MNI_coords_file',
                     filter_ROI_mask_with_GM, 'MNI_coords_rois_file')
    pipeline.connect(inputnode, 'ROI_labels_file',
                     filter_ROI_mask_with_GM, 'labels_rois_file')

    if reslice:
        pipeline.connect(reslice_gm, 'out_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    elif resample:
        pipeline.connect(resample_gm, 'out_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    else:
        pipeline.connect(inputnode, 'gm_anat_file',
                         filter_ROI_mask_with_GM, 'filter_mask_file')

    # Nodes version: use min_BOLD_intensity and
    # return coords where signal is strong enough
    extract_mean_ROI_ts = pe.Node(interface=ExtractTS(
        plot_fig=plot), name='extract_mean_ROI_ts')

    extract_mean_ROI_ts.inputs.percent_signal = percent_signal
    extract_mean_ROI_ts.inputs.min_BOLD_intensity = min_BOLD_intensity

    pipeline.connect(inputnode, 'nii_4D_file', extract_mean_ROI_ts, 'file_4D')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_indexed_rois_file',
                     extract_mean_ROI_ts, 'indexed_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_MNI_coords_rois_file',
                     extract_mean_ROI_ts, 'MNI_coord_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_coords_rois_file',
                     extract_mean_ROI_ts, 'coord_rois_file')
    pipeline.connect(filter_ROI_mask_with_GM, 'filtered_labels_rois_file',
                     extract_mean_ROI_ts, 'label_rois_file')

    # extract white matter signal
    compute_wm_ts = pe.Node(interface=ExtractMeanTS(
        plot_fig=plot), name='extract_wm_ts')
    compute_wm_ts.inputs.suffix = 'wm'

    pipeline.connect(inputnode, 'nii_4D_file', compute_wm_ts, 'file_4D')

    if reslice:
        pipeline.connect(reslice_wm, 'out_file',
                         compute_wm_ts, 'filter_mask_file')

    elif resample:
        pipeline.connect(resample_wm, 'out_file',
                         compute_wm_ts, 'filter_mask_file')

    else:
        pipeline.connect(inputnode, 'wm_anat_file',
                         compute_wm_ts, 'filter_mask_file')

    # extract csf signal
    compute_csf_ts = pe.Node(interface=ExtractMeanTS(
        plot_fig=plot), name='extract_csf_ts')
    compute_csf_ts.inputs.suffix = 'csf'

    pipeline.connect(inputnode, 'nii_4D_file', compute_csf_ts, 'file_4D')

    if reslice:
        pipeline.connect(reslice_csf, 'out_file',
                         compute_csf_ts, 'filter_mask_file')

    elif resample:
        pipeline.connect(resample_csf, 'out_file',
                         compute_csf_ts, 'filter_mask_file')

    else:
        pipeline.connect(inputnode, 'csf_anat_file',
                         compute_csf_ts, 'filter_mask_file')

    # regress covariates
    regress_covar = pe.Node(interface=RegressCovar(plot_fig=plot), iterfield=[
        'masked_ts_file', 'rp_file', 'mean_wm_ts_file', 'mean_csf_ts_file'],
        name='regress_covar')

    pipeline.connect(extract_mean_ROI_ts, 'mean_masked_ts_file',
                     regress_covar, 'masked_ts_file')
    pipeline.connect(inputnode, 'rp_file', regress_covar, 'rp_file')

    pipeline.connect(compute_wm_ts, 'mean_masked_ts_file',
                     regress_covar, 'mean_wm_ts_file')
    pipeline.connect(compute_csf_ts, 'mean_masked_ts_file',
                     regress_covar, 'mean_csf_ts_file')

    # compute correlations
    compute_conf_cor_mat = pe.Node(interface=ComputeConfCorMat(
        plot_mat=plot), name='compute_conf_cor_mat')
    compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

    pipeline.connect(regress_covar, 'resid_ts_file',
                     compute_conf_cor_mat, 'ts_file')
    pipeline.connect(extract_mean_ROI_ts, 'subj_label_rois_file',
                     compute_conf_cor_mat, 'labels_file')

    return pipeline


def create_pipeline_nii_to_weighted_conmat(
        main_path, pipeline_name="nii_to_weighted_conmat",
        concatenated_runs=True, conf_interval_prob=0.05, mult_regnames=True,
        spm_reg=True):
    """
    Description:

    Pipeline from resid_ts_file (after preprocessing) to weighted connectivity
    matrices
    Involves a regressor file as wiehgt for computing weighted correlations

    Parameters:
        * main_path: path where the analysis will be located
        (base_dir of workflow)
        * pipeline_name (default = "nii_to_weighted_conmat"):
        name of the workflow that will be created for this analysis
        * concatenated_runs (default = True):
        If several sessions are contained in the same SPM.mat
        * conf_interval_prob (default = 0.05):
        default confidence interval value for thresholding connectivity matrix
        * mult_regnames (default = True):
        if reg_names is a list
        (instead of the instance of an iterable at the higher level)
        * spm_reg (default = True) : either use spm_mat_file or reg_txt
        (the latter containing directly the weighting use for computing
       weighted correlation)

    Inputs (inputnode):

        * resid_ts_file
        * spm_mat_file (from a typical SPM level1 activation analysis)
        * regress_names (names to look after in spm_mat_file)
        * run_index
        * ROI_labels_file
        * reg_txt (if spm_reg = False)

    Comments:

    Typically used after previous pipeline (create_pipeline_nii_to_conmat)
    and before conmat_to_graph pipeline

    """
    pipeline = pe.Workflow(name=pipeline_name)
    pipeline.base_dir = main_path

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'resid_ts_file', 'spm_mat_file', 'regress_names', 'run_index',
        'ROI_labels_file', 'reg_txt']), name='inputnode')

    if spm_reg:
        if mult_regnames:
            # extract regressor of interest from SPM.mat
            extract_cond = pe.MapNode(interface=FindSPMRegressor(
                only_positive_values=True), iterfield=['regressor_name'],
                name='extract_cond')

            pipeline.connect(inputnode, ('spm_mat_file', show_files),
                             extract_cond, 'spm_mat_file')
            pipeline.connect(inputnode, 'regress_names',
                             extract_cond, 'regressor_name')
            pipeline.connect(inputnode, 'run_index', extract_cond, 'run_index')

            # extract_cond.inputs.run_index = 0
            extract_cond.inputs.concatenated_runs = concatenated_runs

            # compute weighted correlations
            compute_conf_cor_mat = pe.MapNode(interface=ComputeConfCorMat(),
                                              iterfield=['weight_file'],
                                              name='compute_conf_cor_mat')
            # with confidence interval
            compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

            pipeline.connect(inputnode, 'resid_ts_file',
                             compute_conf_cor_mat, 'ts_file')
            pipeline.connect(extract_cond, 'regressor_file',
                             compute_conf_cor_mat, 'weight_file')

            pipeline.connect(inputnode, 'ROI_labels_file',
                             compute_conf_cor_mat, 'labels_file')

        else:

            # extract regressor of interest from SPM.mat
            extract_cond = pe.Node(interface=FindSPMRegressor(
                only_positive_values=True), name='extract_cond')

            pipeline.connect(inputnode, 'spm_mat_file',
                             extract_cond, 'spm_mat_file')
            pipeline.connect(inputnode, 'regress_names',
                             extract_cond, 'regressor_name')
            pipeline.connect(inputnode, 'run_index', extract_cond, 'run_index')

            extract_cond.inputs.concatenated_runs = concatenated_runs

            # compute weighted correlations

            # confidence interval

            compute_conf_cor_mat = pe.Node(
                interface=ComputeConfCorMat(), name='compute_conf_cor_mat')

            compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

            pipeline.connect(inputnode, 'resid_ts_file',
                             compute_conf_cor_mat, 'ts_file')
            pipeline.connect(extract_cond, 'regressor_file',
                             compute_conf_cor_mat, 'weight_file')

            pipeline.connect(inputnode, 'ROI_labels_file',
                             compute_conf_cor_mat, 'labels_file')
    else:
        if mult_regnames:
            compute_conf_cor_mat = pe.MapNode(interface=ComputeConfCorMat(),
                                              iterfield=['weight_file'],
                                              name='compute_conf_cor_mat')

            compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

            pipeline.connect(inputnode, 'resid_ts_file',
                             compute_conf_cor_mat, 'ts_file')
            pipeline.connect(inputnode, 'reg_txt',
                             compute_conf_cor_mat, 'weight_file')

            pipeline.connect(inputnode, 'ROI_labels_file',
                             compute_conf_cor_mat, 'labels_file')
        else:

            compute_conf_cor_mat = pe.Node(
                interface=ComputeConfCorMat(), name='compute_conf_cor_mat')

            compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob

            pipeline.connect(inputnode, 'resid_ts_file',
                             compute_conf_cor_mat, 'ts_file')
            pipeline.connect(inputnode, 'reg_txt',
                             compute_conf_cor_mat, 'weight_file')

            pipeline.connect(inputnode, 'ROI_labels_file',
                             compute_conf_cor_mat, 'labels_file')
    return pipeline
