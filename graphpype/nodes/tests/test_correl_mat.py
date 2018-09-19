import os
from graphpype.nodes.correl_mat import (ExtractTS, IntersectMask,
                                        ExtractMeanTS,
                                        RegressCovar,
                                        ComputeConfCorMat)
from graphpype.utils import _make_tmp_dir

try:
    import neuropycon_data as nd

except ImportError:
    print("neuropycon_data not installed")
    exit()

data_path = os.path.join(nd.__path__[0], "data", "data_nii")
img_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")
gm_mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")
wm_mask_file = os.path.join(data_path, "sub-test_mask-anatWM.nii")
csf_mask_file = os.path.join(data_path, "sub-test_mask-anatCSF.nii")
indexed_mask_file = os.path.join(data_path, "Atlas", "indexed_mask-Atlas.nii")


def test_extract_ts():
    """ test ExtractTS"""
    _make_tmp_dir()

    extra_ts = ExtractTS()
    extra_ts.inputs.indexed_rois_file = indexed_mask_file
    extra_ts.inputs.file_4D = img_file

    val = extra_ts.run().outputs
    print(val)
    assert os.path.exists(val.mean_masked_ts_file)
    os.remove(val.mean_masked_ts_file)


def test_intersect_mask():
    """test IntersectMask"""
    # TODO _make_tmp_dir()
    intersect_mask = IntersectMask()
    intersect_mask.inputs.indexed_rois_file = indexed_mask_file
    intersect_mask.inputs.filter_mask_file = gm_mask_file

    val = intersect_mask.run().outputs
    print(val)
    assert os.path.exists(val.filtered_indexed_rois_file)
    os.remove(val.filtered_indexed_rois_file)


def test_extract_mean_ts():
    """test ExtractMeanTS"""
    _make_tmp_dir()

    extract_mean_ts = ExtractMeanTS()
    extract_mean_ts.inputs.file_4D = img_file
    extract_mean_ts.inputs.mask_file = wm_mask_file
    extract_mean_ts.inputs.suffix = "wm"

    val = extract_mean_ts.run().outputs
    print(val)
    assert os.path.exists(val.mean_masked_ts_file)
    os.remove(val.mean_masked_ts_file)
    # TODO raise Error if incompatible images


def test_regress_covar():
    """ test regress_covar"""
    _make_tmp_dir()

    extract_mean_wm_ts = ExtractMeanTS()
    extract_mean_wm_ts.inputs.file_4D = img_file
    extract_mean_wm_ts.inputs.mask_file = wm_mask_file
    extract_mean_wm_ts.inputs.suffix = "wm"
    mean_wm_ts_file = extract_mean_wm_ts.run().outputs.mean_masked_ts_file

    extract_mean_csf_ts = ExtractMeanTS()
    extract_mean_csf_ts.inputs.file_4D = img_file
    extract_mean_csf_ts.inputs.mask_file = csf_mask_file
    extract_mean_csf_ts.inputs.suffix = "csf"
    mean_csf_ts_file = extract_mean_csf_ts.run().outputs.mean_masked_ts_file

    extra_ts = ExtractTS()
    extra_ts.inputs.indexed_rois_file = indexed_mask_file
    extra_ts.inputs.file_4D = img_file
    masked_ts_file = extra_ts.run().outputs.mean_masked_ts_file

    regress_covar = RegressCovar()
    regress_covar.inputs.masked_ts_file = masked_ts_file
    regress_covar.inputs.mean_wm_ts_file = mean_wm_ts_file
    regress_covar.inputs.mean_csf_ts_file = mean_csf_ts_file

    val = regress_covar.run().outputs
    print(val)

    os.remove(val.resid_ts_file)

    os.remove(masked_ts_file)
    os.remove(mean_csf_ts_file)
    os.remove(mean_wm_ts_file)


def test_compute_conf_cor_mat():
    """test ComputeConfCorMat"""
    _make_tmp_dir()

    extract_mean_wm_ts = ExtractMeanTS()
    extract_mean_wm_ts.inputs.file_4D = img_file
    extract_mean_wm_ts.inputs.mask_file = wm_mask_file
    extract_mean_wm_ts.inputs.suffix = "wm"
    mean_wm_ts_file = extract_mean_wm_ts.run().outputs.mean_masked_ts_file

    extract_mean_csf_ts = ExtractMeanTS()
    extract_mean_csf_ts.inputs.file_4D = img_file
    extract_mean_csf_ts.inputs.mask_file = csf_mask_file
    extract_mean_csf_ts.inputs.suffix = "csf"
    mean_csf_ts_file = extract_mean_csf_ts.run().outputs.mean_masked_ts_file

    extra_ts = ExtractTS()
    extra_ts.inputs.indexed_rois_file = indexed_mask_file
    extra_ts.inputs.file_4D = img_file
    masked_ts_file = extra_ts.run().outputs.mean_masked_ts_file

    regress_covar = RegressCovar()
    regress_covar.inputs.masked_ts_file = masked_ts_file
    regress_covar.inputs.mean_wm_ts_file = mean_wm_ts_file
    regress_covar.inputs.mean_csf_ts_file = mean_csf_ts_file

    resid_ts_file = regress_covar.run().outputs.resid_ts_file

    compute_conf_cor_mat = ComputeConfCorMat()
    compute_conf_cor_mat.inputs.ts_file = resid_ts_file

    val = compute_conf_cor_mat.run().outputs
    print(val)

    assert os.path.exists(val.cor_mat_file)
    assert os.path.exists(val.Z_cor_mat_file)
    assert os.path.exists(val.conf_cor_mat_file)
    assert os.path.exists(val.Z_conf_cor_mat_file)

    os.remove(val.cor_mat_file)
    os.remove(val.conf_cor_mat_file)
    os.remove(val.Z_cor_mat_file)
    os.remove(val.Z_conf_cor_mat_file)

    os.remove(resid_ts_file)
    os.remove(masked_ts_file)
    os.remove(mean_csf_ts_file)
    os.remove(mean_wm_ts_file)
