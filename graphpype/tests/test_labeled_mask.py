from graphpype.labeled_mask import (segment_mask_in_ROI)

import os
import shutil

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_nii_mask")
mask_file = os.path.join(data_path, "rwc1sub-01_T1w.nii")


def test_data():
    """test if test_data is accessible"""
    assert os.path.exists(data_path)
    assert os.path.exists(mask_file)


nb_ROIs = 10

tmp_dir = "/tmp/test_graphpype"

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

os.makedirs(tmp_dir)


def test_segment_mask_in_ROI():
    """test_segment_mask_in_ROI"""
    # with cube (default)
    indexed_mask_rois_file, _, _ = segment_mask_in_ROI(
        mask_file, save_dir=tmp_dir, segment_type="cube", mask_thr=0.99)
    assert os.path.exists(indexed_mask_rois_file)

    # with disjoint_comp
    indexed_mask_rois_file, _, _ = segment_mask_in_ROI(
        mask_file, save_dir=tmp_dir, segment_type="disjoint_comp",
        mask_thr=0.99, min_count_voxel_in_ROI=40)
    assert os.path.exists(indexed_mask_rois_file)
