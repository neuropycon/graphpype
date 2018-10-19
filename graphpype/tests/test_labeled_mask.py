from graphpype.labeled_mask import (segment_mask_in_ROI)

import os
import shutil

try:
    import neuropycon_data as nd

except ImportError:
    print("neuropycon_data not installed")


data_path = os.path.join(nd.__path__[0], "data", "data_nii")
mask_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")


def test_neuropycon_data():
    """test if neuropycon_data is installed"""
    assert os.path.exists(nd.__path__[0])
    assert os.path.exists(os.path.join(nd.__path__[0], 'data'))
    assert os.path.exists(os.path.join(nd.__path__[0], 'data', 'data_nii'))
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
        mask_thr=0.99)
    assert os.path.exists(indexed_mask_rois_file)
