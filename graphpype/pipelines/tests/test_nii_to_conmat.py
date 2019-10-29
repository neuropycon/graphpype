import os
"""
from graphpype.pipelines.nii_to_conmat import (
    create_pipeline_nii_to_conmat_simple,
    create_pipeline_nii_to_conmat_seg_template,
    create_pipeline_nii_to_subj_ROI, create_pipeline_nii_to_conmat)
"""
from graphpype.pipelines.nii_to_conmat import create_pipeline_nii_to_conmat
from graphpype.utils import _make_tmp_dir

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_nii")
nii_4D_file = os.path.join(data_path, "wrsub-01_task-rest_bold.nii")
gm_mask_file = os.path.join(data_path, "rwc1sub-01_T1w.nii")
wm_mask_file = os.path.join(data_path, "rwc2sub-01_T1w.nii")
csf_mask_file = os.path.join(data_path, "rwc3sub-01_T1w.nii")

data_path_HCP = load_test_data("data_nii_HCP")
indexed_mask_file = os.path.join(data_path_HCP, "indexed_mask-ROI_HCP.nii")
labels_file = os.path.join(data_path_HCP, "ROI_labels-ROI_HCP.txt")
coords_file = os.path.join(data_path_HCP, "ROI_coords-ROI_HCP.txt")
MNI_coords_file = os.path.join(data_path_HCP, "ROI_MNI_coords-ROI_HCP.txt")


def test_neuropycon_data():
    """test if neuropycon_data is installed"""
    assert os.path.exists(data_path)
    assert os.path.exists(nii_4D_file)
    assert os.path.exists(gm_mask_file)
    assert os.path.exists(wm_mask_file)
    assert os.path.exists(csf_mask_file)

    assert os.path.exists(data_path_HCP)
    assert os.path.exists(indexed_mask_file)


# Test is simple works with and without labels

"""
def test_nii_to_conmat_simple():

    wf = create_pipeline_nii_to_conmat_simple(
        main_path=data_path, pipeline_name="nii_to_conmat_simple",
        plot = False)

    wf.inputs.inputnode.nii_4D_file = nii_4D_file
    wf.inputs.inputnode.ROI_mask_file = indexed_mask_file

    # Optionnal labels (can be removed)
    wf.inputs.inputnode.ROI_labels_file = labels_file
    wf.inputs.inputnode.ROI_coords_file = coords_file
    wf.inputs.inputnode.ROI_MNI_coords_file = MNI_coords_file

    # Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs": False}

    wf.run(plugin='MultiProc', plugin_args={'n_procs' : 1})

#test with white matter and csf from segmented files.
#Files should have the same dimensions


def test_create_pipeline_nii_to_conmat_seg_template():

    wf = create_pipeline_nii_to_conmat_seg_template(
        main_path=data_path, pipeline_name="nii_to_conmat_seg_template")

    wf.inputs.inputnode.nii_4D_file = nii_4D_file
    wf.inputs.inputnode.ROI_mask_file = indexed_mask_file

    wf.inputs.inputnode.ROI_labels_file = labels_file
    wf.inputs.inputnode.ROI_coords_file = coords_file
    wf.inputs.inputnode.ROI_MNI_coords_file = MNI_coords_file

    wf.inputs.inputnode.wm_anat_file = wm_anat_file
    wf.inputs.inputnode.csf_anat_file = csf_anat_file

    # Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs": False}

    wf.run()

# including a step of selection of ROIs in the Grey matter mask
# (intersect mask)
# test = if gm_anat_file = wm_anat_file,
# crashes because BOLD signal is not enough!

def test_create_pipeline_nii_to_subj_ROI():

    wf = create_pipeline_nii_to_subj_ROI(main_path = data_path,
        pipeline_name = "nii_to_conmat_seg_template")

    wf.inputs.inputnode.nii_4D_file = nii_4D_file
    wf.inputs.inputnode.ROI_mask_file = indexed_mask_file

    wf.inputs.inputnode.gm_anat_file = wm_anat_file

    #Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs":False}

    wf.run()

# subject time series only (no regression of wm and csf signals /
# no correlation matrices)


def test_create_pipeline_nii_to_subj_ROI():

    wf = create_pipeline_nii_to_subj_ROI(
        main_path=data_path, pipeline_name="nii_to_subj_ROI")

    wf.inputs.inputnode.nii_4D_file = nii_4D_file
    wf.inputs.inputnode.ROI_mask_file = indexed_mask_file

    wf.inputs.inputnode.ROI_labels_file = labels_file
    wf.inputs.inputnode.ROI_coords_file = coords_file
    wf.inputs.inputnode.ROI_MNI_coords_file = MNI_coords_file

    wf.inputs.inputnode.gm_anat_file = gm_anat_file

    # Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs": False}

    wf.run()
"""


# the full pipeline
def test_create_pipeline_nii_to_conmat():
    tmp_dir = _make_tmp_dir()
    wf = create_pipeline_nii_to_conmat(
        main_path=tmp_dir, pipeline_name="nii_to_conmat_full")

    wf.inputs.inputnode.nii_4D_file = nii_4D_file
    wf.inputs.inputnode.ROI_mask_file = indexed_mask_file

    wf.inputs.inputnode.gm_anat_file = gm_mask_file
    wf.inputs.inputnode.wm_anat_file = wm_mask_file
    wf.inputs.inputnode.csf_anat_file = csf_mask_file

    # Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs": False}

    wf.run()
