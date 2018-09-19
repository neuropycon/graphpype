import os
"""
from graphpype.pipelines.nii_to_conmat import (
    create_pipeline_nii_to_conmat_simple,
    create_pipeline_nii_to_conmat_seg_template,
    create_pipeline_nii_to_subj_ROI, create_pipeline_nii_to_conmat)
"""
from graphpype.pipelines.nii_to_conmat import create_pipeline_nii_to_conmat
from graphpype.utils import _make_tmp_dir

try:
    import neuropycon_data as nd
except ImportError:
    print("Warning, neuropycon_data not found")
    exit()

data_path = os.path.join(nd.__path__[0], "data", "data_nii")


# Mandatory for create_pipeline_nii_to_conmat_simple
# 4D functionnal MRI volumes
nii_4D_file = os.path.join(data_path, "sub-test_task-rs_bold.nii")

# indexed ROI mask - should be the same dimestions as the functional data
indexed_mask_file = os.path.join(data_path, "Atlas", "indexed_mask-Atlas.nii")

# Optionnal for create_pipeline_nii_to_conmat_simple
labels_file = os.path.join(data_path, "Atlas", "ROI_labels-Atlas.txt")
coords_file = os.path.join(data_path, "Atlas", "ROI_coords-Atlas.txt")
MNI_coords_file = os.path.join(data_path, "Atlas", "ROI_MNI_coords-Atlas.txt")

# mandatory for create_pipeline_nii_to_conmat_seg_template
wm_anat_file = os.path.join(data_path, "sub-test_mask-anatWM.nii")
csf_anat_file = os.path.join(data_path, "sub-test_mask-anatCSF.nii")
gm_anat_file = os.path.join(data_path, "sub-test_mask-anatGM.nii")

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

    wf.inputs.inputnode.gm_anat_file = gm_anat_file
    wf.inputs.inputnode.wm_anat_file = wm_anat_file
    wf.inputs.inputnode.csf_anat_file = csf_anat_file

    # Warning, is necessary, otherwise Figures are removed!
    wf.config['execution'] = {"remove_unnecessary_outputs": False}

    wf.run()
