"""
.. _dyn_graph_to_graph:

=================================================================
Compute connectivity matrices and graph properties from nii files
=================================================================

The nii_to_graph pipeline performs graph analysis from functional MRI file
in NIFTI format.

The **input** data should be preprocessed (i.e. realigned, coregistered, and segmented), and normalized in the same space (e.g. MNI
space) as the template used to define the nodes in the graph.

The data used in this example are the anat and func from the sub-01 in the  `OpenNeuro database ds000208_R1.0.0 <https://openneuro.org/datasets/ds000208/versions/00001>`_, after preprocessing realized with Nipype pipeline `create_preprocess_struct_to_mean_funct_4D_spm12 <https://github.com/davidmeunier79/nipype/blob/master/nipype/workflows/fmri/spm/preprocess.py>`_, with parameters:

* TR = 2.5,

* slice_timing = False

* fast_segmenting = True

* fwhm = [7.5,7.5,8]

* nb_scans_to_remove = 0

The template was generated from the HCP template called HCPMMP1_on_MNI152_ICBM2009a_nlin, by taking a mirror for the right hemisphere and compute a template with 360 ROIS - here 332 regions are kept, and time series length = 300

The **input** data should be a preprocessed, and in the same space (e.g. MNI
space) as the template used to define the nodes in the graph.
"""

# Authors: David Meunier <david_meunier_79@hotmail.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

import os
import os.path as op

import nipype.pipeline.engine as pe

from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio

import json  # noqa
import pprint  # noqa

###############################################################################

# Check if data are available

from graphpype.utils_tests import load_test_data

data_path = load_test_data("data_nii")

data_path_mask = load_test_data("data_nii_HCP")

ROI_mask_file = op.join(data_path_mask, "indexed_mask-ROI_HCP.nii")
ROI_coords_file = op.join(data_path_mask, "ROI_coords-ROI_HCP.txt")
ROI_MNI_coords_file =op.join(data_path_mask, "ROI_MNI_coords-ROI_HCP.txt")
ROI_labels_file = op.join(data_path_mask, "ROI_labels-ROI_HCP.txt")

###############################################################################
# Then, we create our workflow and specify the `base_dir` which tells
# nipype the directory in which to store the outputs.

# workflow directory within the `base_dir`
conmat_analysis_name = 'nii_to_dyn_graph'

#from graphpype.pipelines import create_pipeline_nii_to_split_conmat # noqa
from graphpype.pipelines import create_pipeline_nii_to_conmat # noqa

main_workflow = pe.Workflow(name= conmat_analysis_name)
main_workflow.base_dir = data_path

###############################################################################
# Then we create a node to pass input filenames to DataGrabber from nipype

data_dyn_graph = json.load(open(op.join(op.dirname("__file__"),"params_dyn_graph.json")))
pprint.pprint({'graph parameters': data_dyn_graph})

subject_ids = data_dyn_graph["subject_ids"]
func_sessions = data_dyn_graph["func_sessions"]
conf_interval_prob = data_dyn_graph["conf_interval_prob"]

infosource = pe.Node(interface=IdentityInterface(
    fields=['subject_id','session']),
    name="infosource")

infosource.iterables = [('subject_id', subject_ids),
    ('session', func_sessions)]

###############################################################################
# and a node to grab data. The template_args in this node iterate upon
# the values in the infosource node

datasource = pe.Node(interface=nio.DataGrabber(
    infields=['subject_id','session'],
    outfields= ['img_file','gm_anat_file','wm_anat_file','csf_anat_file']),
    name = 'datasource')

datasource.inputs.base_directory = data_path
datasource.inputs.template = '%ssub-%s%s%s%s'
datasource.inputs.template_args = dict(
img_file=[["wr",'subject_id',"_task-",'session',"_bold.nii"]],
gm_anat_file=[["rwc1",'subject_id',"",'',"_T1w.nii"]],
wm_anat_file=[["rwc2",'subject_id',"",'',"_T1w.nii"]],
csf_anat_file=[["rwc3",'subject_id',"",'',"_T1w.nii"]],
rp_file=[["rp_",'subject_id',"_task-",'session',"_bold.txt"]],
       )

datasource.inputs.sort_filelist = True

###############################################################################
win_length = data_dyn_graph["win_length"]
offset = data_dyn_graph["offset"]

### reasample images, extract time series and compute correlations
#cor_wf = create_pipeline_nii_to_split_conmat(main_path=data_path,
                                       #conf_interval_prob=conf_interval_prob,
                                       #win_length=win_length,
                                       #offset=offset,
                                       #resample=True, background_val=0.0)

cor_wf = create_pipeline_nii_to_conmat(main_path=data_path,
                                       conf_interval_prob=conf_interval_prob,
                                       resample=True, background_val=0.0,
                                       split=True, win_length=win_length,
                                       offset=offset)

### link the datasource outputs to the pipeline inputs
main_workflow.connect(datasource, 'img_file', cor_wf, 'inputnode.nii_4D_file')
main_workflow.connect(datasource, 'gm_anat_file', cor_wf,
                      'inputnode.gm_anat_file')
main_workflow.connect(datasource, 'wm_anat_file', cor_wf,
                      'inputnode.wm_anat_file')
main_workflow.connect(datasource, 'csf_anat_file', cor_wf,
                      'inputnode.csf_anat_file')
main_workflow.connect(datasource, 'rp_file', cor_wf, 'inputnode.rp_file')

### extra arguments: the template used to define nodes
cor_wf.inputs.inputnode.ROI_mask_file = ROI_mask_file
cor_wf.inputs.inputnode.ROI_coords_file = ROI_coords_file
cor_wf.inputs.inputnode.ROI_MNI_coords_file = ROI_MNI_coords_file
cor_wf.inputs.inputnode.ROI_labels_file = ROI_labels_file

###############################################################################
# We then connect the nodes two at a time. We connect the output
# of the infosource node to the datasource node.
# So, these two nodes taken together can grab data.

main_workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
main_workflow.connect(infosource, 'session', datasource, 'session')

################################################################################
#
## This parameter corrdesponds to the percentage of highest connections retains
## for the analyses. con_den = 1.0 means a fully connected graphs (all edges
## are present)

# density of the threshold
con_den = data_dyn_graph['con_den']

# The optimisation sequence
radatools_optim = data_dyn_graph['radatools_optim']

from graphpype.pipelines import create_pipeline_conmat_to_graph_density ## noqa

graph_workflow = create_pipeline_conmat_to_graph_density(
    data_path, con_den=con_den, optim_seq=radatools_optim, multi = True)

main_workflow.connect(cor_wf, 'compute_conf_cor_mat.Z_conf_cor_mat_file',
                      graph_workflow, "inputnode.conmat_file")

################################################################################
## To do so, we first write the workflow graph (optional)
#main_workflow.write_graph(graph2use='colored')  # colored

################################################################################
## and visualize it. Take a moment to pause and notice how the connections
## here correspond to how we connected the nodes.

#from scipy.misc import imread  # noqa
#import matplotlib.pyplot as plt  # noqa
#img = plt.imread(op.join(data_path, conmat_analysis_name, 'graph.png'))
#plt.figure(figsize=(8, 8))
#plt.imshow(img)
#plt.axis('off')
#plt.show()


################################################################################
## Finally, we are now ready to execute our workflow.
main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

main_workflow.run()


## Run workflow locally on 2 CPUs
#main_workflow.run(plugin='MultiProc', plugin_args={'n_procs': 2})

#################################################################################
### plotting

#from graphpype.utils_visbrain import visu_graph_modules, visu_graph_modules_roles
#from visbrain.objects import SceneObj, BrainObj # noqa

#sc = SceneObj(size=(1000, 1000), bgcolor=(1,1,1))

#res_path = op.join(
    #data_path, conmat_analysis_name,
    #"graph_den_pipe_den_"+str(con_den).replace(".", "_"),
    #"_session_rest_subject_id_01")

#lol_file = op.join(res_path, "community_rada", "Z_List.lol")
#net_file = op.join(res_path, "prep_rada", "Z_List.net")
#roles_file = op.join(res_path, "node_roles", "node_roles.txt")

#views = ["left",'top']

#for i_v,view in enumerate(views):

    #b_obj = BrainObj("B1", translucent=True)
    #sc.add_to_subplot(b_obj, row=0, col = i_v, use_this_cam=True, rotate=view,
                        #title=("Modules"),
                        #title_size=14, title_bold=True, title_color='black')

    #c_obj,s_obj = visu_graph_modules(lol_file=lol_file, net_file=net_file,
                                #coords_file=ROI_MNI_coords_file,
                                #inter_modules=False)

    #sc.add_to_subplot(c_obj, row=0, col = i_v)
    #sc.add_to_subplot(s_obj, row=0, col = i_v)

    #b_obj = BrainObj('B1', translucent=True)
    #sc.add_to_subplot(b_obj, row=1, col = i_v, use_this_cam=True, rotate=view,
                    #title=("Modules and node roles"),
                    #title_size=14, title_bold=True, title_color='black')

    #c_obj,list_sources = visu_graph_modules_roles(
        #lol_file=lol_file, net_file=net_file, roles_file=roles_file,
        #coords_file=ROI_MNI_coords_file, inter_modules=True, default_size=10,
        #hub_to_non_hub=3)

    #sc.add_to_subplot(c_obj, row=1, col = i_v)

    #for source in list_sources:
        #sc.add_to_subplot(source, row=1, col = i_v)



#sc.preview()
