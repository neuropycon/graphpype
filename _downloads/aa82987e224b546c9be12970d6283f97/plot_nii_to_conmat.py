"""
.. _nii_to_conmat:

=========================================================
Compute Conmat from nii files properties from a given connectivity matrix
=========================================================
The conmat_to_graph pipeline performs graph analysis .

The **input** data should be a symetrical connecivity matrix in **npy** format.
"""

# Authors: David Meunier <david_meunier_79@hotmail.fr>

# License: BSD (3-clause)
import os.path as op

import nipype.pipeline.engine as pe

from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio

###############################################################################
# Check if data are available
# needs to import neuropycon_data
# 'pip install neuropycon_data' should do the job...


try:
    import neuropycon_data as nd
except ImportError:
    print("Warning, neuropycon_data not found")
    exit()

data_path = op.join(nd.__path__[0], "data", "data_nii")

###############################################################################
# Then, we create our workflow and specify the `base_dir` which tells
# nipype the directory in which to store the outputs.

# workflow directory within the `base_dir`
conmat_analysis_name = 'conmat'

rwm_mask_file = op.join(data_path,"sub-test_mask-anatWM.nii")
rcsf_mask_file = op.join(data_path,"sub-test_mask-anatCSF.nii")

ROI_mask_file = op.join(data_path,"Atlas","indexed_mask-Atlas.nii")
ROI_coords_file = op.join(data_path,"Atlas","ROI_coords-Atlas.txt")
ROI_MNI_coords_file =op.join(data_path,"Atlas","ROI_MNI_coords-Atlas.txt")
ROI_labels_file = op.join(data_path,"Atlas","ROI_labels-Atlas.txt")

###############################################################################
# Then we create a node to pass input filenames to DataGrabber from nipype

subject_ids = ['test']
func_sessions = ['rs']
infosource = pe.Node(interface=IdentityInterface(
    fields=['subject_id','session']),
    name="infosource")

infosource.iterables = [('subject_id', subject_ids),
    ('session', func_sessions)]

###############################################################################
# and a node to grab data. The template_args in this node iterate upon
# the values in the infosource node

# template_path = '*%s/conmat_0_coh.npy'
# template_args = [['freq_band_name']
# datasource = create_datagrabber(data_path, template_path, template_args)

datasource = pe.Node(
    interface=nio.DataGrabber(infields=['subject_id','session'],
                              outfields=['img_file']),
    name='datasource')

datasource = pe.Node(interface=nio.DataGrabber(
    infields=['subject_id','session'],
    outfields= ['img_file']),
    name = 'datasource')

datasource.inputs.base_directory = data_path
datasource.inputs.template = 'sub-%s_task-%s_bold.nii'
datasource.inputs.template_args = dict(

img_file=[['subject_id','session']],

       )
datasource.inputs.sort_filelist = True

#0/0
################################################################################
## This parameter corrdesponds to the percentage of highest connections retains
## for the analyses. con_den = 1.0 means a fully connected graphs (all edges
## are present)

#import json  # noqa
#import pprint  # noqa

#data_graph = json.load(open("params_graph.json"))
#pprint.pprint({'graph parameters': data_graph})

## density of the threshold
#con_den = data_graph['con_den']

## The optimisation sequence
#radatools_optim = data_graph['radatools_optim']

from graphpype.pipelines.nii_to_conmat import create_pipeline_nii_to_conmat_seg_template # noqa


main_workflow = pe.Workflow(name= conmat_analysis_name)
main_workflow.base_dir = data_path

conf_interval_prob = 0.05

    ###### time series and correlations
cor_wf = create_pipeline_nii_to_conmat_seg_template(main_path =
data_path, conf_interval_prob = conf_interval_prob)

main_workflow.connect(datasource,'img_file',
cor_wf,'inputnode.nii_4D_file')
#main_workflow.connect(datasource, 'rp_file', cor_wf,'inputnode.rp_file')

cor_wf.inputs.inputnode.wm_anat_file = rwm_mask_file
cor_wf.inputs.inputnode.csf_anat_file = rcsf_mask_file

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
## To do so, we first write the workflow graph (optional)
#main_workflow.write_graph(graph2use='colored')  # colored

################################################################################
## and visualize it. Take a moment to pause and notice how the connections
## here correspond to how we connected the nodes.

#from scipy.misc import imread  # noqa
#import matplotlib.pyplot as plt  # noqa
#img = plt.imread(op.join(data_path, graph_analysis_name, 'graph.png'))
#plt.figure(figsize=(8, 8))
#plt.imshow(img)
#plt.axis('off')
#plt.show()

###############################################################################
# Finally, we are now ready to execute our workflow.
main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

main_workflow.run()

# Run workflow locally on 2 CPUs
#main_workflow.run(plugin='MultiProc', plugin_args={'n_procs': 2})

################################################################################
## plotting

#from graphpype.utils_visbrain import visu_graph_modules # noqa

#labels_file = op.join(data_path, "correct_channel_names.txt")
#coords_file = op.join(data_path, "MNI_coords.txt")

#from visbrain.objects import SceneObj, BrainObj # noqa

#sc = SceneObj(size=(1000, 1000), bgcolor=(.1, .1, .1))

#for nf, freq_band_name in enumerate(freq_band_names):

    #res_path = op.join(
        #data_path, graph_analysis_name,
        #"graph_den_pipe_den_"+str(con_den).replace(".", "_"),
        #"_freq_band_name_"+freq_band_name)

    #lol_file = op.join(res_path, "community_rada", "Z_List.lol")
    #net_file = op.join(res_path, "prep_rada", "Z_List.net")

    #b_obj = BrainObj("white", translucent=False)
    #sc.add_to_subplot(
        #b_obj, row=nf, use_this_cam=True, rotate='left',
        #title=("Module for {} band".format(freq_band_name)),
        #title_size=14, title_bold=True, title_color='white')

    #c_obj,s_obj = visu_graph_modules(lol_file=lol_file, net_file=net_file,
                               #coords_file=coords_file,
                               #labels_file=labels_file, inter_modules=False,
                               #z_offset=+50)
    #sc.add_to_subplot(c_obj, row=nf)
    #sc.add_to_subplot(s_obj, row=nf)


#sc.preview()