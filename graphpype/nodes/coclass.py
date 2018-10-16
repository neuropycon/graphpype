"""
Definition of nodes for computing reordering and plotting coclass_matrices
"""
import numpy as np
import os

from nipype.utils.filemanip import split_filename as split_f


from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec, isdefined)

from graphpype.utils_cor import return_coclass_mat, return_coclass_mat_labels
from graphpype.utils_net import read_Pajek_corres_nodes, read_lol_file


from graphpype.utils import check_np_shapes
from graphpype.utils_plot import plot_ranged_cormat

# PrepareCoclass


class PrepareCoclassInputSpec(BaseInterfaceInputSpec):

    mod_files = traits.List(
        File(exists=True), desc='list of all files representing modularity \
        assignement (in rada, lol files) for each subject', mandatory=True)

    node_corres_files = traits.List(
        File(exists=True), desc='list of all Pajek files (in txt format) to \
        extract correspondance between nodes in rada analysis and original \
        subject coordinates for each subject - as obtained from PrepRada',
        mandatory=True)

    coords_files = traits.List(File(
        exists=True), desc='list of all coordinates in numpy space files (in\
        txt format) for each subject (after removal of non void data)',
        mandatory=True, xor=['labels_files'])

    labels_files = traits.List(File(
        exists=True), desc='list of labels (in txt format) for each subject\
        (after removal of non void data)', mandatory=True,
        xor=['coords_files'])

    gm_mask_coords_file = File(
        exists=True, desc='Coordinates in numpy space, corresponding to all\
        possible nodes in the original space', mandatory=False,
        xor=['gm_mask_labels_file'])

    gm_mask_labels_file = File(
        exists=True, desc='Labels for all possible nodes - in case coords are\
        varying from one indiv to the other (source space for example)',
        mandatory=False, xor=['gm_mask_coords_file'])


class PrepareCoclassOutputSpec(TraitedSpec):

    group_coclass_matrix_file = File(
        exists=True, desc="all coclass matrices of the group in .npy (pickle\
        format)")

    sum_coclass_matrix_file = File(
        exists=True, desc="sum of coclass matrix of the group in .npy (pickle\
        format)")

    sum_possible_edge_matrix_file = File(
        exists=True, desc="sum of possible edges matrices of the group in .npy\
        (pickle format)")

    norm_coclass_matrix_file = File(
        exists=True, desc="sum of coclass matrix normalized by possible edges\
        matrix of the group in .npy (pickle format)")


class PrepareCoclass(BaseInterface):

    """
    Prepare a list of coclassification matrices, in a similar reference given
    by a coord (resp label)  file based on individual coords (resp labels)
    files

    Inputs:

        mod_files:
            type = List of Files, exists=True, desc='list of all files
            representing modularity assignement (in rada, lol files) for each
            subject', mandatory=True

        node_corres_files:
            type = List of Files, exists=True, desc='list of all Pajek files
            (in txt format) to extract correspondance between nodes in rada
            analysis and original subject coordinates for each subject (as
            obtained from PrepRada)', mandatory=True

        coords_files:
            type = List of Files, exists=True, desc='list of all coordinates in
            numpy space files (in txt format) for each subject (after removal
            of non void data)', mandatory=True, xor = ['labels_files']

        gm_mask_coords_file:
            type = File,exists=True, desc='Coordinates in numpy space,
            corresponding to all possible nodes in the original space',
            mandatory=False, xor = ['gm_mask_labels_file']

        labels_files:
            type = List of Files, exists=True, desc='list of labels (in txt
            format) for each subject (after removal of non void data)',
            mandatory=True, xor = ['coords_files']

        gm_mask_labels_file:
            type = File, exists=True, desc='Labels for all possible nodes - in
            case coords are varying from one indiv to the other (source space
            for example)', mandatory=False, xor = ['gm_mask_coords_file']

    Outputs:

        group_coclass_matrix_file:
            type = File,exists=True, desc="all coclass matrices of the group
            in .npy  format"

        sum_coclass_matrix_file:
            type = File, exists=True, desc="sum of coclass matrix of the group
            in .npy format"

        sum_possible_edge_matrix_file:
            type = File, exists=True, desc="sum of possible edges matrices of
            the group in .npy format"

        norm_coclass_matrix_file:
            type = File, exists=True, desc="sum of coclass matrix normalized
            by possible edges matrix of the group in .npy format"



    """
    input_spec = PrepareCoclassInputSpec
    output_spec = PrepareCoclassOutputSpec

    def _run_interface(self, runtime):

        print('in prepare_coclass')
        mod_files = self.inputs.mod_files

        node_corres_files = self.inputs.node_corres_files

        if isdefined(self.inputs.gm_mask_coords_file) and \
                isdefined(self.inputs.coords_files):

            coords_files = self.inputs.coords_files

            gm_mask_coords = np.loadtxt(self.inputs.gm_mask_coords_file)

            print(gm_mask_coords.shape)

            # read matrix from the first group
            # print Z_cor_mat_files

            sum_coclass_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)
            sum_possible_edge_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=int)

            # print sum_coclass_matrix.shape

            group_coclass_matrix = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0],
                 len(mod_files)), dtype=float)

            print(group_coclass_matrix.shape)

            assert len(mod_files) == len(coords_files) and len(mod_files) == \
                len(node_corres_files), (
                    "Error, length of mod_files, coords_files and \
                    node_corres_files are imcompatible {} {} {}".format(
                        len(mod_files), len(coords_files),
                        len(node_corres_files)))

            for index_file in range(len(mod_files)):
                if (os.path.exists(mod_files[index_file]) and
                        os.path.exists(node_corres_files[index_file]) and
                        os.path.exists(coords_files[index_file])):

                    community_vect = read_lol_file(mod_files[index_file])

                    node_corres_vect = read_Pajek_corres_nodes(
                        node_corres_files[index_file])
                    coords = np.loadtxt(coords_files[index_file])

                    corres_coords = coords[node_corres_vect, :]
                    coclass_mat, possible_edge_mat = return_coclass_mat(
                        community_vect, corres_coords, gm_mask_coords)

                    np.fill_diagonal(coclass_mat, 0)
                    np.fill_diagonal(possible_edge_mat, 1)
                    sum_coclass_matrix += coclass_mat
                    sum_possible_edge_matrix += possible_edge_mat
                    group_coclass_matrix[:, :, index_file] = coclass_mat

                else:
                    print("Warning, one or more files between {}, {}, {} do\
                        not exists".format(mod_files[index_file],
                                           node_corres_files[index_file],
                                           coords_files[index_file]))

        elif (isdefined(self.inputs.gm_mask_labels_file) and
                isdefined(self.inputs.labels_files)):

            labels_files = self.inputs.labels_files
            gm_mask_labels_file = self.inputs.gm_mask_labels_file

            gm_mask_labels = np.array(
                [line.strip() for line in open(gm_mask_labels_file)],
                dtype='str')

            print(gm_mask_labels.shape)

            sum_coclass_matrix = np.zeros(
                (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)
            sum_possible_edge_matrix = np.zeros(
                (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=int)

            # print sum_coclass_matrix.shape

            group_coclass_matrix = np.zeros(
                (gm_mask_labels.shape[0], gm_mask_labels.shape[0],
                 len(mod_files)), dtype=float)

            print(group_coclass_matrix.shape)

            assert len(mod_files) == len(labels_files) and len(mod_files) == \
                len(node_corres_files), (
                    "Error, length of mod_files, labels_files and \
                    node_corres_files are imcompatible {} {} {}".format(
                        len(mod_files), len(labels_files),
                        len(node_corres_files)))

            for index_file in range(len(mod_files)):
                if os.path.exists(mod_files[index_file]) and \
                        os.path.exists(node_corres_files[index_file]) and\
                        os.path.exists(labels_files[index_file]):

                    community_vect = read_lol_file(mod_files[index_file])
                    node_corres_vect = read_Pajek_corres_nodes(
                        node_corres_files[index_file])
                    labels = np.array([line.strip() for line in open(
                        labels_files[index_file])], dtype='str')
                    corres_labels = labels[node_corres_vect]
                    coclass_mat, possible_edge_mat = return_coclass_mat_labels(
                        community_vect, corres_labels, gm_mask_labels)

                    np.fill_diagonal(coclass_mat, 0)
                    np.fill_diagonal(possible_edge_mat, 1)

                    sum_coclass_matrix += coclass_mat
                    sum_possible_edge_matrix += possible_edge_mat
                    group_coclass_matrix[:, :, index_file] = coclass_mat

                else:
                    print("Warning, one or more files between {}, {}, {} do\
                        not exists".format(mod_files[index_file],
                                           node_corres_files[index_file],
                                           labels_files[index_file]))

        else:
            print("Error, gm_mask_coords_file XOR gm_mask_labels_file should\
                be defined")
            return

        group_coclass_matrix_file = os.path.abspath('group_coclass_matrix.npy')

        np.save(group_coclass_matrix_file, group_coclass_matrix)

        print('saving coclass matrix')

        sum_coclass_matrix_file = os.path.abspath('sum_coclass_matrix.npy')

        np.save(sum_coclass_matrix_file, sum_coclass_matrix)

        print('saving possible_edge matrix')

        sum_possible_edge_matrix_file = os.path.abspath(
            'sum_possible_edge_matrix.npy')

        np.save(sum_possible_edge_matrix_file, sum_possible_edge_matrix)

        # save norm_coclass_matrix
        print()

        print(np.where(np.array(sum_possible_edge_matrix == 0)))

        norm_coclass_matrix = np.divide(
            np.array(sum_coclass_matrix, dtype=float),
            np.array(sum_possible_edge_matrix, dtype=float)) * 100

        # 0/0

        print('saving norm coclass matrix')

        norm_coclass_matrix_file = os.path.abspath('norm_coclass_matrix.npy')

        np.save(norm_coclass_matrix_file, norm_coclass_matrix)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["group_coclass_matrix_file"] = os.path.abspath(
            'group_coclass_matrix.npy')

        outputs["sum_coclass_matrix_file"] = os.path.abspath(
            'sum_coclass_matrix.npy')

        outputs["sum_possible_edge_matrix_file"] = os.path.abspath(
            'sum_possible_edge_matrix.npy')

        outputs["norm_coclass_matrix_file"] = os.path.abspath(
            'norm_coclass_matrix.npy')

        return outputs


# DiffMatrices


class DiffMatricesInputSpec(BaseInterfaceInputSpec):

    mat_file1 = File(exists=True, desc='Matrix in npy format', mandatory=True)
    mat_file2 = File(exists=True, desc='Matrix in npy format', mandatory=True)


class DiffMatricesOutputSpec(TraitedSpec):

    diff_mat_file = File(
        exists=True,
        desc='Difference of Matrices (mat1 - mat2) in npy format',
        mandatory=True)


class DiffMatrices(BaseInterface):

    """
    Description:

    Compute difference between two matrices, should have same shape

    Inputs:

        mat_file1:
            type = File,exists=True, desc='Matrix in npy format',
            mandatory=True

        mat_file2:
            type = File,exists=True, desc='Matrix in npy format',
            mandatory=True

    Outputs:
        diff_mat_file:
            type = File, exists=True,
            desc='Difference of Matrices (mat1 - mat2) in npy format',
            mandatory=True

    Comments:

    Not sure where it is used ...

    """
    input_spec = DiffMatricesInputSpec
    output_spec = DiffMatricesOutputSpec

    def _run_interface(self, runtime):

        mat_file1 = self.inputs.mat_file1
        mat_file2 = self.inputs.mat_file2

        mat1 = np.load(mat_file1)
        print(mat1.shape)

        mat2 = np.load(mat_file2)
        print(mat2.shape)

        assert check_np_shapes(mat1.shape, mat2.shape), ("Warning, shapes are \
            different, cannot substrat matrices")

        diff_mat = mat1 - mat2
        print(diff_mat)

        diff_mat_file = os.path.abspath("diff_matrix.npy")

        np.save(diff_mat_file, diff_mat)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["diff_mat_file"] = os.path.abspath("diff_matrix.npy")
        return outputs


# PlotCoclass


class PlotCoclassInputSpec(BaseInterfaceInputSpec):

    coclass_matrix_file = File(
        exists=True,  desc='coclass matrix in npy format', mandatory=True)

    labels_file = File(exists=True,  desc='labels of nodes', mandatory=False)

    list_value_range = traits.ListInt(
        desc='force the range of the plot', mandatory=False)


class PlotCoclassOutputSpec(TraitedSpec):

    plot_coclass_matrix_file = File(
        exists=True, desc="eps file with graphical representation")


class PlotCoclass(BaseInterface):

    """
    Description :

    Plot coclass matrix with matplotlib matshow

    - labels are optional
    - range values are optional (default is min and max values of the matrix)

    Inputs:

        coclass_matrix_file:
            type = File, exists=True,  desc='coclass matrix in npy format',
            mandatory=True

        labels_file:
            type = File, exists=True,  desc='labels of nodes', mandatory=False

        list_value_range
            type = ListInt, desc='force the range of the plot', mandatory=False

    Outputs:

        plot_coclass_matrix_file:
            type = File, exists=True,
            desc="eps file with graphical representation"


    """
    input_spec = PlotCoclassInputSpec
    output_spec = PlotCoclassOutputSpec

    def _run_interface(self, runtime):

        coclass_matrix_file = self.inputs.coclass_matrix_file
        labels_file = self.inputs.labels_file
        list_value_range = self.inputs.list_value_range

        coclass_mat = np.load(coclass_matrix_file)

        if isdefined(labels_file):

            labels = [line.strip() for line in open(labels_file)]

        else:
            labels = []

        if not isdefined(list_value_range):
            list_value_range = [np.amin(coclass_mat), np.amax(coclass_mat)]

        path, fname, ext = split_f(coclass_matrix_file)
        plot_coclass_matrix_file = os.path.abspath('heatmap_' + fname + '.eps')
        plot_ranged_cormat(plot_coclass_matrix_file, coclass_mat,
                           labels, fix_full_range=list_value_range)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        path, fname, ext = split_f(self.inputs.coclass_matrix_file)
        outputs["plot_coclass_matrix_file"] = os.path.abspath(
            'heatmap_' + fname + '.eps')
        return outputs
