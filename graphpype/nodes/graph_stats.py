"""
Preparing, computing stats and permutations
author:: David Meunier <david_meunier_79@univ-amu.fr>
"""
import numpy as np
import os

import itertools as iter

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined


from nipype.utils.filemanip import split_filename as split_f


import graphpype.utils_stats as stats

from graphpype.utils_cor import (return_corres_correl_mat,
                                 return_corres_correl_mat_labels)


# StatsPairBinomial


class StatsPairBinomialInputSpec(BaseInterfaceInputSpec):

    group_coclass_matrix_file1 = File(
        exists=True,
        desc='file of group 1 coclass matrices in npy format',
        mandatory=True)

    group_coclass_matrix_file2 = File(
        exists=True,
        desc='file of group 2 coclass matrices in npy format',
        mandatory=True)

    conf_interval_binom_fdr = traits.Float(
        0.05,
        usedefault=True,
        desc='Alpha value used as FDR implementation',
        mandatory=False)


class StatsPairBinomialOutputSpec(TraitedSpec):

    signif_signed_adj_fdr_mat_file = File(
        exists=True,
        desc="int matrix with corresponding codes to significance")


class StatsPairBinomial(BaseInterface):
    """
    StatsPairBinomialInputSpec
    """
    input_spec = StatsPairBinomialInputSpec
    output_spec = StatsPairBinomialOutputSpec

    def _run_interface(self, runtime):

        group_coclass_matrix_file1 = self.inputs.group_coclass_matrix_file1
        group_coclass_matrix_file2 = self.inputs.group_coclass_matrix_file2
        conf_interval_binom_fdr = self.inputs.conf_interval_binom_fdr

        # loading group_coclass_matrix1
        group_coclass_matrix1 = np.array(
            np.load(group_coclass_matrix_file1), dtype=float)
        print(group_coclass_matrix1.shape)

        # loading group_coclass_matrix2
        group_coclass_matrix2 = np.array(
            np.load(group_coclass_matrix_file2), dtype=float)
        print(group_coclass_matrix2.shape)

        # check input matrices
        # TODO checks are not done in compute_pairwise_binom_fdr already?
        Ix, Jx, nx = group_coclass_matrix1.shape
        Iy, Jy, ny = group_coclass_matrix2.shape

        assert Ix == Iy
        assert Jx == Jy
        assert Ix == Jx
        assert Iy == Jy

        signif_signed_adj_mat = stats.compute_pairwise_binom_fdr(
            group_coclass_matrix1, group_coclass_matrix2,
            conf_interval_binom_fdr)

        # save pairwise signed stat file
        signif_signed_adj_fdr_mat_file = os.path.abspath(
            'signif_signed_adj_fdr_' + str(conf_interval_binom_fdr) + '.npy')
        np.save(signif_signed_adj_fdr_mat_file, signif_signed_adj_mat)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["signif_signed_adj_fdr_mat_file"] = \
            os.path.abspath('signif_signed_adj_fdr_{}.npy'.format(
                self.inputs.conf_interval_binom_fdr))
        return outputs


# StatsPairTTest


class StatsPairTTestInputSpec(BaseInterfaceInputSpec):

    group_cormat_file1 = File(
        exists=True,
        desc='file of group 1 cormat matrices in npy format', mandatory=True)
    group_cormat_file2 = File(
        exists=True,
        desc='file of group 2 cormat matrices in npy format', mandatory=True)

    t_test_thresh_fdr = traits.Float(
        0.05, usedefault=True,
        desc='Alpha value used as FDR implementation', mandatory=False)

    paired = traits.Bool(True, usedefault=True,
                         desc='Ttest is paired or not', mandatory=False)


class StatsPairTTestOutputSpec(TraitedSpec):

    signif_signed_adj_fdr_mat_file = File(
        exists=True,
        desc="int matrix with corresponding codes to significance")


class StatsPairTTest(BaseInterface):

    """
    Compute ttest stats between 2 group of matrix
    - matrix are arranged in group_cormat, with order (Nx,Ny,Nsubj).
    Nx = Ny (each matricx is square)
    - t_test_thresh_fdr is optional (default, 0.05)
    - paired in indicate if ttest is pairde or not.
    If paired, both group have the same number of samples
    """
    input_spec = StatsPairTTestInputSpec
    output_spec = StatsPairTTestOutputSpec

    def _run_interface(self, runtime):

        print('in plot_cormat')

        group_cormat_file1 = self.inputs.group_cormat_file1
        group_cormat_file2 = self.inputs.group_cormat_file2
        t_test_thresh_fdr = self.inputs.t_test_thresh_fdr

        paired = self.inputs.paired
        print("loading group_cormat1")

        group_cormat1 = np.array(np.load(group_cormat_file1), dtype=float)
        print(group_cormat1.shape)

        print("loading group_cormat2")

        group_cormat2 = np.array(np.load(group_cormat_file2), dtype=float)
        print(group_cormat2.shape)

        print("compute NBS stats")

        # check input matrices
        Ix, Jx, nx = group_cormat1.shape
        Iy, Jy, ny = group_cormat2.shape

        assert Ix == Iy
        assert Jx == Jy
        assert Ix == Jx
        assert Iy == Jy

        # TODO checks are not done in compute_pairwise_ttest_fdr already?
        signif_signed_adj_mat = stats.compute_pairwise_ttest_fdr(
            group_cormat1, group_cormat2, t_test_thresh_fdr, paired)

        print('save pairwise signed stat file')

        signif_signed_adj_fdr_mat_file = os.path.abspath(
            'signif_signed_adj_fdr_' + str(t_test_thresh_fdr) + '.npy')
        np.save(signif_signed_adj_fdr_mat_file, signif_signed_adj_mat)

        # return signif_signed_adj_fdr_mat_file

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["signif_signed_adj_fdr_mat_file"] = \
            os.path.abspath('signif_signed_adj_fdr_{}.npy'.format(
                self.inputs.t_test_thresh_fdr))

        return outputs


# PrepareCormat


class PrepareCormatInputSpec(BaseInterfaceInputSpec):

    cor_mat_files = traits.List(
        File(exists=True),
        desc='list of all correlation matrice files (in npy format) for each \
            subject',
        mandatory=True)

    coords_files = traits.List(
        File(exists=True),
        desc='list of all coordinates in numpy space files (in txt format) \
            for each subject',
        mandatory=True,
        xor=['labels_files'])

    labels_files = traits.List(File(
        exists=True),
        desc='list of labels (in txt format) for each subject',
        mandatory=True, xor=['coords_files'])

    gm_mask_coords_file = File(
        exists=True,
        desc='Coordinates in numpy space, corresponding to all possible nodes\
            in the original space',
        mandatory=False,
        xor=['gm_mask_labels_file'])

    gm_mask_labels_file = File(
        exists=True,
        desc='Labels for all possible nodes - in case coords are varying from\
            one indiv to the other',
        mandatory=False, xor=['gm_mask_coords_file'])


class PrepareCormatOutputSpec(TraitedSpec):

    group_cormat_file = File(
        exists=True,
        desc="all cormat matrices of the group in .npy (pickle format)")

    avg_cormat_file = File(
        exists=True,
        desc="average of cormat matrix of the group in .npy (pickle format)")

    group_vect_file = File(
        exists=True,
        desc="degree by nodes * indiv of the group in .npy (pickleformat)")


class PrepareCormat(BaseInterface):

    """
    Description:

        Average correlation matrices, within a common reference
        (based on labels, or coordinates)

    Inputs:

        cor_mat_files:
            type = List of File, (exists=True),
            desc='list of all correlation matrice files (in npy format) for
            each subject', mandatory=True

        coords_files:
            type = List of File, (exists=True),
            desc='list of all coordinates in numpy space files
            for each subject', mandatory=True, xor = ['labels_files']

        labels_files:
             type = List of File,  (exists=True),
            desc='list of labels (in txt format) for each subject',
            mandatory=True, xor = ['coords_files'])

        gm_mask_coords_file:
            type = File(exists=True,
            desc='Coordinates in numpy space, corresponding to all possible
            nodes in the original space',
            mandatory=False, xor = ['gm_mask_labels_file'])

        gm_mask_labels_file :
            type = File, (exists=True),
            desc='Labels for all possible nodes - in case coords are varying
            from one indiv to the other (source space for example)',
            mandatory=False, xor = ['gm_mask_coords_file'])

    """
    input_spec = PrepareCormatInputSpec
    output_spec = PrepareCormatOutputSpec

    def _prep_coords_cormat(self):
        cor_mat_files = self.inputs.cor_mat_files
        nb_cormats = len(cor_mat_files)

        coords_files = self.inputs.coords_files
        gm_mask_coords_file = self.inputs.gm_mask_coords_file

        # loading gm mask corres
        gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file), dtype='int')
        nb_nodes = gm_mask_coords.shape[0]

        # defining return matrices
        sum_cormat = np.zeros((nb_nodes, nb_nodes), dtype=float)
        group_cormat = np.zeros((nb_nodes, nb_nodes, nb_cormats), dtype=float)
        group_vect = np.zeros((nb_nodes, nb_cormats), dtype=float)

        assert nb_cormats == len(coords_files), \
            ("Error, nb_cormats and coords_files are imcompatible {} \
             {}".format(nb_cormats, len(coords_files)))

        for index_file in range(nb_cormats):

            assert os.path.exists(cor_mat_files[index_file])
            assert os.path.exists(coords_files[index_file])

            Z_cor_mat = np.load(cor_mat_files[index_file])
            print(Z_cor_mat.shape)

            coords = np.array(np.loadtxt(
                coords_files[index_file]), dtype='int')
            print(coords.shape)

            corres_cor_mat, possible_edge_mat = \
                return_corres_correl_mat(Z_cor_mat, coords, gm_mask_coords)

            corres_cor_mat = corres_cor_mat + np.transpose(corres_cor_mat)

            sum_cormat += corres_cor_mat

            group_cormat[:, :, index_file] = corres_cor_mat

            group_vect[:, index_file] = np.sum(corres_cor_mat, axis=0)

        return sum_cormat, group_cormat, group_vect

    def _prep_labels_cormat(self):

        cor_mat_files = self.inputs.cor_mat_files

        nb_cormats = len(cor_mat_files)

        labels_files = self.inputs.labels_files
        gm_mask_labels_file = self.inputs.gm_mask_labels_file

        # loading gm mask labels
        gm_mask_labels = np.array(
            [line.strip() for line in open(gm_mask_labels_file)], dtype='str')

        nb_nodes = gm_mask_labels.shape[0]

        # defining return matrices
        sum_cormat = np.zeros((nb_nodes, nb_nodes), dtype=float)

        group_cormat = np.zeros((nb_nodes, nb_nodes, nb_cormats), dtype=float)

        group_vect = np.zeros((nb_nodes, len(cor_mat_files)), dtype=float)

        assert nb_cormats == len(labels_files), \
            ("Error, length of cor_mat_files, labels_files are imcompatible {}\
             {}".format(nb_cormats, len(labels_files)))

        for index_file in range(len(cor_mat_files)):

            assert os.path.exists(cor_mat_files[index_file])
            assert os.path.exists(labels_files[index_file])

            Z_cor_mat = np.load(cor_mat_files[index_file])

            labels = np.array([line.strip() for line in open(
                labels_files[index_file])], dtype='str')

            corres_cor_mat, possible_edge_mat = \
                return_corres_correl_mat_labels(Z_cor_mat, labels,
                                                gm_mask_labels)

            corres_cor_mat = corres_cor_mat + np.transpose(corres_cor_mat)

            sum_cormat += corres_cor_mat

            group_cormat[:, :, index_file] = corres_cor_mat

            group_vect[:, index_file] = np.sum(corres_cor_mat, axis=0)

        return sum_cormat, group_cormat, group_vect

    def _run_interface(self, runtime):

        cor_mat_files = self.inputs.cor_mat_files

        if isdefined(self.inputs.gm_mask_coords_file) and \
                isdefined(self.inputs.coords_files):

            sum_cormat, group_cormat, group_vect = self._prep_coords_cormat()

        elif isdefined(self.inputs.gm_mask_labels_file) and \
                isdefined(self.inputs.labels_files):

            sum_cormat, group_cormat, group_vect = self._prep_labels_cormat()

        else:
            print("Error, neither coords nor labels are defined properly")

        group_cormat_file = os.path.abspath('group_cormat.npy')
        np.save(group_cormat_file, group_cormat)

        group_vect_file = os.path.abspath('group_vect.npy')
        np.save(group_vect_file, group_vect)

        # saving cor_mat matrix
        avg_cormat_file = os.path.abspath('avg_cormat.npy')

        if (len(cor_mat_files) != 0):
            avg_cormat = sum_cormat / len(cor_mat_files)
            np.save(avg_cormat_file, avg_cormat)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["group_cormat_file"] = os.path.abspath('group_cormat.npy')
        outputs["avg_cormat_file"] = os.path.abspath('avg_cormat.npy')
        outputs["group_vect_file"] = os.path.abspath('group_vect.npy')
        return outputs

# SwapLists


class SwapListsInputSpec(BaseInterfaceInputSpec):

    list_of_lists = traits.List(
        traits.List(traits.List(File(exists=True))),
        desc='list of all correlation matrice files for each subject',
        mandatory=True)

    unbalanced = traits.Bool(False, default=True, usedefault=True)

    seed = traits.Int(-1, desc='value for seed',
                      mandatory=True, usedefault=True)


class SwapListsOutputSpec(TraitedSpec):

    permut_lists_of_lists = traits.List(
        traits.List(traits.List(File(exists=True))),
        desc='swapped list of all correlation matrice files for each subject',
        mandatory=True)


class SwapLists(BaseInterface):

    """
    Description:

    Exchange lists of files in a random fashion (based on seed value)
    Typically, cor_mat, coords -> 2, or Z_list, node_corres and labels -> 3

    If seed = -1, no swap is done (keep original values)

    Inputs:

        list_of_lists:
            type = List of List of  List of Files,
            exists=True,
            desc='list of all correlation matrice files (in npy format) for
            each subject',
            mandatory=True

        seed:
            type = Int, default = -1, desc='value for seed', mandatory=True,
            usedefault = True

    Outputs:

        permut_lists_of_lists:
            type = List of  List of List of Files ,exists=True,
            desc='swapped list of all correlation matrice files (in npy format)
            for each subject', mandatory=True


    """
    input_spec = SwapListsInputSpec
    output_spec = SwapListsOutputSpec

    def _run_interface(self, runtime):

        list_of_lists = self.inputs.list_of_lists
        seed = self.inputs.seed
        unbalanced = self.inputs.unbalanced

        # checking lists homogeneity
        if unbalanced:
            nb_files_per_list = []

        else:

            nb_files_per_list = -1

        # number of elements tomix from (highest level)
        nb_set_to_shuffle = len(list_of_lists)

        # number of files per case to shuffle
        # (typically cor_mat, coords -> 2, or Z_list,
        # node_corres and labels -> 3)

        nb_args = -1

        for i in range(nb_set_to_shuffle):

            if nb_args == -1:
                nb_args = len(list_of_lists[i])
            else:
                assert nb_args == len(list_of_lists[i]), \
                    ("Error list length {} != than list {} length {}".format(
                        nb_args, i, len(list_of_lists[i])))

            if unbalanced:
                nb_files_per_list.append(len(list_of_lists[i][0]))
            else:

                if nb_files_per_list == -1:
                    nb_files_per_list = len(list_of_lists[i][0])
                else:
                    assert nb_files_per_list == len(list_of_lists[i][0]), \
                        ("Error list length {} != than list {} length \
                            {}".format(nb_files_per_list, i,
                                       len(list_of_lists[i][0])))

        print(nb_files_per_list)

        # generating 0 or 1 for each subj:

        if seed == -1:
            self.permut_lists_of_lists = self.inputs.list_of_lists

            return runtime

        np.random.seed(seed)

        if unbalanced:

            def prod(x, y): return x * y

            print(sum(nb_files_per_list))

            is_permut = np.array(np.random.randint(
                nb_set_to_shuffle, size=sum(nb_files_per_list)), dtype=int)

            print(is_permut)

        else:

            def prod(x, y): return x * y

            # print reduce(prod,nb_files_per_list,1)

            is_permut = np.array(np.random.randint(
                nb_set_to_shuffle, size=nb_files_per_list), dtype=int)

            print(is_permut)

        np.savetxt(os.path.abspath("is_permut.txt"), is_permut, fmt="%d")

        # shuffling list

        self.permut_lists_of_lists = [
            [[] for i in range(nb_args)] for j in range(nb_set_to_shuffle)]

        if unbalanced:

            merged_group_lists = []

            for i in range(nb_args):

                merged_group_list = []

                for group in range(nb_set_to_shuffle):
                    merged_group_list = merged_group_list + \
                        list_of_lists[group][i]

                merged_group_lists.append(merged_group_list)

            for index_file, permut in enumerate(is_permut):

                for i in range(nb_args):
                    self.permut_lists_of_lists[permut][i].append(
                        merged_group_lists[i][index_file])

        else:

            for index_file, permut in enumerate(is_permut):

                for j in range(nb_set_to_shuffle):

                    shift = j + permut
                    rel_shift = shift % nb_set_to_shuffle

                    for i in range(nb_args):
                        self.permut_lists_of_lists[j][i].append(
                            list_of_lists[rel_shift][i][index_file])

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["permut_lists_of_lists"] = self.permut_lists_of_lists
        return outputs


# ShuffleMatrix


class ShuffleMatrixInputSpec(BaseInterfaceInputSpec):

    original_matrix_file = File(
        exists=True, desc='original matrix in npy format', mandatory=True)

    seed = traits.Int(-1, desc='value for seed',
                      mandatory=True, usedefault=True)


class ShuffleMatrixOutputSpec(TraitedSpec):

    shuffled_matrix_file = File(
        exists=True, desc='shuffled matrix in npy format', mandatory=True)


class ShuffleMatrix(BaseInterface):

    """
    Description:

    Compute randomisation of matrix, keeping the same distribution

    If seed = -1, no shuffling is done (keep original values)

    Inputs:

        original_matrix_file:
            type = File, exists=True, desc='original matrix in npy format',
            mandatory=True

        seed:
            type = Int, default = -1, desc='value for seed', mandatory=True,
            usedefault = True

    Outputs:

        shuffled_matrix_file:

            type = File, exists=True, desc='shuffled matrix in npy format',
            mandatory=True


    """
    input_spec = ShuffleMatrixInputSpec
    output_spec = ShuffleMatrixOutputSpec

    def _run_interface(self, runtime):

        original_matrix_file = self.inputs.original_matrix_file
        seed = self.inputs.seed

        path, fname, ext = split_f(original_matrix_file)

        orig_mat = np.load(original_matrix_file)

        if seed == -1:
            print("keeping original matrix")
            shuffled_matrix = orig_mat

        else:

            print("randomizing " + str(seed))
            np.random.seed(seed)

            np.fill_diagonal(orig_mat, np.nan)

            shuf_mat = np.zeros(shape=orig_mat.shape, dtype=orig_mat.dtype)

            for i, j in iter.combinations(list(range(orig_mat.shape[0])), 2):
                bool_ok = False

                while not bool_ok:

                    new_ind = np.random.randint(
                        low=orig_mat.shape[0], size=2)

                    if not np.isnan(orig_mat[new_ind[0], new_ind[1]]):

                        shuf_mat[i, j] = orig_mat[new_ind[0], new_ind[1]]
                        shuf_mat[j, i] = orig_mat[new_ind[0], new_ind[1]]

                        orig_mat[new_ind[0], new_ind[1]] = np.nan
                        orig_mat[new_ind[1], new_ind[0]] = np.nan

                        bool_ok = True

        shuffled_matrix_file = os.path.abspath("shuffled_matrix.npy")
        np.save(shuffled_matrix_file, shuffled_matrix)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["shuffled_matrix_file"] = os.path.abspath(
            "shuffled_matrix.npy")

        return outputs


# ShuffleNetList


class ShuffleNetListInputSpec(BaseInterfaceInputSpec):

    orig_net_list_file = File(
        exists=True, desc='original net list in txt format', mandatory=True)

    seed = traits.Int(-1, desc='value for seed',
                      mandatory=True, usedefault=True)


class ShuffleNetListOutputSpec(TraitedSpec):

    shuffled_net_list_file = File(
        exists=True, desc='shuffled net list in txt format', mandatory=True)


class ShuffleNetList(BaseInterface):

    """
    Extract mean time series from a labelled mask in Nifti Format
    where the voxels of interest have values 1
    """
    input_spec = ShuffleNetListInputSpec
    output_spec = ShuffleNetListOutputSpec

    def _run_interface(self, runtime):

        orig_net_list_file = self.inputs.orig_net_list_file
        seed = self.inputs.seed
        path, fname, ext = split_f(orig_net_list_file)
        original_net_list = np.loadtxt(orig_net_list_file)

        if seed == -1:
            print("keeping original matrix")

        else:
            print("randomizing " + str(seed))
            np.random.seed(seed)
            np.random.shuffle(original_net_list[:, 0])
            np.random.shuffle(original_net_list[:, 1])

        shuffled_net_list_file = os.path.abspath("shuffled_net_list.txt")
        np.savetxt(shuffled_net_list_file, original_net_list, fmt="%d %d %d")
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["shuffled_net_list_file"] = os.path.abspath(
            "shuffled_net_list.txt")

        return outputs
