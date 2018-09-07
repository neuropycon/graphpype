# -*- coding: utf-8 -*-
"""
.. module:: grapĥ_stats
   :platform: Unix
   :synopsis: Preparing, computing stats and permutations

.. moduleauthor:: David Meunier <david.meunier@inserm.fr>


"""
import numpy as np
import os

import itertools as iter

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined


from nipype.utils.filemanip import split_filename as split_f

############################################################################################### StatsPairBinomial #####################################################################################################

import graphpype.utils_stats as stats


class StatsPairBinomialInputSpec(BaseInterfaceInputSpec):

    group_coclass_matrix_file1 = File(
        exists=True,  desc='file of group 1 coclass matrices in npy format', mandatory=True)
    group_coclass_matrix_file2 = File(
        exists=True,  desc='file of group 2 coclass matrices in npy format', mandatory=True)

    conf_interval_binom_fdr = traits.Float(
        0.05, usedefault=True, desc='Alpha value used as FDR implementation', mandatory=False)


class StatsPairBinomialOutputSpec(TraitedSpec):

    signif_signed_adj_fdr_mat_file = File(
        exists=True, desc="int matrix with corresponding codes to significance")


class StatsPairBinomial(BaseInterface):

    """
    StatsPairBinomialInputSpec

    """
    input_spec = StatsPairBinomialInputSpec
    output_spec = StatsPairBinomialOutputSpec

    def _run_interface(self, runtime):

        print('in plot_coclass')

        group_coclass_matrix_file1 = self.inputs.group_coclass_matrix_file1
        group_coclass_matrix_file2 = self.inputs.group_coclass_matrix_file2
        conf_interval_binom_fdr = self.inputs.conf_interval_binom_fdr

        print("loading group_coclass_matrix1")

        group_coclass_matrix1 = np.array(
            np.load(group_coclass_matrix_file1), dtype=float)
        print(group_coclass_matrix1.shape)

        print("loading group_coclass_matrix2")

        group_coclass_matrix2 = np.array(
            np.load(group_coclass_matrix_file2), dtype=float)
        print(group_coclass_matrix2.shape)

        print("compute NBS stats")

        # check input matrices
        Ix, Jx, nx = group_coclass_matrix1.shape
        Iy, Jy, ny = group_coclass_matrix2.shape

        assert Ix == Iy
        assert Jx == Jy
        assert Ix == Jx
        assert Iy == Jy

        signif_signed_adj_mat = stats.compute_pairwise_binom_fdr(
            group_coclass_matrix1, group_coclass_matrix2, conf_interval_binom_fdr)

        print('save pairwise signed stat file')

        signif_signed_adj_fdr_mat_file = os.path.abspath(
            'signif_signed_adj_fdr_' + str(conf_interval_binom_fdr) + '.npy')
        np.save(signif_signed_adj_fdr_mat_file, signif_signed_adj_mat)

        # return signif_signed_adj_fdr_mat_file

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["signif_signed_adj_fdr_mat_file"] = os.path.abspath(
            'signif_signed_adj_fdr_' + str(self.inputs.conf_interval_binom_fdr) + '.npy')

        return outputs

############################################################################################### StatsPairTTest #####################################################################################################


import graphpype.utils_stats as stats


class StatsPairTTestInputSpec(BaseInterfaceInputSpec):

    group_cormat_file1 = File(
        exists=True,  desc='file of group 1 cormat matrices in npy format', mandatory=True)
    group_cormat_file2 = File(
        exists=True,  desc='file of group 2 cormat matrices in npy format', mandatory=True)

    t_test_thresh_fdr = traits.Float(
        0.05, usedefault=True, desc='Alpha value used as FDR implementation', mandatory=False)

    paired = traits.Bool(True, usedefault=True,
                         desc='Ttest is paired or not', mandatory=False)


class StatsPairTTestOutputSpec(TraitedSpec):

    signif_signed_adj_fdr_mat_file = File(
        exists=True, desc="int matrix with corresponding codes to significance")


class StatsPairTTest(BaseInterface):

    """
    Compute ttest stats between 2 group of matrix 
    - matrix are arranged in group_cormat, with order (Nx,Ny,Nsubj). Nx = Ny (each matricx is square)
    - t_test_thresh_fdr is optional (default, 0.05)
    - paired in indicate if ttest is pairde or not. If paired, both group have the same number of samples
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

        outputs["signif_signed_adj_fdr_mat_file"] = os.path.abspath(
            'signif_signed_adj_fdr_' + str(self.inputs.t_test_thresh_fdr) + '.npy')

        return outputs


############################################################################################### PrepareCormat #####################################################################################################

from graphpype.utils_cor import return_corres_correl_mat, return_corres_correl_mat_labels
# ,return_hierachical_order


class PrepareCormatInputSpec(BaseInterfaceInputSpec):

    cor_mat_files = traits.List(File(
        exists=True), desc='list of all correlation matrice files (in npy format) for each subject', mandatory=True)

    #coords_files = traits.List(File(exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True)

    #gm_mask_coords_file = File(exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=True)

    coords_files = traits.List(File(
        exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True, xor=['labels_files'])

    labels_files = traits.List(File(
        exists=True), desc='list of labels (in txt format) for each subject (after removal of non void data)', mandatory=True, xor=['coords_files'])

    gm_mask_coords_file = File(
        exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=False, xor=['gm_mask_labels_file'])

    gm_mask_labels_file = File(
        exists=True, desc='Labels for all possible nodes - in case coords are varying from one indiv to the other (source space for example)', mandatory=False, xor=['gm_mask_coords_file'])


class PrepareCormatOutputSpec(TraitedSpec):

    group_cormat_file = File(
        exists=True, desc="all cormat matrices of the group in .npy (pickle format)")

    avg_cormat_file = File(
        exists=True, desc="average of cormat matrix of the group in .npy (pickle format)")

    group_vect_file = File(
        exists=True, desc="degree (?) by nodes * indiv of the group in .npy (pickle format)")


class PrepareCormat(BaseInterface):

    """
    Description:

        Average correlation matrices, within a common reference (based on labels, or coordiantes

    Inputs:

        cor_mat_files:
            type = List of File, (exists=True), desc='list of all correlation matrice files (in npy format) for each subject', mandatory=True

        coords_files:
            type = List of File, (exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True, xor = ['labels_files']

        labels_files = traits.List(File(exists=True), desc='list of labels (in txt format) for each subject (after removal of non void data)', mandatory=True, xor = ['coords_files'])

        gm_mask_coords_file = File(exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=False, xor = ['gm_mask_labels_file'])

        gm_mask_labels_file = File(exists=True, desc='Labels for all possible nodes - in case coords are varying from one indiv to the other (source space for example)', mandatory=False, xor = ['gm_mask_coords_file'])



    """
    input_spec = PrepareCormatInputSpec
    output_spec = PrepareCormatOutputSpec

    def _run_interface(self, runtime):

        print('in prepare_cormat')

        cor_mat_files = self.inputs.cor_mat_files

        if isdefined(self.inputs.gm_mask_coords_file) and isdefined(self.inputs.coords_files):

            coords_files = self.inputs.coords_files

            gm_mask_coords_file = self.inputs.gm_mask_coords_file

            print('loading gm mask corres')

            gm_mask_coords = np.array(np.loadtxt(
                gm_mask_coords_file), dtype='int')

            print(gm_mask_coords.shape)

            # read matrix from the first group
            # print Z_cor_mat_files

            sum_cormat = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0]), dtype=float)
            print(sum_cormat.shape)

            group_cormat = np.zeros(
                (gm_mask_coords.shape[0], gm_mask_coords.shape[0], len(cor_mat_files)), dtype=float)
            print(group_cormat.shape)

            group_vect = np.zeros(
                (gm_mask_coords.shape[0], len(cor_mat_files)), dtype=float)
            print(group_vect.shape)

            if len(cor_mat_files) != len(coords_files):
                print("warning, length of cor_mat_files, coords_files are imcompatible {} {} {}".format(
                    len(cor_mat_files), len(coords_files)))

            for index_file in range(len(cor_mat_files)):

                print(cor_mat_files[index_file])

                if os.path.exists(cor_mat_files[index_file]) and os.path.exists(coords_files[index_file]):

                    Z_cor_mat = np.load(cor_mat_files[index_file])
                    print(Z_cor_mat.shape)

                    coords = np.array(np.loadtxt(
                        coords_files[index_file]), dtype='int')
                    print(coords.shape)

                    corres_cor_mat, possible_edge_mat = return_corres_correl_mat(
                        Z_cor_mat, coords, gm_mask_coords)

                    corres_cor_mat = corres_cor_mat + \
                        np.transpose(corres_cor_mat)

                    print(corres_cor_mat.shape)
                    print(group_cormat.shape)

                    sum_cormat += corres_cor_mat

                    group_cormat[:, :, index_file] = corres_cor_mat

                    group_vect[:, index_file] = np.sum(corres_cor_mat, axis=0)

                else:
                    print("Warning, one or more files between " +
                          cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists")

        elif isdefined(self.inputs.gm_mask_labels_file) and isdefined(self.inputs.labels_files):

            labels_files = self.inputs.labels_files

            gm_mask_labels_file = self.inputs.gm_mask_labels_file

            print('loading gm mask labels')

            #gm_mask_labels = [line.strip() for line in open(gm_mask_labels_file)]

            # print len(gm_mask_labels)

            gm_mask_labels = np.array(
                [line.strip() for line in open(gm_mask_labels_file)], dtype='str')

            print(gm_mask_labels.shape)

            # read matrix from the first group
            # print Z_cor_mat_files

            sum_cormat = np.zeros(
                (gm_mask_labels.shape[0], gm_mask_labels.shape[0]), dtype=float)
            print(sum_cormat.shape)

            group_cormat = np.zeros(
                (gm_mask_labels.shape[0], gm_mask_labels.shape[0], len(cor_mat_files)), dtype=float)
            print(group_cormat.shape)

            group_vect = np.zeros(
                (gm_mask_labels.shape[0], len(cor_mat_files)), dtype=float)
            print(group_vect.shape)

            if len(cor_mat_files) != len(labels_files):
                print("warning, length of cor_mat_files, labels_files are imcompatible {} {} {}".format(
                    len(cor_mat_files), len(labels_files)))

            print(cor_mat_files)

            for index_file in range(len(cor_mat_files)):

                print(cor_mat_files[index_file])

                if os.path.exists(cor_mat_files[index_file]) and os.path.exists(labels_files[index_file]):

                    Z_cor_mat = np.load(cor_mat_files[index_file])
                    print(Z_cor_mat)
                    print(Z_cor_mat.shape)

                    labels = np.array([line.strip() for line in open(
                        labels_files[index_file])], dtype='str')
                    print("labels_subj:")
                    print(labels.shape)

                    corres_cor_mat, possible_edge_mat = return_corres_correl_mat_labels(
                        Z_cor_mat, labels, gm_mask_labels)

                    corres_cor_mat = corres_cor_mat + \
                        np.transpose(corres_cor_mat)

                    print(corres_cor_mat)
                    print(group_cormat.shape)

                    sum_cormat += corres_cor_mat

                    group_cormat[:, :, index_file] = corres_cor_mat

                    group_vect[:, index_file] = np.sum(corres_cor_mat, axis=0)

                else:
                    print("Warning, one or more files between " +
                          cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists")

            print(group_cormat)

        else:
            print("Error, neither coords nor labels are defined properly")

            print(self.inputs.gm_mask_coords_file)
            print(self.inputs.gm_mask_labels_file)
            print(self.inputs.coords_files)

            0/0

        group_cormat_file = os.path.abspath('group_cormat.npy')

        np.save(group_cormat_file, group_cormat)

        group_vect_file = os.path.abspath('group_vect.npy')

        np.save(group_vect_file, group_vect)

        print('saving cor_mat matrix')

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

############################ SwapLists ####################################################################################################


class SwapListsInputSpec(BaseInterfaceInputSpec):

    list_of_lists = traits.List(traits.List(traits.List(File(
        exists=True))), desc='list of all correlation matrice files (in npy format) for each subject', mandatory=True)

    unbalanced = traits.Bool(False, default=True, usedefault=True)

    seed = traits.Int(-1, desc='value for seed',
                      mandatory=True, usedefault=True)


class SwapListsOutputSpec(TraitedSpec):

    permut_lists_of_lists = traits.List(traits.List(traits.List(File(
        exists=True))), desc='swapped list of all correlation matrice files (in npy format) for each subject', mandatory=True)


class SwapLists(BaseInterface):

    """
    Description:

    Exchange lists of files in a random fashion (based on seed value)
    Typically, cor_mat, coords -> 2, or Z_list, node_corres and labels -> 3

    If seed = -1, no swap is done (keep original values)

    Inputs:

        list_of_lists:
            * type = List of List of  List of Files,
            * exists=True, 
            * desc='list of all correlation matrice files (in npy format) for each subject', 
            * mandatory=True

        seed:
            type = Int, default = -1, desc='value for seed', mandatory=True, usedefault = True

    Outputs:

        permut_lists_of_lists:
            type = List of  List of List of Files ,exists=True, desc='swapped list of all correlation matrice files (in npy format) for each subject', mandatory=True


    """
    input_spec = SwapListsInputSpec
    output_spec = SwapListsOutputSpec

    def _run_interface(self, runtime):

        print('in SwapLists')
        list_of_lists = self.inputs.list_of_lists
        seed = self.inputs.seed
        unbalanced = self.inputs.unbalanced

        # checking lists homogeneity
        print(len(list_of_lists))

        print(len(list_of_lists[0][0]))

        if unbalanced:
            nb_files_per_list = []

        else:

            nb_files_per_list = -1

        # number of elements tomix from (highest level)
        nb_set_to_shuffle = len(list_of_lists)

        # number of files per case to shuffle (typically cor_mat, coords -> 2, or Z_list, node_corres and labels -> 3)

        nb_args_per_list = -1

        for i in range(nb_set_to_shuffle):

            print(i)

            print(len(list_of_lists[i]))

            if nb_args_per_list == -1:
                nb_args_per_list = len(list_of_lists[i])
            else:
                assert nb_args_per_list == len(list_of_lists[i]), "Error list length {} != than list {} length {}".format(
                    nb_args_per_list, i, len(list_of_lists[i]))

            if unbalanced:
                nb_files_per_list.append(len(list_of_lists[i][0]))
            else:

                if nb_files_per_list == -1:
                    nb_files_per_list = len(list_of_lists[i][0])
                else:
                    assert nb_files_per_list == len(list_of_lists[i][0]), "Error list length {} != than list {} length {}".format(
                        nb_files_per_list, i, len(list_of_lists[i][0]))

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
            [[] for i in range(nb_args_per_list)] for j in range(nb_set_to_shuffle)]

        print(len(self.permut_lists_of_lists))
        print(len(self.permut_lists_of_lists[0]))
        print(len(self.permut_lists_of_lists[0][0]))

        if unbalanced:

            merged_group_lists = []

            for i in range(nb_args_per_list):

                merged_group_list = []

                for group in range(nb_set_to_shuffle):
                    # print group

                    merged_group_list = merged_group_list + \
                        list_of_lists[group][i]

                    # print len(merged_group_list)

                # print len(merged_group_list)

                merged_group_lists.append(merged_group_list)

            print(len(merged_group_lists))
            print(len(merged_group_lists[0]))

            for index_file, permut in enumerate(is_permut):

                print("index_file:", end=' ')
                print(index_file)

                print("permut:", end=' ')
                print(permut)

                for i in range(nb_args_per_list):

                    self.permut_lists_of_lists[permut][i].append(
                        merged_group_lists[i][index_file])

            print(len(self.permut_lists_of_lists))
            print(len(self.permut_lists_of_lists[0]))
            print(len(self.permut_lists_of_lists[0][0]))
            print(len(self.permut_lists_of_lists[1][0]))

            print(self.permut_lists_of_lists[0][0])

        else:

            for index_file, permut in enumerate(is_permut):

                print("index_file:", end=' ')
                print(index_file)

                print("permut:", end=' ')
                print(permut)

                print("nb_set_to_shuffle:", end=' ')
                print(nb_set_to_shuffle)

                for j in range(nb_set_to_shuffle):

                    print(j, permut)

                    shift = j + permut

                    rel_shift = shift % nb_set_to_shuffle

                    print(shift, rel_shift)

                    for i in range(nb_args_per_list):
                        self.permut_lists_of_lists[j][i].append(
                            list_of_lists[rel_shift][i][index_file])

            # print self.permut_lists_of_lists

            print(len(self.permut_lists_of_lists))
            print(len(self.permut_lists_of_lists[0]))

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["permut_lists_of_lists"] = self.permut_lists_of_lists

        return outputs


############################ ShuffleMatrix ####################################################################################################
import itertools as iter


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
            type = File, exists=True, desc='original matrix in npy format', mandatory=True

        seed:
            type = Int, default = -1, desc='value for seed', mandatory=True, usedefault = True

    Outputs:

        shuffled_matrix_file:

            type = File, exists=True, desc='shuffled matrix in npy format', mandatory=True


    """
    input_spec = ShuffleMatrixInputSpec
    output_spec = ShuffleMatrixOutputSpec

    def _run_interface(self, runtime):

        print('in prepare_coclass')
        original_matrix_file = self.inputs.original_matrix_file
        seed = self.inputs.seed

        path, fname, ext = split_f(original_matrix_file)

        original_matrix = np.load(original_matrix_file)

        print(original_matrix)
        print(original_matrix.shape)

        if seed == -1:
            print("keeping original matrix")
            shuffled_matrix = original_matrix

        else:

            print("randomizing " + str(seed))
            np.random.seed(seed)

            np.fill_diagonal(original_matrix, np.nan)

            shuffled_matrix = np.zeros(
                shape=original_matrix.shape, dtype=original_matrix.dtype)

            for i, j in iter.combinations(list(range(original_matrix.shape[0])), 2):
                # print i,j

                bool_ok = False

                while not bool_ok:

                    new_indexes = np.random.randint(
                        low=original_matrix.shape[0], size=2)

                    # print new_indexes

                    if not np.isnan(original_matrix[new_indexes[0], new_indexes[1]]):

                        shuffled_matrix[i, j] = shuffled_matrix[j,
                                                                i] = original_matrix[new_indexes[0], new_indexes[1]]

                        original_matrix[new_indexes[0], new_indexes[1]
                                        ] = original_matrix[new_indexes[1], new_indexes[0]] = np.nan

                        bool_ok = True

                    # print bool_ok,np.sum(np.isnan(original_matrix))

                # print "**********************************"

        print(original_matrix)
        print(shuffled_matrix)

        shuffled_matrix_file = os.path.abspath("shuffled_matrix.npy")

        np.save(shuffled_matrix_file, shuffled_matrix)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["shuffled_matrix_file"] = os.path.abspath(
            "shuffled_matrix.npy")

        return outputs


############################ ShuffleNetList ####################################################################################################
import itertools as iter


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
    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1
    """
    input_spec = ShuffleNetListInputSpec
    output_spec = ShuffleNetListOutputSpec

    def _run_interface(self, runtime):

        print('in prepare_coclass')
        orig_net_list_file = self.inputs.orig_net_list_file
        seed = self.inputs.seed

        path, fname, ext = split_f(orig_net_list_file)

        original_net_list = np.loadtxt(orig_net_list_file)

        print(original_net_list)

        if seed == -1:
            print("keeping original matrix")

        else:

            print("randomizing " + str(seed))
            np.random.seed(seed)

            np.random.shuffle(original_net_list[:, 0])
            np.random.shuffle(original_net_list[:, 1])

            # print original_net_list[:,:2].shape
            # np.random.shuffle(original_net_list[:,2])

            # print original_net_list[:,2]
            # print original_net_list[:,2].shape

        print(original_net_list)

        shuffled_net_list_file = os.path.abspath("shuffled_net_list.txt")

        np.savetxt(shuffled_net_list_file, original_net_list, fmt="%d %d %d")

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["shuffled_net_list_file"] = os.path.abspath(
            "shuffled_net_list.txt")

        return outputs
