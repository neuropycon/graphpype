# -*- coding: utf-8 -*-

#from graphpype.plot_igraph import *

#import rpy,os
import os
import nibabel as nib
import numpy as np


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec

# from nipype.interfaces.base import CommandLine, CommandLineInputSpec ### not used but should be done for PrepRada

from nipype.interfaces.base import traits, File, TraitedSpec, isdefined

from nipype.utils.filemanip import split_filename as split_f


#from enthought.traits.trait_base import Undefined
######################################################################################## ComputeNetList ##################################################################################################################

from graphpype.utils_net import return_net_list

from graphpype.utils_net import export_Louvain_net_from_list


class ComputeNetListInputSpec(BaseInterfaceInputSpec):

    Z_cor_mat_file = File(
        exists=True, desc='Normalized correlation matrix', mandatory=True)

    coords_file = File(
        exists=True, desc='Corresponding coordiantes', mandatory=False)

    threshold = traits.Float(xor=['density'], mandatory=False)

    density = traits.Float(xor=['threshold'], mandatory=False)

    export_Louvain = traits.Bool(
        False, desc="whether to export data as Louvain Traag as well", usedefault=True)


class ComputeNetListOutputSpec(TraitedSpec):

    net_List_file = File(exists=True, desc="net list for radatools")

    net_Louvain_file = File(desc="net list for Louvain")

    #out_coords_file = File(exists=True, desc='Corresponding coordiantes (copy from previous one)')


class ComputeNetList(BaseInterface):

    """
    Description:

        Format correlation matrix to a list: format i j weight (integer = float * 1000)

    Inputs:

        Z_cor_mat_file:
            type = File, exists=True, desc='Normalized correlation matrix', mandatory=True

        coords_file:
            type = File, exists=True, desc='Corresponding coordiantes', mandatory=False

        threshold:
            type = Float, xor = ['density'], mandatory = False

        density:
            type = Float, xor = ['threshold'], mandatory = False

    Outputs:

        net_List_file:
            type = File, exists=True, desc="net list for radatools"

    """

    input_spec = ComputeNetListInputSpec
    output_spec = ComputeNetListOutputSpec

    def _run_interface(self, runtime):

        Z_cor_mat_file = self.inputs.Z_cor_mat_file
        threshold = self.inputs.threshold
        density = self.inputs.density

        print("loading Z_cor_mat_file")

        Z_cor_mat = np.load(Z_cor_mat_file)

        print(threshold, density)

        if threshold != traits.Undefined and density == traits.Undefined:

            Z_cor_mat[np.abs(Z_cor_mat) < threshold] = 0.0

            Z_list = return_net_list(Z_cor_mat)

            print(Z_list)

        elif threshold == traits.Undefined and density != traits.Undefined:

            print(density)

            Z_list = return_net_list(Z_cor_mat)
            print(Z_list.shape)

            N = int(Z_list.shape[0]*density)

            all_sorted_indexes = (-np.abs(Z_list[:, 2])).argsort()

            print(all_sorted_indexes)
            sorted_indexes = all_sorted_indexes[:N]

            max_thr_for_den_file = os.path.abspath('max_thr_for_den.txt')

            if N == Z_list.shape[0]:
                N = N-1

            max_thr_for_den = np.array(Z_list[all_sorted_indexes[N], 2])

            with open(max_thr_for_den_file, "w") as f:

                f.write("max_thr_for_den:{}".format(max_thr_for_den))

            Z_list = Z_list[sorted_indexes, :]

            print(Z_list)

        else:

            Z_list = return_net_list(Z_cor_mat)

        # Z correl_mat as list of edges
        print("saving Z_list as list of edges")

        net_List_file = os.path.abspath('Z_List.txt')

        np.savetxt(net_List_file, Z_list, fmt='%d %d %d')

        if self.inputs.export_Louvain:

            coords = np.loadtxt(self.inputs.coords_file)

            net_Louvain_file = os.path.abspath('Z_Louvain.txt')

            export_Louvain_net_from_list(net_Louvain_file, Z_list, coords)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["net_List_file"] = os.path.abspath("Z_List.txt")

        if self.inputs.export_Louvain:

            outputs["net_Louvain_file"] = os.path.abspath("Z_Louvain.txt")

        return outputs

######################################################################################## ComputeIntNetList ##################################################################################################################


from graphpype.utils_net import return_int_net_list
from graphpype.utils_net import export_Louvain_net_from_list


class ComputeIntNetListInputSpec(BaseInterfaceInputSpec):

    int_mat_file = File(exists=True, desc='Integer matrix', mandatory=True)

    threshold = traits.Int(
        exists=True, desc="Interger Value (optional) for thresholding", mandatory=False)

    coords_file = File(
        exists=True, desc='Corresponding coordiantes', mandatory=False)

    export_Louvain = traits.Bool(
        False, desc="whether to export data as Louvain-Traag as well", usedefault=True)


class ComputeIntNetListOutputSpec(TraitedSpec):

    net_List_file = File(exists=True, desc="net list for radatools")

    net_Louvain_file = File(desc="net list for Louvain")


class ComputeIntNetList(BaseInterface):

    """
    Format integer matrix to a list format i j weight 
    Option for thresholding 
    """

    input_spec = ComputeIntNetListInputSpec
    output_spec = ComputeIntNetListOutputSpec

    def _run_interface(self, runtime):

        int_mat_file = self.inputs.int_mat_file
        #coords_file = self.inputs.coords_file
        threshold = self.inputs.threshold

        print("loading int_mat_file")

        int_mat = np.load(int_mat_file)

        # print 'load coords'

        #coords = np.array(np.loadtxt(coords_file),dtype = int)

        # compute int_list

        if not isdefined(threshold):

            threshold = 0

        int_list = return_int_net_list(int_mat, threshold)

        # int correl_mat as list of edges

        print("saving int_list as list of edges")

        net_List_file = os.path.abspath('int_List.txt')

        export_List_net_from_list(net_List_file, int_list)

        # int correl_mat as Louvain format
        if self.inputs.export_Louvain:

            print("saving net_list as Louvain format")

            coords = np.loadtxt(self.inputs.coords_file)

            net_Louvain_file = os.path.abspath('Z_Louvain.txt')

            export_Louvain_net_from_list(net_Louvain_file, Z_list, coords)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["net_List_file"] = os.path.abspath("int_List.txt")

        if self.inputs.export_Louvain:

            outputs["net_Louvain_file"] = os.path.abspath("int_Louvain.txt")

        return outputs

######################################################################################## ComputeNodeRoles ##################################################################################################################


from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
from graphpype.utils_mod import compute_roles, read_lol_file


class ComputeNodeRolesInputSpec(BaseInterfaceInputSpec):

    rada_lol_file = File(
        exists=True, desc='lol file, describing modular structure of the network', mandatory=True)
    Pajek_net_file = File(
        exists=True, desc='net description in Pajek format', mandatory=True)

    role_type = traits.Enum('Amaral_roles', '4roles', desc='definition of node roles, Amaral_roles = original 7 roles defined for transport network (useful for big network), 4_roles defines only provincial/connecteur from participation coeff',
                            usedefault=True)


class ComputeNodeRolesOutputSpec(TraitedSpec):

    node_roles_file = File(exists=True, desc="node roles with an integer code")
    all_Z_com_degree_file = File(
        exists=True, desc="value of quantity, describing the hub/non-hub role of the nodes")
    all_participation_coeff_file = File(
        exists=True, desc="value of quality, descibing the provincial/connector role of the nodes")


class ComputeNodeRoles(BaseInterface):

    """
    Description:

    Compute node roles from lol modular partition and original network

    Inputs: 

        rada_lol_file: 
            type = File, exists=True,desc='lol file, describing modular structure of the network', mandatory=True


        Pajek_net_file:
            type = File(exists=True, desc='net description in Pajek format', mandatory=True

        role_type:
            One of Enum('Amaral_roles', '4roles'), desc='definition of node roles, Amaral_roles = original 7 roles defined for transport network (useful for big network), 4_roles = defines only provincial/connecteur from participation coeff', usedefault=True

    Outputs:

        node_roles_file:
            type = File, exists=True, desc="node roles with an integer code"

        all_Z_com_degree_file:
            type = File,exists=True, desc="value of quantity, describing the hub/non-hub role of the nodes"

        all_participation_coeff_file
            type = File, exists=True, desc="value of quality, descibing the provincial/connector role of the nodes"

    """
    input_spec = ComputeNodeRolesInputSpec
    output_spec = ComputeNodeRolesOutputSpec

    def _run_interface(self, runtime):

        rada_lol_file = self.inputs.rada_lol_file
        Pajek_net_file = self.inputs.Pajek_net_file

        print('Loading Pajek_net_file for reading node_corres')

        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(
            Pajek_net_file)

        print(sparse_mat.todense())

        print(node_corres.shape, sparse_mat.todense().shape)

        print("Loading community belonging file " + rada_lol_file)

        community_vect = read_lol_file(rada_lol_file)

        print(community_vect)

        print("Computing node roles")

        node_roles, all_Z_com_degree, all_participation_coeff = compute_roles(
            community_vect, sparse_mat, role_type=self.inputs.role_type)

        print(node_roles)

        node_roles_file = os.path.abspath('node_roles.txt')

        np.savetxt(node_roles_file, node_roles, fmt='%d')

        all_Z_com_degree_file = os.path.abspath('all_Z_com_degree.txt')

        np.savetxt(all_Z_com_degree_file, all_Z_com_degree, fmt='%f')

        all_participation_coeff_file = os.path.abspath(
            'all_participation_coeff.txt')

        np.savetxt(all_participation_coeff_file,
                   all_participation_coeff, fmt='%f')

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["node_roles_file"] = os.path.abspath('node_roles.txt')
        outputs["all_Z_com_degree_file"] = os.path.abspath(
            'all_Z_com_degree.txt')
        outputs["all_participation_coeff_file"] = os.path.abspath(
            'all_participation_coeff.txt')

        return outputs
