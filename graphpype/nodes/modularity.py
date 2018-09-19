# -*- coding: utf-8 -*-

import os
import numpy as np

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import traits, File, TraitedSpec, isdefined

from graphpype.utils_net import (return_net_list, return_int_net_list,
                                 export_Louvain_net_from_list,
                                 export_List_net_from_list)

from graphpype.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
from graphpype.utils_mod import (compute_roles, read_lol_file,
                                 _inter_module_avgmat)

from graphpype.utils import is_symetrical
# ComputeNetList


class ComputeNetListInputSpec(BaseInterfaceInputSpec):

    Z_cor_mat_file = File(
        exists=True, desc='Normalized correlation matrix', mandatory=True)
    coords_file = File(
        exists=True, desc='Corresponding coordiantes', mandatory=False)
    threshold = traits.Float(xor=['density'], mandatory=False)
    density = traits.Float(xor=['threshold'], mandatory=False)
    export_Louvain = traits.Bool(
        False, desc="whether to export data as Louvain Traag as well",
        usedefault=True)


class ComputeNetListOutputSpec(TraitedSpec):

    net_List_file = File(exists=True, desc="net list for radatools")
    net_Louvain_file = File(desc="net list for Louvain")


class ComputeNetList(BaseInterface):
    """
    Description:

        Format correlation matrix to a list: format i j weight
        (integer = float * 1000)

    Inputs:

        Z_cor_mat_file:
            type = File, exists=True, desc='Normalized correlation matrix',
            mandatory=True

        coords_file:
            type = File, exists=True, desc='Corresponding coordiantes',
            mandatory=False

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

        Z_cor_mat = np.load(Z_cor_mat_file)

        if threshold != traits.Undefined and density == traits.Undefined:

            Z_cor_mat[np.abs(Z_cor_mat) < threshold] = 0.0
            Z_list = return_net_list(Z_cor_mat)

        elif threshold == traits.Undefined and density != traits.Undefined:

            Z_list = return_net_list(Z_cor_mat)
            N = int(Z_list.shape[0]*density)
            all_sorted_indexes = (-np.abs(Z_list[:, 2])).argsort()
            sorted_indexes = all_sorted_indexes[:N]
            max_thr_for_den_file = os.path.abspath('max_thr_for_den.txt')

            if N == Z_list.shape[0]:
                N = N-1

            max_thr_for_den = np.array(Z_list[all_sorted_indexes[N], 2])

            with open(max_thr_for_den_file, "w") as f:
                f.write("max_thr_for_den:{}".format(max_thr_for_den))

            Z_list = Z_list[sorted_indexes, :]
        else:

            Z_list = return_net_list(Z_cor_mat)

        # Z correl_mat as list of edges
        net_List_file = os.path.abspath('Z_List.txt')
        export_List_net_from_list(net_List_file, Z_list)

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

# ComputeIntNetList


class ComputeIntNetListInputSpec(BaseInterfaceInputSpec):

    int_mat_file = File(exists=True, desc='Integer matrix', mandatory=True)

    threshold = traits.Int(
        exists=True, desc="Interger Value (optional) for thresholding",
        mandatory=False)

    coords_file = File(
        exists=True, desc='Corresponding coordiantes', mandatory=False)

    export_Louvain = traits.Bool(
        False, desc="whether to export data as Louvain-Traag as well",
        usedefault=True)


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
        threshold = self.inputs.threshold

        print("loading int_mat_file")

        int_mat = np.load(int_mat_file)
        if not isdefined(threshold):
            threshold = 0

        int_list = return_int_net_list(int_mat, threshold)

        net_List_file = os.path.abspath('int_List.txt')

        export_List_net_from_list(net_List_file, int_list)

        # int correl_mat as Louvain format
        if self.inputs.export_Louvain:

            coords = np.loadtxt(self.inputs.coords_file)
            net_Louvain_file = os.path.abspath('Z_Louvain.txt')
            export_Louvain_net_from_list(net_Louvain_file, int_list, coords)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["net_List_file"] = os.path.abspath("int_List.txt")
        if self.inputs.export_Louvain:
            outputs["net_Louvain_file"] = os.path.abspath("int_Louvain.txt")

        return outputs


# ComputeNodeRoles


class ComputeNodeRolesInputSpec(BaseInterfaceInputSpec):

    rada_lol_file = File(
        exists=True,
        desc='lol file, describing modular structure of the network',
        mandatory=True)

    Pajek_net_file = File(
        exists=True,
        desc='net description in Pajek format', mandatory=True)

    role_type = traits.Enum('Amaral_roles', '4roles',
                            desc='definition of node roles',
                            usedefault=True)


class ComputeNodeRolesOutputSpec(TraitedSpec):

    node_roles_file = File(exists=True, desc="node roles with an integer code")

    all_Z_com_degree_file = File(
        exists=True,
        desc="value of quantity, describing the hub/non-hub role of the nodes")

    all_participation_coeff_file = File(
        exists=True,
        desc="value of quality, descibing the provincial/connector role of the\
            nodes")


class ComputeNodeRoles(BaseInterface):

    """
    Description:

    Compute node roles from lol modular partition and original network

    Inputs:

        rada_lol_file:
            type = File, exists=True,
            desc='lol file, describing modular structure of the network',
            mandatory=True


        Pajek_net_file:
            type = File, exists=True, desc='net description in Pajek format',
            mandatory=True

        role_type:
            One of Enum('Amaral_roles', '4roles'),
            desc='definition of node roles, Amaral_roles = original 7 roles
            defined for transport network (useful for big network), 4_roles =
            defines only provincial/connecteur from participation coeff',
            usedefault=True

    Outputs:

        node_roles_file:
            type = File, exists=True, desc="node roles with an integer code"

        all_Z_com_degree_file:
            type = File,exists=True,
            desc="value of quantity, describing the hub/non-hub role of the
            nodes"

        all_participation_coeff_file
            type = File, exists=True,
            desc="value of quality, descibing the provincial/connector role of
            the nodes"

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


# ComputeModuleMatProp


class ComputeModuleMatPropInputSpec(BaseInterfaceInputSpec):

    rada_lol_file = File(
        exists=True,
        desc='lol file, describing modular structure of the network',
        mandatory=True)

    Pajek_net_file = File(
        exists=True,
        desc='net description in Pajek format', mandatory=True)

    conmat_file = File(
        exists=True,
        desc='full matrix in npy format', mandatory=True)

    export_excel = traits.Bool(
        False, desc="export data as xls (as well as csv)",
        usedefault=True)


class ComputeModuleMatPropOutputSpec(TraitedSpec):

    df_avgmat_file = File(
        exists=True,
        desc="module properties")

    df_avgmat_excel_file = File(
        desc="module properties in xls format")


class ComputeModuleMatProp(BaseInterface):
    """
    Description:

    Compute module and intermodule properties from Mat

    Inputs:

        rada_lol_file:
            type = File, exists=True,
            desc='lol file, describing modular structure of the network',
            mandatory=True

        conmat_file:
            #type = File, exists=True, desc='full matrix in npy format',
            #mandatory=True

        Pajek_net_file:
            type = File, exists=True, desc='net description in Pajek format',
            mandatory=True

        export_excel:
            type = File

        corres:

    Outputs:

    df_avgmat_file:
        type = File,
        exists=True,
        desc="module properties"

    optional if export_excel:

    df_avgmat_excel_file:
        type = File
        desc="module properties in xls format"

    """
    input_spec = ComputeModuleMatPropInputSpec
    output_spec = ComputeModuleMatPropOutputSpec

    def _run_interface(self, runtime):

        rada_lol_file = self.inputs.rada_lol_file
        Pajek_net_file = self.inputs.Pajek_net_file
        conmat_file = self.inputs.conmat_file
        export_excel = self.inputs.export_excel

        community_vect = read_lol_file(rada_lol_file)
        corres_nodes, sparse_mat = \
            read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)

        # density
        conmat = np.load(conmat_file)
        corres_mat = conmat[:, corres_nodes][corres_nodes, :]

        # intermodule
        if not is_symetrical(corres_mat):
            corres_mat = corres_mat + np.transpose(corres_mat)

        df_avgmat = _inter_module_avgmat(corres_mat, community_vect)
        df_avgmat_file = os.path.abspath("res_avgmat.csv")
        df_avgmat.to_csv(df_avgmat_file)

        if export_excel:
            try:
                import xlwt # noqa
                df_avgmat_excel_file = os.path.abspath("res_avgmat.xls")
                df_avgmat.to_excel(df_avgmat_excel_file)

            except ImportError:
                print("Error, xlwt not installed, cannot export Excel file")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["df_avgmat_file"] = os.path.abspath("res_avgmat.csv")

        if self.inputs.export_excel:
            outputs["df_avgmat_excel_file"] = os.path.abspath("res_avgmat.xls")

        return outputs
