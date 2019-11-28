
import os

import numpy as np

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec

from bct import (kcoreness_centrality_bu, kcoreness_centrality_bd)


class KCoreInputSpec(BaseInterfaceInputSpec):

    np_mat_file = File(
        exists=True,
        desc='numpy matrix N*N to apply the data to',
        mandatory=True)

    is_directed = traits.Bool(
        False, usedefault=True, desc="Is the matrix directed ?")


class KCoreOutputSpec(TraitedSpec):

    coreness_file = File(
        exists=True,
        desc="coreness vector file")

    distrib_k_file = File(
        exists=True,
        desc="distrib_k vector file")


class KCore(BaseInterface):
    """
    Description:
        Compute K core
        wraps of kcoreness_centrality_bu and kcoreness_centrality_bd in bctpy

    Inputs:
        np_mat_file:
            type = File,
            exists=True,
            desc='numpy matrix N*N to apply the data to',
            mandatory=True)

        is_directed = traits.Bool(
            False, usedefault = True, desc="Is the matrix directed ?")


    Outputs:
        coreness_file:
            type = File,
            exists=True,
            desc="coreness vector file"


        distrib_k_file =
            type = File
            exists=True,
            desc="distrib_k vector file"
    """
    input_spec = KCoreInputSpec
    output_spec = KCoreOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        is_directed = self.inputs.is_directed

        # loading data
        np_mat = np.load(np_mat_file)

        # running bctpy
        if is_directed:
            coreness, distrib_k = kcoreness_centrality_bd(np_mat)
        else:
            coreness, distrib_k = kcoreness_centrality_bu(np_mat)
        print(coreness)

        np.save(os.path.abspath("coreness.npy"), coreness)
        np.save(os.path.abspath("distrib_k.npy"), distrib_k)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["coreness_file"] = os.path.abspath("coreness.npy")
        outputs["distrib_k_file"] = os.path.abspath("distrib_k.npy")
        return outputs
