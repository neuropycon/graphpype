"""test rada"""
import os
import numpy as np

from graphpype.utils import _make_tmp_dir
from graphpype.interfaces.bct import (KCore)

N = 100
np_mat = np.random.randint(0, 2, size=N*N, dtype='int').reshape(N, N)


def test_KCore_directed():
    _make_tmp_dir()

    np_mat_file = os.path.abspath("rand_bin_mat.npy")
    np.save(np_mat_file, np_mat)

    kcore = KCore()

    kcore.inputs.np_mat_file = os.path.abspath("rand_bin_mat.npy")
    kcore.inputs.is_directed = True

    kcore.run()

    assert os.path.exists(os.path.abspath("coreness.npy"))
    assert os.path.exists(os.path.abspath("distrib_k.npy"))


def test_KCore_undirected():
    _make_tmp_dir()

    np_mat_file = os.path.abspath("rand_bin_mat.npy")
    np.save(np_mat_file, np_mat)

    kcore = KCore()
    kcore.inputs.np_mat_file = os.path.abspath("rand_bin_mat.npy")
    kcore.inputs.is_directed = False
    kcore.run()

    assert os.path.exists(os.path.abspath("coreness.npy"))
    assert os.path.exists(os.path.abspath("distrib_k.npy"))
