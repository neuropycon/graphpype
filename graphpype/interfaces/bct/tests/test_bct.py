"""test rada"""
import os
import numpy as np

from graphpype.interfaces.bct import (KCore)

N = 100
np_mat = np.random.randint(0, 2, size=N*N, dtype='int').reshape(N, N)

print(np_mat)

np_mat_file = os.path.abspath("rand_bin_mat.npy")
np.save(np_mat_file, np_mat)


def test_KCore_directed():

    kcore = KCore()

    kcore.inputs.np_mat_file = os.path.abspath("rand_bin_mat.npy")
    kcore.inputs.is_directed = True

    kcore.run()

    assert os.path.exists(os.path.abspath("coreness.npy"))


def test_KCore_undirected():

    kcore = KCore()

    kcore.inputs.np_mat_file = os.path.abspath("rand_bin_mat.npy")
    kcore.inputs.is_directed = False

    kcore.run()

    assert os.path.exists(os.path.abspath("coreness.npy"))


test_KCore_directed()
test_KCore_undirected()
