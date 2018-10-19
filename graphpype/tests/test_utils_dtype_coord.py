import numpy as np

from graphpype.utils_dtype_coord import (find_index_in_coords, where_in_coords)

nb_subj_nodes = 5

subj_coords = np.random.randint(low=-70, high=70, size=(nb_subj_nodes, 3))

corres_coords = np.concatenate(
    (np.random.randint(low=-70, high=70, size=(5, 3)), subj_coords), axis=0)

nb_corres_nodes = corres_coords.shape[0]
np.random.shuffle(corres_coords)


def test_where_in_coords():
    """test_where_in_coords"""
    where_in_corres = where_in_coords(subj_coords, corres_coords)

    assert len(where_in_corres) == nb_subj_nodes
    assert np.max(where_in_corres) < corres_coords.shape[0]


def test_find_index_in_coords():
    """test find_index_in_coords"""
    val = find_index_in_coords(corres_coords, subj_coords)
    assert len(val) == nb_corres_nodes
