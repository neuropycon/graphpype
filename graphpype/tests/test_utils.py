import pytest
import itertools
import numpy as np

from graphpype.utils import (get_first, get_second, show_length, show_files,
                             get_multiple_indexes, random_product,
                             check_dimensions, check_np_shapes,
                             check_np_dimension, is_symetrical)


def test_gets():
    """test get_first and get_second"""
    two_elem_int_list = [1, 2]
    two_elem_int_tuple = (1, 2)

    with pytest.raises(ValueError):
        get_first(1)

    with pytest.raises(ValueError):
        get_first([])

    assert get_first(two_elem_int_list) == 1
    assert get_first(two_elem_int_tuple) == 1

    with pytest.raises(ValueError):
        get_second([1])

    assert get_second(two_elem_int_list) == 2
    assert get_second(two_elem_int_tuple) == 2


def test_shows():
    """test show_length and show_files """
    with pytest.raises(ValueError):
        show_length(1)

    with pytest.raises(ValueError):
        show_files(1)


def test_get_multiple_indexes():
    """test_get_multiple_indexes"""
    list_test = [1, 2, 1, 2]
    assert get_multiple_indexes(list_test, 1) == [0, 2]
    assert get_multiple_indexes(list_test, 2) == [1, 3]


def test_random_product():
    """test_random_product"""
    list_prod1 = [1, 2, 3]
    list_prod2 = ['a', 'b']
    val_random = random_product(list_prod1, list_prod2)
    val_iter = itertools.product(list_prod1, list_prod2)
    assert val_random in [val for val in val_iter]


def test_checks():
    """test check_dimensions and check_np_shapes and check_np_dimension"""
    mat_3D_zeros = np.zeros((2, 3, 4), dtype='int')
    mat_3D_ones = np.ones((2, 3, 4), dtype='int')
    wrong_mat_3D_ones = np.ones((2, 3, 10), dtype='int')
    index_OK = [1, 2, 3]
    wrong_index = [4, 5, 6]
    np_index_OK = np.array(index_OK, dtype="int")
    np_wrong_index = np.array(wrong_index, dtype="int")

    assert check_dimensions(index_OK, mat_3D_zeros.shape)
    assert check_dimensions(wrong_index, mat_3D_zeros.shape) is False

    with pytest.raises(AssertionError):
        check_np_shapes(mat_3D_zeros, mat_3D_ones)

    assert check_np_shapes(mat_3D_zeros.shape, mat_3D_ones.shape)
    assert check_np_shapes(mat_3D_zeros.shape, wrong_mat_3D_ones.shape) \
        is False

    assert check_np_dimension(mat_3D_zeros.shape, np_index_OK)
    assert check_np_dimension(mat_3D_zeros.shape, np_wrong_index) \
        is False


def test_is_symetrical():
    """
    test is matrix is symmetrical
    """
    mat = np.random.rand(2, 2)
    print(mat)
    assert not is_symetrical(mat)

    mat = mat + np.transpose(mat)
    print(mat)

    assert is_symetrical(mat)
