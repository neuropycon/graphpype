"""
Little utils functions, was in define_variables before
Specific to some nipype usage (inserted in connect, to check validity of
inputs/outputs between nodes with out defining an intermediate node for that
sole purpopse)

Author:
    David Meunier <david_meunier_79@hotmail.fr>
"""
import random
import numpy as np
import os
import shutil
# fast util functions for getting first or second element (mostly in tuple)


def get_first(string_list):
    """get first element"""
    if not isinstance(string_list, (list, tuple)):
        raise ValueError("{} should be iterables".format(string_list))

    if len(string_list) == 0:
        raise ValueError("{} should have at least one \
            element".format(string_list))

    return string_list[0]


def get_second(string_list):
    """get second element"""
    if not isinstance(string_list, (list, tuple)):
        raise ValueError("{} should be iterables".format(string_list))

    if len(string_list) < 2:
        raise ValueError("{} should have at least 2 \
            elements".format(string_list))
    return string_list[1]


def show_length(files):
    """check length of list (without creating a new node)"""
    if not isinstance(files, (list, tuple)):
        raise ValueError("{} should be iterables".format(files))

    print(len(files))
    return files


def show_files(files):
    """ check content of list (filenames) (without creating a new node)"""
    if not isinstance(files, (list, tuple)):
        raise ValueError("{} should be iterables".format(files))

    print(files)
    return files


def get_multiple_indexes(cur_list, item):
    """equivalent to list::index(), but return all indices where item
    is present (instead of only the first encounterd"""
    return [i for i, list_item in enumerate(cur_list) if list_item == item]


def random_product(*args, **kwds):
    # Taken from somewhere I do not recall....
    """Random selection from itertools.product(*args, **kwds)"""
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)


def check_dimensions(indexes, shape):
    """ check if 3D indices and given shape are compatible"""
    # TODO check if the modified function is compatible with previous
    # implementation
    if indexes[0] >= 0 and indexes[0] < shape[0]:
        if indexes[1] >= 0 and indexes[1] < shape[1]:
            if indexes[2] >= 0 and indexes[2] < shape[2]:
                return True
    return False


def check_np_shapes(np_shape1, np_shape2):
    """ check if both shapes have the same length and the same values"""
    # TODO check if the modified function is compatible with previous
    # implementation
    assert isinstance(np_shape1, tuple) and isinstance(np_shape2, tuple), \
        ("Error, arguments passed to check_np_shapes should be shapes")

    if len(np_shape1) == len(np_shape2):
        if np.all(np_shape1 == np_shape2):
            return True
        else:
            print("Warning, number of elements for dimension {} is different \
                  {}".format(np_shape1, np_shape2))
    return False


def check_np_dimension(np_shape, np_coords):
    """check if one coord is compatible with ndarray shape"""
    # TODO surely a better version can be done...
    # check dimensions
    if len(np_shape) != np_coords.shape[0]:
        print("Warning dimensions for nd array {} and coords {} do not \
               match".format(len(np_shape), np_coords.shape[0]))
        return False

    for dim in range(len(np_shape)):

        if np_shape[dim] <= np_coords[dim]:

            print("Warning nb elements for nd array {} and coord {} do not \
                  match (dimension %d)".format(np_shape[dim], np_coords[dim],
                                               dim))
            return False

        if np_coords[dim] < 0:
            print("Warning negative coord {} (dimension \
                  {})".format(np_coords[dim], dim))
            return False
    return True


def is_symetrical(mat):
    """test if a matrix is is_symetrical i.e. upper triangle part is the same
    as lower triangle part"""
    triu_mat = mat[np.triu_indices(mat.shape[0], k=1)]
    tril_mat = mat[np.tril_indices(mat.shape[0], k=-1)]
    return np.all(triu_mat == tril_mat)


def _make_tmp_dir():
    tmp_dir = "/tmp/test_graphpype"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)
    return tmp_dir
