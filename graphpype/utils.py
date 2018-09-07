# -*- coding: utf-8 -*-
"""
Little utils functions, was in define_variables before
Specific to some nipype usage (inserted in connect, to check validity of inputs/outputs between nodes with out defining an intermediate node for that sole purpopse)
"""
import random

#### fonctions utilitaires vite faites ###################################


def get_first(string_list):
    # print string_list
    return string_list[0]


def get_second(string_list):
    return string_list[1]

# utilitaire pour vérifier la longueur d'une liste sans avoir à créer de nouveaux noeuds


def show_length(files):

    print(len(files))

    return files

# utilitaire pour vérifier une liste sans avoir à créer de nouveaux noeuds


def show_files(files):

    print(files)

    return files

# idem fonction list::index(), mais retourne tous les indices de la liste ou l'item est présent


def get_multiple_indexes(cur_list, item):

    return [i for i, list_item in enumerate(cur_list) if list_item == item]

# generate itertools.product in a random order


def random_product(*args, **kwds):
    "Random selection from itertools.product(*args, **kwds)"
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)

# test dimensions


def check_dimensions(indexes, shape):

    if indexes[0] >= 0 and indexes[0] < shape[0] and indexes[1] >= 0 and indexes[1] < shape[1] and indexes[2] >= 0 and indexes[2] < shape[2]:
        return True
    else:
        return False


def check_np_shapes(np_shape1, np_shape2):

    if len(np_shape1) != len(np_shape2):

        print("Warning, dimensions for nd array1 %d and nd array2 %d do not match" % (
            len(np_shape1), len(np_shape2)))

        return 0

    for i in range(len(np_shape1)):

        if np_shape1[i] != np_shape2[i]:

            print("Warning, number of elements for dimension %d is different : %d != %d" % (
                i, np_shape1[i], np_shape2[i]))

            return 0

    return 1


def check_np_dimension(np_shape, np_coords):

    # verification des dimensions
    if len(np_shape) != np_coords.shape[0]:

        print("Warning dimensions for nd array %d and coords %d do not match" %
              (len(np_shape), np_coords.shape[0]))

        return 0

    for dim in range(len(np_shape)):

        if np_shape[dim] <= np_coords[dim]:

            print("Warning nb elements for nd array %d and coord %d do not match (dimension %d)" % (
                np_shape[dim], np_coords[dim], dim))

            return 0

        if np_coords[dim] < 0:

            print("Warning negative coord %d (dimension %d)" %
                  (np_coords[dim], dim))

            return 0

    return 1
