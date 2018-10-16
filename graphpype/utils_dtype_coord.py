"""
Support function to handle coordinates as list of single elements
specially convenient for use of 'in' with lists
"""
import numpy as np

coord_dt = np.dtype([('x', int), ('y', int), ('z', int)])


def _convert_np_coords_to_coords_dt(np_coords):
    """convert numpy coords to coord_dt"""
    return np_coords.view(coord_dt).reshape(-1)


def _convert_coords_dt_to_np_coords(coords_dt):
    """convert coord_dt to numpy coords """
    return coords_dt.view(int).reshape(-1, 3)


def _is_in_coords(np_coords1, np_coords2):
    """check if and where some coords1 are in coords2"""
    coords_dt1 = _convert_np_coords_to_coords_dt(np_coords1)
    coords_dt2 = _convert_np_coords_to_coords_dt(np_coords2)
    tab = np.array([e in coords_dt1 for e in coords_dt2], dtype=bool)
    return tab


def _coords_dt_equals(coords_dt1, coords_dt2):
    """operation of equality between 2 coord_dt objects"""
    equal_x = coords_dt1['x'] == coords_dt2['x']
    equal_y = coords_dt1['y'] == coords_dt2['y']
    equal_z = coords_dt1['z'] == coords_dt2['z']
    return equal_x and equal_y and equal_z


def _find_index_in_coords_dt(coords_dt, coord_dt):
    """find the first index of a coords_dt within a list of coord_dt(s)
    return -1 if not found"""
    for ind_c, c in enumerate(coords_dt):
        if _coords_dt_equals(c, coord_dt):
            return ind_c
    return -1


# public methods


def find_index_in_coords(np_coords1, np_coords2):
    """inverse operation of where_in_coords: add -1 when no correspondance
    are found"""
    find_index = np.zeros(np_coords1.shape[0], dtype='int64')-1

    coords_dt1 = _convert_np_coords_to_coords_dt(np_coords1)
    coords_dt2 = _convert_np_coords_to_coords_dt(np_coords2)

    for ind_c1, c1 in enumerate(coords_dt1):
        if c1 in coords_dt2:
            find_index[ind_c1] = _find_index_in_coords_dt(coords_dt2, c1)

    return find_index


def where_in_coords(np_coords1, np_coords2):
    """return indexes of  numpy coords1 in coords2"""
    indexes = np.where(_is_in_coords(np_coords1, np_coords2))
    return np.array(indexes, dtype='int64').reshape(-1)
