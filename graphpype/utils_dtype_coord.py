# -*- coding: utf-8 -*-
"""
Support function to handle coordinates as list of single elements
specially convenient for use of 'in' with lists
"""

import numpy as np

coord_dt = np.dtype([('x', int), ('y', int), ('z', int)])

def convert_np_coords_to_coords_dt(np_coords):

    coords_dt = np_coords.view(coord_dt).reshape(-1)

    return coords_dt

def convert_coords_dt_to_np_coords(coords_dt):

    np_coords = coords_dt.view(int).reshape(-1, 3)

    return np_coords

def is_in_coords(np_coords1, np_coords2):

    coords_dt1 = convert_np_coords_to_coords_dt(np_coords1)
    coords_dt2 = convert_np_coords_to_coords_dt(np_coords2)

    tab = np.array([e in coords_dt1 for e in coords_dt2], dtype=bool)
    
    return tab

def where_in_coords(np_coords1, np_coords2):

    return np.array(np.where(is_in_coords(np_coords1, np_coords2)), dtype='int64').reshape(-1)

def coords_dt_equals(coords_dt1, coords_dt2):

    return coords_dt1['x'] == coords_dt2['x'] and coords_dt1['y'] == coords_dt2['y'] and coords_dt1['z'] == coords_dt2['z']

def find_index_in_coords_dt(coords_dt, coord_dt):

    for ind_c, c in enumerate(coords_dt):
        if coords_dt_equals(c, coord_dt):
            return ind_c
    return -1


def find_index_in_coords(np_coords1, np_coords2):

    find_index = np.zeros(np_coords1.shape[0], dtype='int64')-1

    coords_dt1 = convert_np_coords_to_coords_dt(np_coords1)
    coords_dt2 = convert_np_coords_to_coords_dt(np_coords2)

    for ind_c1, c1 in enumerate(coords_dt1):

        if c1 in coords_dt2:
            find_index[ind_c1] = find_index_in_coords_dt(coords_dt2, c1)

    return find_index
