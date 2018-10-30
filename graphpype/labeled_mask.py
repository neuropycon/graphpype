"""
Support for computing ROIs mask by different means:
- peak activations
- MNI coordinates
- ROI files (in nifti format)
- from template (Harvard Oxford = HO)

The outputs of the functions will always be a labeled mask, with values
starting from 1 (0 being the background image)
"""

import nipype.interfaces.spm as spm
from nipype.utils.filemanip import split_filename as split_f

from graphpype.utils import check_np_dimension

import itertools as iter

import numpy as np
import nibabel as nib
import glob
import os


from scipy import ndimage as ndimg
from scipy.spatial.distance import cdist


def _coord_transform(x, y, z, affine):
    """copied from nipy project"""
    """
    Convert the x, y, z coordinates from one image space to another space.

        Parameters
        ----------
        x : number or ndarray
            The x coordinates in the input space
        y : number or ndarray
            The y coordinates in the input space
        z : number or ndarray
            The z coordinates in the input space
        affine : 2D 4x4 ndarray
            affine that maps from input to output space.

        Returns
        -------
        x : number or ndarray
            The x coordinates in the output space
        y : number or ndarray
            The y coordinates in the output space
        z : number or ndarray
            The z coordinates in the output space

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
    """
    coords = np.c_[np.atleast_1d(x).flat,
                   np.atleast_1d(y).flat,
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


# from a list of MNI coords


def create_indexed_mask(ref_img_file, MNI_coords_list, ROI_dir,
                        ROI_mask_prefix="def", ROI_shape="cube", ROI_size=10):
    """
    Create indexed mask at the around ROI coords

        MNI_coords_list: list of list of 3 integer values in MNI space
        ref_img_file: nifti1 file, the generated indexed mask
        will use its shape and affine
        ROI_shape: "cube", or "sphere"
        ROI_size: ROI size in mm (from MNI space)
    """

    np_coord = np.array(MNI_coords_list)

    if len(np_coord.shape) > 1:

        dist = cdist(np_coord, np_coord, metric='euclidean')

        assert np.all(dist[np.triu_indices(dist.shape[0], k=1)]
                      > ROI_size), "Error, distance < {}".format(ROI_size)

    ref_img = nib.load(ref_img_file)

    # data (shape)
    ref_img_shape = ref_img.get_data().shape

    if len(ref_img_shape) == 4:

        print("using 4D image for computing 3D mask, reducing shape")

        ref_img_shape = ref_img_shape[:-1]

    print(ref_img_shape)

    # affine
    ref_img_affine = ref_img.affine
    inv_affine = np.linalg.inv(ref_img_affine)

    # header
    ref_img_hd = ref_img.header
    pixdims = ref_img_hd['pixdim'][1:4]

    # building indexed mask
    indexed_mask_data = np.zeros(shape=ref_img_shape) - 1

    # shape of the ROI
    if ROI_shape not in ["sphere", "cube"]:

        print("Warning, could not determine shape {}, using cube instead"
              .format(ROI_shape))

        ROI_shape = "cube"

    if ROI_shape == "cube":

        print("ROI_shape = cube")

        vox_dims = list(map(int, float(ROI_size)/pixdims))

        print(vox_dims)

        neigh_range = []

        for vox_dim in vox_dims:

            vox_neigh = vox_dim/2

            # case odd vox_dim
            if vox_dim % 2 == 1:

                cur_range = np.arange(-vox_neigh, vox_neigh+1)

            elif vox_dim % 2 == 0:

                cur_range = np.arange(-vox_neigh+1, vox_neigh+1)

            neigh_range.append(cur_range)

        ROI_coords = []

        for index_mask, MNI_coords in enumerate(MNI_coords_list):

            ijk_coord = _coord_transform(MNI_coords[0], MNI_coords[1],
                                         MNI_coords[2], inv_affine)

            neigh_coords = np.array(
                [list(i) for i in iter.product(*neigh_range)], dtype=int)

            cur_coords = np.array([list(map(int, ijk_coord + neigh_coord))
                                   for neigh_coord in neigh_coords])

            max_i, max_j, max_k = indexed_mask_data.shape

            keep = (0 <= cur_coords[:, 0]) & (cur_coords[:, 0] < max_i) & \
                (0 <= cur_coords[:, 1]) & (cur_coords[:, 1] < max_j) & \
                (0 <= cur_coords[:, 2]) & (cur_coords[:, 2] < max_k)

            if np.all(keep is False):
                continue

            indexed_mask_data[cur_coords[keep, 0], cur_coords[keep, 1],
                              cur_coords[keep, 2]] = index_mask

            print(np.sum(indexed_mask_data == index_mask))
            ROI_coords.append(ijk_coord)

    elif ROI_shape == "sphere":

        print("building spheres of {} mm".format(ROI_size))

        radius = ROI_size/2.0

        print(radius)

        vox_dims = list(map(int, float(radius)/pixdims))

        print(vox_dims)

        r2_dim = []
        neigh_range = []

        for i, vox_dim in enumerate(vox_dims):

            pixdim = pixdims[i]

            cur_range = np.arange(-vox_dim, (vox_dim+1))

            print(cur_range)

            cur_r2 = (cur_range*pixdim)**2

            print(cur_r2)

            neigh_range.append(cur_range.tolist())

            r2_dim.append(cur_r2)

        print(neigh_range)

        neigh_coords = np.array(
            [list(i) for i in iter.product(*neigh_range)], dtype=int)

        neigh_dist = np.array([np.sum(i) for i in iter.product(*r2_dim)])

        neigh_range = neigh_coords[neigh_dist < radius**2]

        ROI_coords = []

        # pour les ROI
        for index_mask, MNI_coords in enumerate(MNI_coords_list):

            ijk_coord = np.dot(inv_affine, np.array(
                MNI_coords + [1], dtype='int'))[:-1]

            ROI_coords.append(ijk_coord)

            cur_coords = np.array([list(map(int, ijk_coord + neigh_coord))
                                   for neigh_coord in neigh_range.tolist()])

            indexed_mask_data[cur_coords[:, 0],
                              cur_coords[:, 1], cur_coords[:, 2]] = index_mask

            print(np.sum(indexed_mask_data == index_mask))

    try:
        os.makedirs(ROI_dir)

    except OSError:
        print("directory already created")

    indexed_mask_file = os.path.join(
        ROI_dir, "indexed_mask-" + ROI_mask_prefix + ".nii")

    # save ROI_coords_labelled_mask
    nib.save(nib.Nifti1Image(indexed_mask_data,
                             ref_img_affine), indexed_mask_file)

    ROI_coords_file = os.path.join(
        ROI_dir, "ROI_coords-" + ROI_mask_prefix + ".txt")

    # save np coords
    np.savetxt(ROI_coords_file, np.array(ROI_coords, dtype=int), fmt="%d")

    return indexed_mask_file

# from a list of MNI coords (output one VOI binary mask nii image)


def compute_ROI_nii_from_ROI_coords_files(
        ref_img_file, MNI_coords_file, labels_file, neighbourhood=1):
    """
    Export single file VOI binary nii image
    """
    ref_image = nib.load(ref_img_file)
    ref_image_data = ref_image.get_data()
    ref_image_data_shape = ref_image_data.shape
    ref_image_data_sform = ref_image.get_sform()

    ROI_MNI_coords_list = np.array(np.loadtxt(
        MNI_coords_file), dtype='int').tolist()

    ROI_labels = [lign.strip() for lign in open(labels_file)]

    # transform MNI coords to numpy coords
    mni_sform_inv = np.linalg.inv(ref_image_data_sform)

    ROI_coords = np.array([_coord_transform(x, y, z, mni_sform_inv)
                           for x, y, z in ROI_MNI_coords_list], dtype="int64")

    for i, ROI_coord in enumerate(ROI_coords):

        ROI_coords_labelled_mask = np.zeros(
            shape=ref_image_data_shape, dtype='int64')

        neigh_range = list(range(-neighbourhood, neighbourhood+1))

        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x, neigh_y, neigh_z = ROI_coord + relative_coord

            print(neigh_x, neigh_y, neigh_z)

            if check_np_dimension(ROI_coords_labelled_mask.shape,
                                  np.array([neigh_x, neigh_y, neigh_z],
                                           dtype='int64')):

                ROI_coords_labelled_mask[neigh_x, neigh_y, neigh_z] = 1

        print(ROI_coords_labelled_mask)

        path, fname, ext = split_f(MNI_coords_file)

        ROI_coords_labelled_mask_file = os.path.join(
            path, "ROI_{}-neigh_{}_2.nii".format(ROI_labels[i],
                                                 str(neighbourhood)))

        # save ROI_coords_labelled_mask
        nib.save(nib.Nifti1Image(
            ROI_coords_labelled_mask, ref_image.affine,
            ref_image.header), ROI_coords_labelled_mask_file)

    return ROI_coords_labelled_mask_file


def compute_labelled_mask_from_anat_ROIs(
        ref_img_file, ROI_dir, list_ROI_img_files=[]):
    """
    compute labelled_mask from a list of img files,
    presenting ROIs extracted from MRIcron in the nii or img format
    each ROI is represented by a different IMG file and
    should start by 'ROI_'. Resampling is done based on the shape of
    ref_img_file
    """

    ref_image = nib.load(ref_img_file)

    ref_image_data = ref_image.get_data()

    ref_image_data_shape = ref_image_data.shape

    # case ref is 4D
    if len(ref_image_data_shape) != 3:

        mean_ref_data = np.mean(ref_image_data, axis=3)

        mean_ref_img_file = os.path.join(ROI_dir, "mean_ref.nii")

        nib.save(nib.Nifti1Image(mean_ref_data, ref_image.affine,
                                 ref_image.header), mean_ref_img_file)

        # reloading mean_file as ref_file
        ref_image = nib.load(mean_ref_img_file)

        ref_image_data = ref_image.get_data()

        ref_image_data_shape = ref_image_data.shape

    if len(list_ROI_img_files) == 0:

        resliced_ROI_files = glob.glob(os.path.join(ROI_dir, "rROI*.nii"))

        ROI_files = glob.glob(os.path.join(ROI_dir, "ROI*.nii"))

        if len(resliced_ROI_files) != len(ROI_files):

            for i, ROI_file in enumerate(ROI_files):

                ROI_image = nib.load(ROI_file)

                ROI_data = ROI_image.get_data()

                print("Original ROI template {} shape: {}".
                      format(i, ROI_data.shape))

                reslice_ROI = spm.Reslice()
                reslice_ROI.inputs.in_file = ROI_file
                reslice_ROI.inputs.space_defining = ref_img_file

                resliced_ROI_file = reslice_ROI.run().outputs.out_file

            resliced_ROI_files = glob.glob(os.path.join(ROI_dir, "rROI*.nii"))
    else:

        ROI_files = [os.path.join(ROI_dir, ROI_img_file)
                     for ROI_img_file in list_ROI_img_files]

        print(ROI_files)

        resliced_ROI_files = []

        for ROI_img_file in list_ROI_img_files:
            full_ROI_img_file = os.path.join(ROI_dir, "r"+ROI_img_file)
            if os.path.exists(full_ROI_img_file):
                resliced_ROI_files.append(full_ROI_img_file)

        print(resliced_ROI_files)
        print(len(resliced_ROI_files))

        if len(resliced_ROI_files) != len(ROI_files):

            resliced_ROI_files = []

            for i, ROI_file in enumerate(ROI_files):

                ROI_image = nib.load(ROI_file)

                ROI_data = ROI_image.get_data()

                print("Original ROI template {} shape: {}".
                      format(i, ROI_data.shape))

                reslice_ROI = spm.Reslice()
                reslice_ROI.inputs.in_file = ROI_file
                reslice_ROI.inputs.space_defining = ref_img_file

                resliced_ROI_file = reslice_ROI.run().outputs.out_file

                resliced_ROI_files.append(resliced_ROI_file)

    resliced_ROI_files.sort()

    print(resliced_ROI_files)
    print(len(resliced_ROI_files))

    labels = []

    labelled_mask_data = np.zeros(shape=ref_image_data.shape, dtype='int') - 1

    print(labelled_mask_data.shape)

    for i, resliced_ROI_file in enumerate(resliced_ROI_files):

        print(i)

        path, fname, ext = split_f(resliced_ROI_file)

        labels.append(fname)

        resliced_ROI_img = nib.load(resliced_ROI_file)

        resliced_ROI_data = resliced_ROI_img.get_data()

        print(resliced_ROI_data.shape)

        print(np.sum(resliced_ROI_data != 0))

        labelled_mask_data[resliced_ROI_data != 0] = i

        print(np.unique(labelled_mask_data))

    print(np.unique(labelled_mask_data).shape)

    print(len(labels))

    # save labeled_mask
    labelled_mask_data_file = os.path.join(
        ROI_dir, "all_ROIs_labelled_mask.nii")

    nib.save(nib.Nifti1Image(labelled_mask_data, ref_image.affine,
                             ref_image.header), labelled_mask_data_file)

    # save labels
    labels_list_file = os.path.join(ROI_dir, "labels_all_ROIs.txt")
    np.savetxt(labels_list_file, np.array(labels, dtype='string'), fmt="%s")

    return labelled_mask_data_file, labels_list_file


def compute_MNI_coords_from_indexed_template(indexed_template_file):
    """
    compute MNI coords from an indexed template
    """
    path, base, ext = split_f(indexed_template_file)

    print(base)

    if len(base.split("-")) > 1:
        base_name = base.split("-")[1]
    else:
        base_name = base

    ref_image = nib.load(indexed_template_file)

    ref_image_data = ref_image.get_data()

    print(ref_image_data.shape)

    ref_image_affine = ref_image.affine

    print(ref_image_affine)

    ROI_coords = []

    ROI_MNI_coords = []

    for index in np.unique(ref_image_data)[1:]:

        i, j, k = np.where(ref_image_data == index)

        mean_coord_ijk = np.mean(np.array((i, j, k)), axis=1)

        print(mean_coord_ijk)

        ROI_coords.append(mean_coord_ijk)

        MNI_coord = np.dot(ref_image_affine, np.append(mean_coord_ijk, 1))

        print(MNI_coord)

        ROI_MNI_coords.append(MNI_coord[:3])

    ROI_coords = np.array(ROI_coords, dtype=float)

    print(ROI_coords)

    ROI_MNI_coords = np.array(ROI_MNI_coords, dtype=float)

    print(ROI_MNI_coords)

    ROI_coords_file = os.path.join(path, "ROI_coords-" + base_name + ".txt")

    np.savetxt(ROI_coords_file, ROI_coords, fmt="%.3f %.3f %.3f")

    ROI_MNI_coords_file = os.path.join(
        path, "ROI_MNI_coords-" + base_name + ".txt")

    np.savetxt(ROI_MNI_coords_file, ROI_MNI_coords, fmt="%.3f %.3f %.3f")

    return ROI_coords_file, ROI_MNI_coords_file
    # nib.load(ref_img_file)


def segment_mask_in_ROI(
        mask_file, save_dir=0, segment_type="cube", mask_thr=0.99,
        min_count_voxel_in_ROI=100, cub_size=1, min_frac_vox_in_bin_mask=0.5):

    print(mask_file)

    if not save_dir:
        save_dir, fname, ext = split_f(mask_file)

    # load mask
    mask = nib.load(mask_file)
    mask_data = mask.get_data()
    mask_header = mask.header
    mask_affine = mask.affine

    i_mask, j_mask, k_mask = mask_data.shape

    if 'int' in str(mask_data.dtype):
        bin_mask_data = mask_data

    else:
        bin_mask_data = np.zeros(shape=mask_data.shape, dtype='int64')
        bin_mask_data[mask_data > mask_thr] = 1

    if segment_type == "cube":

        indexed_mask_data = np.zeros(
            shape=bin_mask_data.shape, dtype='int64')-1

        i_coords = np.arange(cub_size, i_mask-cub_size, step=2*cub_size+1)
        j_coords = np.arange(cub_size, j_mask-cub_size, step=2*cub_size+1)
        k_coords = np.arange(cub_size, k_mask-cub_size, step=2*cub_size+1)

        cub_range = np.arange(-cub_size, cub_size+1)
        print(cub_range)

        val = 0
        for x, y, z in iter.product(i_coords, j_coords, k_coords):

            vox = []
            vox_in_bin = []

            for neigh in iter.product(cub_range, repeat=3):
                if bin_mask_data[x+neigh[0], y+neigh[1], z+neigh[2]] == 1:
                    vox_in_bin.append(neigh)
                vox.append(neigh)

            if len(vox_in_bin) == 0:
                continue

            frac_vox_in_bin_mask = len(vox)/float(len(vox_in_bin))

            if frac_vox_in_bin_mask > min_frac_vox_in_bin_mask:
                for neigh in iter.product(cub_range, repeat=3):
                    indexed_mask_data[x+neigh[0], y+neigh[1], z+neigh[2]] = val
                val = val + 1

            else:
                print("not enough voxels in bin_mask, percent = {}".format(
                    frac_vox_in_bin_mask))

            print(val)

        ROI_mask_prefix = segment_type + "_ROI_" + \
            str(cub_size) + "_min_frac_" + str(min_frac_vox_in_bin_mask)

    elif segment_type == 'disjoint_comp':

        print("computing disjoint cluster")

        ROI_mask_prefix = segment_type + "_ROI_" + str(min_count_voxel_in_ROI)

        raw_indexed_mask_rois_data = ndimg.label(bin_mask_data)[0]

        print(np.unique(raw_indexed_mask_rois_data))

        for index_ROI in np.unique(raw_indexed_mask_rois_data):
            count_voxel_in_ROI = np.sum(
                raw_indexed_mask_rois_data == index_ROI)

            print(index_ROI, count_voxel_in_ROI)

            if count_voxel_in_ROI < min_count_voxel_in_ROI:
                raw_indexed_mask_rois_data[raw_indexed_mask_rois_data ==
                                           index_ROI] = 0

        print(np.unique(raw_indexed_mask_rois_data))

        # reordering indexes
        indexed_mask_data = np.zeros(
            shape=raw_indexed_mask_rois_data.shape)

        for i, index_ROI in enumerate(np.unique(raw_indexed_mask_rois_data)):
            indexed_mask_data[raw_indexed_mask_rois_data == index_ROI] = i-1

    else:
        raise ValueError("Error, could not find segment_type {}".format(
            segment_type))

    # TODO if segment_type == "voxel"
    indexed_mask_rois_file = os.path.join(
        save_dir, "indexed_mask-" + ROI_mask_prefix + ".nii")

    nib.save(nib.Nifti1Image(
        dataobj=indexed_mask_data, header=mask_header,
        affine=mask_affine), indexed_mask_rois_file)

    ROI_coords_file, ROI_MNI_coords_file = \
        compute_MNI_coords_from_indexed_template(indexed_mask_rois_file)

    return indexed_mask_rois_file, ROI_coords_file, ROI_MNI_coords_file
