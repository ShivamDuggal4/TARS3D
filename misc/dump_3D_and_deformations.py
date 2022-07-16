import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch


def create_all_meshes(opt, var):
    for i in var.idx:
        filename = "{}/dump/{}_{}".format(opt.output_path, i, "color_coded_mesh")
        create_mesh(opt, var, filename)


def create_mesh(opt, var, filename, N=128, max_batch=64 ** 3, offset=None, scale=None, level=0.0, get_color=True):
    start = time.time()
    ply_filename = filename

    voxel_origin = [-0.6*2.0, -0.6*2.0, -0.6*2.0]
    voxel_size = 2 * 1.2 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head:min(head + max_batch, num_samples), 0:3].to(opt.device)[None, ...]
        samples[head:min(head + max_batch, num_samples), 3] = (
            var.impl_func(var, sample_subset)
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_with_color_to_ply(
        opt, var,
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
        level
    )


def get_mesh_color(mesh_points, opt, var):
    mesh_colors = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]

    max_batch = 64**3
    head = 0
    while head < num_samples:
        sample_subset = \
            torch.from_numpy(mesh_points[head:min(head + max_batch, num_samples), 0:3]).float().cuda()[None, ...]
        mesh_colors[head:min(head + max_batch, num_samples), 0:3] = (
            (sample_subset + var.impl_func(var, sample_subset, get_deformations=True)[1])
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    return mesh_colors


def convert_sdf_samples_with_color_to_ply(
    opt,
    var,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):

    start_time = time.time()
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except Exception:
        pass

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    mesh_colors = get_mesh_color(mesh_points, opt, var)

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    colors_tuple = np.zeros((num_verts,), dtype=[("red", "f4"), ("green", "f4"), ("blue", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in range(0, num_verts):
        colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts, verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces], text=True)
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
