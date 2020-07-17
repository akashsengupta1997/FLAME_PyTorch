import numpy as np
import os
import trimesh
from natsort import natsorted
import cv2

import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.rigid_transform_utils import compute_similarity_transform
from vis.visualise_vertex_deviations import visualise_vertex_deviations_for_actor


def compute_contour_vertex_deviations_over_sequence(path, base_contour_vertices, contour_indices,
                                                    apply_similarity_transform=False):
    """
    Computes squared deviations of vertices along given contour over a sequence of actor actions.

    path:
        path to folder of action sequence ply files.

    base_contour_vertices:
        array with shape (num contour verts, 3) specifying 3D locations of fixed base contour vertices
        from which deviations will be calculated.

    contour_indices:
        array with shape (num contour verts,) specifying the indices of the vertices which lie upon the
        chosen contour.

    apply_similarity_transform:
        bool flag, if True determine the optimal transformation V_trans = RV + t that minimises
        RMSD between V_trans and base_contour_vertices (where V = contour_vertices at each timestep of the sequence).
    """
    plyfiles = sorted([f for f in os.listdir(path) if f.endswith('.ply')])
    all_squared_deviations = []
    if plyfiles:
        for idx, file in enumerate(plyfiles):
            mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
            contour_vertices = mesh.vertices[contour_indices]
            # plt.figure()
            # plt.subplot(111, projection='3d')
            # plt.plot(xs=contour_vertices[:, 0], ys=contour_vertices[:, 1], zs=contour_vertices[:, 2], c='b')
            # plt.plot(xs=base_contour_vertices[:, 0], ys=base_contour_vertices[:, 1], zs=base_contour_vertices[:, 2], c='r')

            if apply_similarity_transform:
                contour_vertices = compute_similarity_transform(contour_vertices, base_contour_vertices)
                # plt.figure()
                # plt.subplot(111, projection='3d')
                # plt.plot(xs=contour_vertices[:, 0], ys=contour_vertices[:, 1], zs=contour_vertices[:, 2], c='b')
                # plt.plot(xs=base_contour_vertices[:, 0], ys=base_contour_vertices[:, 1], zs=base_contour_vertices[:, 2],
                #          c='r')
            # plt.show()

            deviations = np.sum((contour_vertices - base_contour_vertices) ** 2, axis=-1)  # (num contour verts,)
            all_squared_deviations.append(deviations)
        return np.stack(all_squared_deviations, axis=0)  # (seq length, num contour verts)


def compute_contour_vertex_deviations_for_actor_sequence(actor_path, base_contour_vertices, contour_indices,
                                                         apply_similarity_transform=False, save_deviation_image=True,
                                                         reduce='mean'):
    """
    Computes squared deviations of vertices along given contour over ALL action sequences for a particular actor.
    Deviations are computed from a resting face, which is taken to be the first frame of EACH action sequence.

    actor_path:
        path to folder of folders of action sequence ply files for a particular actor.

    base_contour_vertices:
        array with shape (num contour verts, 3) specifying 3D locations of fixed base contour vertices
        from which deviations will be calculated.

    contour_indices:
        array with shape (num contour verts,) specifying the indices of the vertices which lie upon the
        chosen contour.

    apply_similarity_transform:
        bool flag, if True determine the optimal transformation V_trans = RV + t that minimises
        RMSD between V_trans and base_contour_vertices (where V = contour_vertices at each timestep of the sequence).

    save_deviation_image:
        bool flag, if true save a visualisation of per-vertex squared deviations
    """
    seq_paths = natsorted([os.path.join(actor_path, f) for f in os.listdir(actor_path)
                           if os.path.isdir(os.path.join(actor_path, f))])
    all_deviations_array = []
    for path in seq_paths:
        print(path)
        seq_deviations = compute_contour_vertex_deviations_over_sequence(path, base_contour_vertices, contour_indices,
                                                                         apply_similarity_transform=apply_similarity_transform)  # (seq length, num contour verts)
        if seq_deviations is not None:
            all_deviations_array.append(seq_deviations)
    all_deviations_array = np.concatenate(all_deviations_array, axis=0)  # (sum of seq lengths, num contour verts)
    print(all_deviations_array.shape)
    if save_deviation_image:
        initial_mesh_path = sorted([os.path.join(seq_paths[0], f)for f in os.listdir(seq_paths[0])
                                    if f.endswith('.ply')])[0]
        all_deviations_for_vis = np.zeros((all_deviations_array.shape[0], 5023))  # (sum of seq lengths, num face verts)
        all_deviations_for_vis[:, contour_indices] = all_deviations_array
        initial_mesh = trimesh.load(initial_mesh_path, file_type='ply', process=False)
        deviation_image = visualise_vertex_deviations_for_actor(initial_mesh, all_deviations_for_vis, reduce=reduce)
        cv2.imwrite(os.path.join(actor_path, 'contour_dynamic_deviation_image.png'), cv2.cvtColor(deviation_image, cv2.COLOR_RGB2BGR))

    return all_deviations_array


def compute_contour_vertex_deviations_for_actor_static(actor_mesh_path, base_contour_vertices, contour_indices,
                                                       apply_similarity_transform=False, save_deviation_image=True):
    actor_mesh = trimesh.load(actor_mesh_path, file_type='ply', process=False)
    actor_contour_vertices = actor_mesh.vertices[contour_indices]
    plt.figure()
    plt.subplot(111, projection='3d')
    plt.plot(xs=actor_contour_vertices[:, 0], ys=actor_contour_vertices[:, 1], zs=actor_contour_vertices[:, 2], c='b')
    plt.plot(xs=base_contour_vertices[:, 0], ys=base_contour_vertices[:, 1], zs=base_contour_vertices[:, 2], c='r')

    if apply_similarity_transform:
        actor_contour_vertices = compute_similarity_transform(actor_contour_vertices, base_contour_vertices)
        plt.figure()
        plt.subplot(111, projection='3d')
        plt.plot(xs=actor_contour_vertices[:, 0], ys=actor_contour_vertices[:, 1], zs=actor_contour_vertices[:, 2], c='b')
        plt.plot(xs=base_contour_vertices[:, 0], ys=base_contour_vertices[:, 1], zs=base_contour_vertices[:, 2],
                 c='r')
    # plt.show()

    deviations = np.sqrt(np.sum((actor_contour_vertices - base_contour_vertices) ** 2, axis=-1))  # (num contour verts,)
    if save_deviation_image:
        deviations_for_vis = np.zeros(5023)
        deviations_for_vis[contour_indices] = deviations
        scatter_sizes = np.ones(5023) * 5.
        scatter_sizes[contour_indices] = 25.
        deviation_image, fig = visualise_vertex_deviations_for_actor(actor_mesh, deviations_for_vis, reduce=None,
                                                                     normalise_render=False, scatter_sizes=scatter_sizes,
                                                                     contour_indices=contour_indices)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(actor_mesh_path)),
                                 'contour_static_deviation_image.png'),
                    cv2.cvtColor(deviation_image, cv2.COLOR_RGB2BGR))
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(actor_mesh_path)),
                                 'contour_static_deviation_scatter_plot.png'))
    return deviations


