import numpy as np
import os
import trimesh
from natsort import natsorted
import cv2

from vis.visualise_vertex_deviations import visualise_vertex_deviations_for_actor, visualise_vertex_deviations_all

def compute_vertex_deviations_over_sequence(path):
    plyfiles = sorted([f for f in os.listdir(path) if f.endswith('.ply')])
    all_squared_deviations = []
    if plyfiles:
        for idx, file in enumerate(plyfiles):
            mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
            vertices = mesh.vertices
            if idx == 0:
                initial_vertices = np.copy(vertices)
            else:
                deviations = np.sum((vertices - initial_vertices) ** 2, axis=-1)
                all_squared_deviations.append(deviations)
        return np.stack(all_squared_deviations, axis=0)


def compute_vertex_deviations_for_actor(actor_path, save_deviation_image=True):
    seq_paths = natsorted([os.path.join(actor_path, f) for f in os.listdir(actor_path)
                           if os.path.isdir(os.path.join(actor_path, f))])
    all_deviations_array = []
    for path in seq_paths:
        print(path)
        seq_deviations = compute_vertex_deviations_over_sequence(path)
        if seq_deviations is not None:
            all_deviations_array.append(seq_deviations)
    all_deviations_array = np.concatenate(all_deviations_array, axis=0)
    print(all_deviations_array.shape)
    if save_deviation_image:
        initial_mesh_path = sorted([os.path.join(seq_paths[0], f)for f in os.listdir(seq_paths[0])
                                    if f.endswith('.ply')])[0]
        initial_mesh = trimesh.load(initial_mesh_path, file_type='ply', process=False)
        deviation_image = visualise_vertex_deviations_for_actor(initial_mesh, all_deviations_array)
        cv2.imwrite(os.path.join(actor_path, 'deviation_image.png'), cv2.cvtColor(deviation_image, cv2.COLOR_RGB2BGR))

    return all_deviations_array


def compute_vertex_deviations_for_all(d3dfacs_path, save_deviation_image=True):
    actor_paths = natsorted([os.path.join(d3dfacs_path, f) for f in os.listdir(d3dfacs_path)
                           if os.path.isdir(os.path.join(d3dfacs_path, f))])
    all_deviations_array = []
    for idx, path in enumerate(actor_paths):
        actor_deviations = compute_vertex_deviations_for_actor(path, save_deviation_image=True)
        all_deviations_array.append(actor_deviations)
    all_deviations_array = np.concatenate(all_deviations_array, axis=0)
    print('SHAPE', all_deviations_array.shape)
    print('MAX', all_deviations_array.max())
    if save_deviation_image:
        deviation_image = visualise_vertex_deviations_all(all_deviations_array)
        cv2.imwrite(os.path.join(d3dfacs_path, 'all_deviation_image.png'),
                    cv2.cvtColor(deviation_image, cv2.COLOR_RGB2BGR))


