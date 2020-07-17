import numpy as np
import os
import trimesh
from natsort import natsorted
import cv2

from vis.visualise_vertex_deviations import visualise_vertex_deviations_for_actor, visualise_vertex_deviations_all
from utils.rigid_transform_utils import compute_similarity_transform

def compute_vertex_deviations_over_sequence(path, rigid_transform=False):
    """
    Computes squared deviations of all face vertices over a sequence of actor actions. Deviations are computed from
    a resting face, which is taken to be the first frame of the action sequence.
    path: path to folder of action sequence ply files.
    """
    plyfiles = sorted([f for f in os.listdir(path) if f.endswith('.ply')])
    all_squared_deviations = []
    if plyfiles:
        for idx, file in enumerate(plyfiles):
            mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
            vertices = mesh.vertices
            if idx == 0:
                initial_vertices = np.copy(vertices)
            else:
                if rigid_transform:
                    # Apply rigid transform to align mesh vertices with initial vertices in terms of global R and t (deals with small R,t alignment noise during registration)
                    vertices = compute_similarity_transform(vertices, initial_vertices)
                deviations = np.sum((vertices - initial_vertices) ** 2, axis=-1)
                all_squared_deviations.append(deviations)
        return np.sqrt(np.stack(all_squared_deviations, axis=0))  # (seq length, num vertices)


def compute_vertex_deviations_for_actor(actor_path, save_deviation_image=True, reduce='mean',
                                        normalise_render=False, rigid_transform=False):
    """
    Computes squared deviations of face vertices over ALL action sequences for a particular actor.
    Deviations are computed from a resting face, which is taken to be the first frame of EACH action sequence.

    actor_path: path to folder of folders of action sequence ply files for a particular actor.
    save_deviation_image: bool flag, if true save a visualisation of per-vertex squared deviations
    """
    seq_paths = natsorted([os.path.join(actor_path, f) for f in os.listdir(actor_path)
                           if os.path.isdir(os.path.join(actor_path, f))])
    all_deviations_array = []
    for path in seq_paths:
        print(path)
        seq_deviations = compute_vertex_deviations_over_sequence(path,
                                                                         rigid_transform=rigid_transform)  # (seq length, num vertices)
        if seq_deviations is not None:
            all_deviations_array.append(seq_deviations)
    all_deviations_array = np.concatenate(all_deviations_array, axis=0)  # (sum of seq lengths, num vertices)
    print(all_deviations_array.shape)

    if save_deviation_image:
        # Saving visualisation of deviations
        initial_mesh_path = sorted([os.path.join(seq_paths[0], f)for f in os.listdir(seq_paths[0])
                                    if f.endswith('.ply')])[0]
        initial_mesh = trimesh.load(initial_mesh_path, file_type='ply', process=False)
        deviation_image, scatter_fig = visualise_vertex_deviations_for_actor(initial_mesh, all_deviations_array,
                                                                             reduce=reduce,
                                                                             normalise_render=normalise_render)

        deviation_out_fname = '{}_deviation_image'.format(reduce)
        scatter_out_fname = '{}_scatter_plot'.format(reduce)
        if rigid_transform:
            deviation_out_fname += '_rigid_transform'
            scatter_out_fname += '_rigid_transform'
        cv2.imwrite(os.path.join(actor_path, deviation_out_fname + '.png'),
                    cv2.cvtColor(deviation_image, cv2.COLOR_RGB2BGR))
        scatter_fig.savefig(os.path.join(actor_path, scatter_out_fname + '.png'))

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


