import os
import trimesh
import cv2
import numpy as np
from natsort import natsorted
from matplotlib import cm

from renderer.weak_perspective_pyrender_renderer import Renderer
from utils.rigid_transform_utils import compute_similarity_transform


def render_sequence(path, resolution=(512, 512), save=True):
    renderer = Renderer(faces=None, resolution=resolution)
    cam = np.array([5.0, 0., 0.05])
    plyfiles = sorted([f for f in os.listdir(path) if f.endswith('.ply')])
    img_array = []
    for file in plyfiles:
        mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
        rend_img = renderer.render(None, cam, mesh=mesh, angle=-30, axis=[1, 0, 0])
        img_array.append(rend_img)

    if save:
        out = cv2.VideoWriter(os.path.join(path, 'render.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, resolution)
        for i in range(len(img_array)):
            out.write(cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR))
        out.release()
    else:
        return img_array


def render_sequence_deviations(path, resolution=(512, 512), save=True, max_dev_for_colourmap=0.004,
                               rigid_transform=False, contour_indices=None):
    """
    Renders a sequence of actions from plyfiles. Vertex colours are determined by the deviation of the vertex from the
    resting face. The resting face is taken to be the first frame (i.e first plyfile) of the sequence.
    """
    renderer = Renderer(faces=None, resolution=resolution)
    cam = np.array([5.0, 0., 0.05])
    plyfiles = natsorted([f for f in os.listdir(path) if f.endswith('.ply')])
    img_array = []
    for idx, file in enumerate(plyfiles):
        mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
        if idx == 0:
            intial_vertices = mesh.vertices

        if rigid_transform:
            # Apply rigid transform to align mesh vertices with initial vertices in terms of global R and t (deals with small R,t alignment noise during registration)
            mesh_vertices = compute_similarity_transform(mesh.vertices, intial_vertices)
            mesh.vertices = mesh_vertices

        deviations = np.sqrt(np.sum((mesh.vertices - intial_vertices) ** 2, axis=-1))  # (num vertices,) - this is same as L2 norm
        # Scale deviations such that deviations >= max_dev_for_colourmap are set to 1
        deviations = deviations * (1 / max_dev_for_colourmap)
        deviations[deviations > 1] = 1

        if contour_indices is not None:
            deviations[[i for i in range(5023) if i not in contour_indices]] = 0.

        # Set vertex colour maps for render image
        cmap = cm.get_cmap('jet', deviations.shape[0])
        vertex_colours_for_render = cmap(deviations)
        mesh.visual.vertex_colors = vertex_colours_for_render

        rend_img = renderer.render(None, cam, mesh=mesh, angle=-30, axis=[1, 0, 0])
        img_array.append(rend_img)

    if save:
        out = cv2.VideoWriter(os.path.join(path, 'deviation_render.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, resolution)
        for i in range(len(img_array)):
            out.write(cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR))
        out.release()
    else:
        return img_array


def render_all_sequences_for_actor(actor_path, resolution=(400,400), colour_deviations=False,
                                   max_dev_for_colourmap=0.004, rigid_transform=False, contour_indices=None):
    seq_paths = natsorted([os.path.join(actor_path, f) for f in os.listdir(actor_path)
                        if os.path.isdir(os.path.join(actor_path, f))])
    all_img_array = []
    for path in seq_paths:
        print(path)
        if colour_deviations:
            img_array = render_sequence_deviations(path, resolution=resolution, save=False,
                                                   max_dev_for_colourmap=max_dev_for_colourmap,
                                                   rigid_transform=rigid_transform,
                                                   contour_indices=contour_indices)
        else:
            img_array = render_sequence(path, resolution=resolution, save=False)
        all_img_array += img_array

    if colour_deviations:
        out_fname = 'combined_deviation_render'
        if rigid_transform:
            out_fname += '_rigid_transform'
        if contour_indices is not None:
            out_fname += '_contour'
        out_fname += '.avi'
        out_path = os.path.join(actor_path, out_fname)
    else:
        out_path = os.path.join(actor_path, 'combined_render.avi')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 60, resolution)

    for i in range(len(all_img_array)):
        out.write(cv2.cvtColor(all_img_array[i], cv2.COLOR_RGB2BGR))
    out.release()

contour_indices = [2094, 2098, 2097, 2100, 1575, 3727, 3726, 3725, 3588, 3587, 3643, 3636, 3635, 3634, 3630,
                   3414, 3413, 3415, 3416, 3417, 3419, 3389, 3390, 3470, 3471, 3472, 2711, 3127, 3124, 3125,
                   3121, 3122, 3105, 3104, 2761, 3602, 2973, 3561, 1895, 3814, 1644, 2069, 2070, 2095, 2094]

render_all_sequences_for_actor("/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Joe",
                               resolution=(512, 512),
                               colour_deviations=True,
                               max_dev_for_colourmap=0.01,
                               rigid_transform=True,
                               contour_indices=contour_indices)





