import os
import trimesh
import cv2
import numpy as np
from natsort import natsorted

from renderer.weak_perspective_pyrender_renderer import Renderer


def render_sequence(path, resolution=(512, 512), save=True):
    renderer = Renderer(faces=None, resolution=resolution)
    cam = np.array([5.0, 0., 0.])
    plyfiles = sorted([f for f in os.listdir(path) if f.endswith('.ply')])
    img_array = []
    for file in plyfiles:
        mesh = trimesh.load(os.path.join(path, file), file_type='ply', process=False)
        rend_img = renderer.render(None, cam, mesh=mesh, angle=-30, axis=[1, 0, 0])
        img_array.append(rend_img)

    if save:
        out = cv2.VideoWriter(os.path.join(path, 'render.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, resolution)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    else:
        return img_array


def render_all_sequences_for_actor(actor_path, resolution=(512,512)):
    seq_paths = natsorted([os.path.join(actor_path, f) for f in os.listdir(actor_path)
                        if os.path.isdir(os.path.join(actor_path, f))])
    all_img_array = []
    for path in seq_paths:
        img_array = render_sequence(path, resolution=resolution, save=False)
        all_img_array += img_array

    out = cv2.VideoWriter(os.path.join(actor_path, 'combined_render.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, resolution)

    for i in range(len(all_img_array)):
        out.write(all_img_array[i])
    out.release()

