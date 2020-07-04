import numpy as np
from matplotlib import cm
from FLAME import FLAME
import torch

from config import get_config
from renderer.weak_perspective_pyrender_renderer import Renderer


def visualise_vertex_deviations_for_actor(base_mesh, deviations, resolution=(512, 512)):
    mean_deviations = np.mean(deviations, axis=0)
    mean_deviations = mean_deviations / mean_deviations.max()
    inferno = cm.get_cmap('inferno', mean_deviations.shape[0])
    vertex_colours = inferno(mean_deviations)
    base_mesh.visual.vertex_colors = vertex_colours
    renderer = Renderer(faces=None, resolution=resolution)
    cam = np.array([5.0, 0., 0.])
    rend_img1 = renderer.render(None, cam, mesh=base_mesh, angle=-30, axis=[1, 0, 0])
    rend_img2 = renderer.render(None, cam, mesh=base_mesh, angle=-30, axis=[1, 0, 0])
    rend_img3 = renderer.render(None, cam, mesh=base_mesh, angle=-45, axis=[0, 1, 0])
    rend_img4 = renderer.render(None, cam, mesh=base_mesh, angle=90, axis=[0, 1, 0])
    rend_img_top = np.concatenate([rend_img1, rend_img2], axis=1)
    rend_img_bot = np.concatenate([rend_img3, rend_img4], axis=1)
    rend_img = np.concatenate([rend_img_top, rend_img_bot], axis=0)
    return rend_img


def visualise_vertex_deviations_all(deviations, resolution=(512, 512)):
    config = get_config()
    flamelayer = FLAME(config)
    shape_params = torch.zeros(1, 100)
    shape_params[:, 0] = -2
    pose_params = torch.zeros(1, 6)
    # pose_params[:, 2] = np.pi
    expression_params = torch.zeros(1, 50, dtype=torch.float32)
    neck_pose = torch.zeros(1, 3)
    eye_pose = torch.zeros(1, 6)
    vertices, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
    vertices = vertices[0].cpu().detach().numpy()

    mean_deviations = np.mean(deviations, axis=0)
    mean_deviations = mean_deviations / mean_deviations.max()
    inferno = cm.get_cmap('inferno', mean_deviations.shape[0])
    vertex_colours = inferno(mean_deviations)

    renderer = Renderer(faces=flamelayer.faces, resolution=resolution)
    cam = np.array([5.0, 0., 0.])
    rend_img1 = renderer.render(vertices, cam, vertex_colours=vertex_colours)
    rend_img2 = renderer.render(vertices, cam, angle=-45, axis=[0, 1, 0], vertex_colours=vertex_colours)
    rend_img3 = renderer.render(vertices, cam, angle=45, axis=[0, 1, 0], vertex_colours=vertex_colours)
    rend_img = np.concatenate([rend_img1, rend_img2, rend_img3], axis=1)
    return rend_img
