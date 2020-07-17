import numpy as np
from matplotlib import cm
from FLAME import FLAME
import torch
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt

from config import get_config
from renderer.weak_perspective_pyrender_renderer import Renderer


def visualise_vertex_deviations_for_actor(base_mesh, deviations, resolution=(512, 512), reduce='mean',
                                          normalise_render=False, scatter_sizes=None, contour_indices=None):
    """
    Visualises a scatter plot and rendered image of vertex deviations for an actor.
    """
    if reduce == 'mean':
        deviations = np.mean(deviations, axis=0)  # (num vertices,)
        max_dev_for_colourmap = 0.003
    elif reduce == 'max':
        deviations = np.max(deviations, axis=0)  # (num vertices,)
        max_dev_for_colourmap = 0.015
    else:
        max_dev_for_colourmap = 0.01
        print(deviations.max())

    # Scale deviations such that deviations >= max_dev_for_colourmap are set to 1 for scatter plots
    deviations_for_scatter = deviations * (1 / max_dev_for_colourmap)
    deviations_for_scatter[deviations_for_scatter > 1] = 1
    if normalise_render:
        # Normalise deviations to range 0-1, which is the full range of the colourmap, so it looks nice on the render image
        deviations_for_render = (deviations - deviations.min()) / (deviations.max() - deviations.min())
    else:
        deviations_for_render = deviations_for_scatter

    cmap = cm.get_cmap('jet', deviations.shape[0])

    # Set vertex colour maps for scatter plot and render image
    vertex_colours_for_scatter = cmap(deviations_for_scatter)
    scatter_depth_threshold = -1.003
    vertex_colours_for_render = cmap(deviations_for_render)
    base_mesh.visual.vertex_colors = vertex_colours_for_render
    renderer = Renderer(faces=None, resolution=resolution)
    cam = np.array([5.0, 0., 0.05])

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    if scatter_sizes is None:
        scatter_sizes = 16.

    rend_img1 = renderer.render(None, cam, mesh=base_mesh, angle=-30, axis=[1, 0, 0])
    plt.subplot(141)
    plt.gca().axis('off')
    plt.scatter(base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold*2.6, 0],
                base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold*2.6, 1],
                c=vertex_colours_for_scatter[base_mesh.vertices[:, 2] > scatter_depth_threshold*2.6],
                s=scatter_sizes[base_mesh.vertices[:, 2] > scatter_depth_threshold*2.6])
    if contour_indices is not None:
        # Re-plotting contour vertices so they are not obscured by other vertices
        plt.scatter(base_mesh.vertices[contour_indices, 0],  base_mesh.vertices[contour_indices, 1],
                    c=vertex_colours_for_scatter[contour_indices],
                    s=scatter_sizes[contour_indices])
    # for debugging - sanity check that colour map is working
    # plotted_vertices = base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, :]
    # plotted_deviations = deviations[base_mesh.vertices[:, 2] > scatter_depth_threshold]
    # plotted_vertex_colours = vertex_colours_for_scatter[base_mesh.vertices[:, 2] > scatter_depth_threshold]
    # print(plotted_vertices.shape, plotted_deviations.shape, plotted_vertex_colours.shape)
    # for i in range(plotted_vertices.shape[0]):
    #     if i % 10 == 0:
    #         plt.scatter(plotted_vertices[i, 0], plotted_vertices[i, 1], c=plotted_vertex_colours[i])
    #         plt.text(plotted_vertices[i, 0], plotted_vertices[i, 1], '{0:.1f}'.format(plotted_deviations[i] * 1000),
    #                  fontsize=6)
    # debugging end
    plt.gca().set_aspect('equal', adjustable='box')

    rend_img2 = renderer.render(None, cam, mesh=base_mesh, angle=-30, axis=[1, 0, 0])
    plt.subplot(142)
    plt.gca().axis('off')
    plt.scatter(base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 0],
                base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 1],
                c=vertex_colours_for_scatter[base_mesh.vertices[:, 2] > scatter_depth_threshold],
                s=scatter_sizes[base_mesh.vertices[:, 2] > scatter_depth_threshold])
    if contour_indices is not None:
        # Re-plotting contour vertices so they are not obscured by other vertices
        plt.scatter(base_mesh.vertices[contour_indices, 0], base_mesh.vertices[contour_indices, 1],
                    c=vertex_colours_for_scatter[contour_indices],
                    s=scatter_sizes[contour_indices])
    plt.gca().set_aspect('equal', adjustable='box')

    rend_img3 = renderer.render(None, cam, mesh=base_mesh, angle=-45, axis=[0, 1, 0])
    plt.subplot(143)
    plt.gca().axis('off')
    plt.scatter(base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 0],
                base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 1],
                c=vertex_colours_for_scatter[base_mesh.vertices[:, 2] > scatter_depth_threshold],
                s=scatter_sizes[base_mesh.vertices[:, 2] > scatter_depth_threshold])
    if contour_indices is not None:
        # Re-plotting contour vertices so they are not obscured by other vertices
        plt.scatter(base_mesh.vertices[contour_indices, 0], base_mesh.vertices[contour_indices, 1],
                    c=vertex_colours_for_scatter[contour_indices],
                    s=scatter_sizes[contour_indices])
    plt.gca().set_aspect('equal', adjustable='box')

    rend_img4 = renderer.render(None, cam, mesh=base_mesh, angle=90, axis=[0, 1, 0])
    plt.subplot(144)
    plt.gca().axis('off')
    plt.scatter(base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 0],
                base_mesh.vertices[base_mesh.vertices[:, 2] > scatter_depth_threshold, 1],
                c=vertex_colours_for_scatter[base_mesh.vertices[:, 2] > scatter_depth_threshold],
                s=scatter_sizes[base_mesh.vertices[:, 2] > scatter_depth_threshold])
    if contour_indices is not None:
        # Re-plotting contour vertices so they are not obscured by other vertices
        plt.scatter(base_mesh.vertices[contour_indices, 0], base_mesh.vertices[contour_indices, 1],
                    c=vertex_colours_for_scatter[contour_indices],
                    s=scatter_sizes[contour_indices])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    rend_img_top = np.concatenate([rend_img1, rend_img2], axis=1)
    rend_img_bot = np.concatenate([rend_img3, rend_img4], axis=1)
    rend_img = np.concatenate([rend_img_top, rend_img_bot], axis=0)
    return rend_img, fig


def visualise_vertex_deviations_all(deviations, resolution=(512, 512)):
    config = get_config()
    flamelayer = FLAME(config)
    shape_params = torch.zeros(1, 100)
    pose_params = torch.zeros(1, 6)
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
