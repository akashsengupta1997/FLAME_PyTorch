import numpy as np
from FLAME import FLAME
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
import torch

from config import get_config


# config = get_config()
# flamelayer = FLAME(config)
# shape_params = torch.zeros(1, 100)  # Using average FLAME face as "mask model"
# # shape_params[:, 0] = -5
# pose_params = torch.zeros(1, 6)
# expression_params = torch.zeros(1, 50, dtype=torch.float32)
# neck_pose = torch.zeros(1, 3)
# eye_pose = torch.zeros(1, 6)
# vertices, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
# vertices = vertices[0].cpu().detach().numpy()
# pose_params[:, 1] = np.pi/3
# rot_vertices, _ = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
# rot_vertices = rot_vertices[0].cpu().detach().numpy()
# pose_params[:, 1] = -np.pi/3
# rot_vertices2, _ = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
# rot_vertices2 = rot_vertices2[0].cpu().detach().numpy()

import trimesh
import math
actor_mesh = trimesh.load("/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Joe/1+2/1+2_175.ply",
                          file_type='ply', process=False)
R = trimesh.transformations.rotation_matrix(math.radians(-30), [1, 0, 0])
actor_mesh.apply_transform(R)
vertices = np.copy(actor_mesh.vertices)
R = trimesh.transformations.rotation_matrix(math.radians(-60), [0, 1, 0])
actor_mesh.apply_transform(R)
rot_vertices = np.copy(actor_mesh.vertices)
R = trimesh.transformations.rotation_matrix(math.radians(120), [0, 1, 0])
actor_mesh.apply_transform(R)
rot_vertices2 = np.copy(actor_mesh.vertices)

indices = np.arange(vertices.shape[0])

threshold = -0.01  # threshold by depth to discard back-of-head vertices
thresholded_vertices = vertices[vertices[:, 2] > threshold, :]
thresholded_indices = indices[vertices[:, 2] > threshold]
threshold_rot_vertices = rot_vertices[vertices[:, 2] > threshold, :]
threshold_rot_vertices2 = rot_vertices2[vertices[:, 2] > threshold, :]

c = np.array([[0., 0., 1.]] * threshold_rot_vertices.shape[0])
s = np.ones(thresholded_vertices.shape[0]) * 3.

print(vertices.shape, indices.shape)
print(thresholded_vertices.shape, thresholded_indices.shape)
print(c.shape, s.shape)

mask_indices = [2094, 2098, 2097, 2100, 1575, 3727, 3726, 3725, 3588, 3587, 3643, 3636, 3635, 3634, 3630,
                3414, 3413, 3415, 3416, 3417, 3419, 3389, 3390, 3470, 3471, 3472, 2711, 3127, 3124, 3125,
                3121, 3122, 3105, 3104, 2761, 3602, 2973, 3561, 1895, 3814, 1644, 2069, 2070, 2095, 2094]
for index in mask_indices:
    c[thresholded_indices == index, :] = [0., 0.5, 0.]
    s[thresholded_indices == index] = 30.

fig = plt.figure(figsize=(18, 10))
plt.subplot(131)
ax = plt.gca()
sc = plt.scatter(thresholded_vertices[:, 0], thresholded_vertices[:, 1], s=s, c=c)
plt.gca().set_aspect('equal', adjustable='box')
plt.subplot(132)
plt.scatter(threshold_rot_vertices[:, 0], threshold_rot_vertices[:, 1], s=s, c=c)
plt.gca().set_aspect('equal', adjustable='box')
plt.subplot(133)
plt.scatter(threshold_rot_vertices2[:, 0], threshold_rot_vertices2[:, 1], s=s, c=c)
plt.gca().set_aspect('equal', adjustable='box')

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = str(thresholded_indices[ind['ind'][0]])
    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

# fig.canvas.mpl_connect("motion_notify_event", hover)  # Uncomment to see vertex index when hovering over vertex


# Visualise the mask
from mpl_toolkits.mplot3d import Axes3D
mask_vertices = vertices[mask_indices, :]
rot_mask_vertices = rot_vertices[mask_indices, :]
rot_mask_vertices2 = rot_vertices2[mask_indices, :]
plt.figure(figsize=(12, 8))
plt.subplot(111, projection='3d')
plt.plot(xs=mask_vertices[:, 0]*1000, ys=mask_vertices[:, 1]*1000, zs=mask_vertices[:, 2]*1000, c=[0., 0.5, 0.])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.gca().set_zlabel('z (mm)')
plt.show()




