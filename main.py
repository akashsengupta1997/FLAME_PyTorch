"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
import os
import numpy as np
import torch
from FLAME import FLAME
import pyrender
import trimesh
from renderer.weak_perspective_pyrender_renderer import Renderer
from config import get_config
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt

gpu = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.zeros(8, 100).to(device)
shape_params[:, 0] = -2

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
# pose_params_numpy = np.array([[0.0, 30.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, -30.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, 85.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, -48.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, 10.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, -15.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, 0.0*radian, 0.0, 0.0, 0.0, 0.0],
#                                 [0.0, -0.0*radian, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
# pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(device)
pose_params = torch.zeros(8, 6).to(device)
# pose_params[:, 2] = np.pi

# Cerating a batch of neutral expressions
expression_params = torch.zeros(8, 50, dtype=torch.float32).to(device)
flamelayer.to(device)

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project
print(vertice.size(), landmark.size())

if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(8, 3).to(device)
    eye_pose = torch.zeros(8, 6).to(device)
    vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

faces = flamelayer.faces
renderer = Renderer(faces, resolution=(512,512))
cam = np.array([4.0, 0., 0.])
for i in range(1):
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    # tri_mesh = trimesh.Trimesh(vertices, faces,
    #                             vertex_colors=vertex_colors)
    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    rend_img = renderer.render(vertices, cam, angle=-20, axis=[0., 1., 0.])

    plt.figure()
    plt.subplot(121)
    plt.scatter(vertices[:, 0], vertices[:, 1], s=1.0, c=vertices[:, 2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplot(122)
    plt.imshow(rend_img)
    plt.show()

np.save('model/flame_faces.npy', faces)
