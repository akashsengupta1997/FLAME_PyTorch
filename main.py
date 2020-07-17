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
shape_params = torch.zeros(1, 100).to(device)  # Using 100 here but can use up to 300 (set in config)
# shape_params[:, 0] = -5

# Creating a batch of different global poses
pose_params = torch.zeros(1, 6).to(device)  # First 3 components are global rotation of head, next 3 params is jaw rotation
# pose_params[:, 5] = np.pi/4

# Cerating a batch of neutral expressions
expression_params = torch.zeros(1, 50, dtype=torch.float32).to(device)  # Using 50 here but can use up to 100 (set in config)
flamelayer.to(device)

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(1, 3).to(device)
    # neck_pose[:, 1] = np.pi/4
    eye_pose = torch.zeros(1, 6).to(device)
    # eye_pose[:, 1] = np.pi
    vertices, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
else:
    vertices, landmark = flamelayer(shape_params, expression_params, pose_params)  # For RingNet project
print(vertices.size(), landmark.size())

faces = flamelayer.faces
print(faces.shape)
renderer = Renderer(faces, resolution=(512, 512))
cam = np.array([4.0, 0., 0.])
for i in range(1):
    vertices = vertices[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    rend_img = renderer.render(vertices, cam, angle=0, axis=[0., 1., 0.])

    plt.figure()
    plt.subplot(121)
    plt.scatter(vertices[:, 0], vertices[:, 1], s=1.0, c=vertices[:, 2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplot(122)
    plt.imshow(rend_img)
    plt.show()
