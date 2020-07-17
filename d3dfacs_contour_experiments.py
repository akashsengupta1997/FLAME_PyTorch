from FLAME import FLAME
import torch

from config import get_config
from metrics.contour_vertex_deviation import compute_contour_vertex_deviations_for_actor_sequence, \
    compute_contour_vertex_deviations_for_actor_static

# Pre-selected contour indices
contour_indices = [2094, 2098, 2097, 2100, 1575, 3727, 3726, 3725, 3588, 3587, 3643, 3636, 3635, 3634, 3630,
                   3414, 3413, 3415, 3416, 3417, 3419, 3389, 3390, 3470, 3471, 3472, 2711, 3127, 3124, 3125,
                   3121, 3122, 3105, 3104, 2761, 3602, 2973, 3561, 1895, 3814, 1644, 2069, 2070, 2095, 2094]

config = get_config()
flamelayer = FLAME(config)
# If all shape and expression params set to 0, just using average neutral FLAME face for contour.
shape_params = torch.zeros(1, 100)  # Using 100 shape components here but can use up to 300 (set in config.py).
# shape_params[:, 0] = 4
pose_params = torch.zeros(1, 6)
expression_params = torch.zeros(1, 50, dtype=torch.float32)  # Using 50 expression params here but can use up to 100 (set in config.py)
neck_pose = torch.zeros(1, 3)
eye_pose = torch.zeros(1, 6)
vertices, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
vertices = vertices[0].cpu().detach().numpy()
base_contour_vertices = vertices[contour_indices, :]

# seq_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Darren/1+2+4+5+11+17+38"
# compute_contour_vertex_deviations_over_sequence(seq_path, base_contour_vertices, contour_indices,
#                                                 apply_similarity_transform=True)

# actor_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Joe"
# compute_contour_vertex_deviations_for_actor_sequence(actor_path, base_contour_vertices, contour_indices,
#                                                      apply_similarity_transform=True, save_deviation_image=True,
#                                                      reduce='mean')
actor_mesh_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Michaela/1+2/1+2_150.ply"
compute_contour_vertex_deviations_for_actor_static(actor_mesh_path, base_contour_vertices, contour_indices,
                                                   apply_similarity_transform=True, save_deviation_image=True)
# actor_mesh_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Michaela/1+2/1+2_150.ply"
# compute_contour_vertex_deviations_for_actor_static(actor_mesh_path, base_contour_vertices, contour_indices,
#                                                    apply_similarity_transform=True, save_deviation_image=True)