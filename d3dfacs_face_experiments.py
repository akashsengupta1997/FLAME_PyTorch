import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt

import numpy as np

from metrics.full_face_vertex_deviation import compute_vertex_deviations_over_sequence, compute_vertex_deviations_for_actor, \
    compute_vertex_deviations_for_all

# d3dfacs_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments"
# compute_vertex_deviations_for_all(d3dfacs_path)

actor_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments/Joe"
deviations_array = compute_vertex_deviations_for_actor(actor_path, save_deviation_image=True, reduce='mean',
                                                       normalise_render=False, rigid_transform=True)
# mean_deviations = np.mean(deviations_array, axis=0)
# print(mean_deviations.shape)
# plt.figure()
# plt.hist(mean_deviations)
# plt.show()
