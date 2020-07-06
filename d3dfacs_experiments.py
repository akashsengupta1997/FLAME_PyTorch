from vis.render_sequence import render_sequence, render_all_sequences_for_actor
from metrics.vertex_variance import compute_vertex_deviations_over_sequence, compute_vertex_deviations_for_actor, \
    compute_vertex_deviations_for_all

d3dfacs_path = "/Users/Akash_Sengupta/Documents/Datasets/d3dfacs_alignments"
compute_vertex_deviations_for_all(d3dfacs_path)
