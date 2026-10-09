import pickle
import numpy as np

from local_marker_func import plot_local_marker_comparison, calc_opdm_from_rho

# LITE results
loaded = pickle.load(open("density_matrix.pkl", "rb"))

t_index = 0
snapshot = loaded[t_index]

max_level = max(k.level for k in snapshot.keys())
full_keys = [k for k in snapshot.keys() if k.level == max_level]

if len(full_keys) != 1:
    raise ValueError(f"Expected exactly one full-system key, found {len(full_keys)}")

many_body_rho_lite = snapshot[full_keys[0]]
opdm_lite = calc_opdm_from_rho(many_body_rho_lite)

# Wrap single snapshot into a time series of length 1
opdm_t_lite = np.array([opdm_lite])
times_lite = np.array([float(t_index)])

# Correlation matrix evolution
opdm_t_corr = np.load("opdm_t_corr.npy")
times_corr = np.load("times_corr.npy")

plot_local_marker_comparison(
    opdm_t_lite,
    times_lite,
    opdm_t_corr,
    times_corr,
    lindblad_label="Lindblad evolution",
    corr_label="Correlation matrix evolution",
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
    use_lines=True,
    truncate_to_common_times=False,
)



    
