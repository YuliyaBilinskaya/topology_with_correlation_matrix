import numpy as np
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from IsingMajorana_class import IsingMajorana, local_marker, spectrum, SuperOperator, SS_degen
import infoLattice
import functions
from parser import parse
import os
import tables as tb
import psutil
from datetime import datetime
from scipy.sparse.linalg import expm
from infoLattice import plot_info_lattice, plot_info_latt
from functions import calc_ini_C_mk_A_mk_t0, init_corr_from_blocks, build_generalized_correlation_matrix
from corr_integrators import make_stepper
from infoLattice import plot_info_latt_from_corr, plot_info_lattice_from_corr
from local_marker_func import plot_local_marker, plot_local_marker_comparison

#%%
start_time = time.time()

args = parse()  # Get parameters from parser

# Model parameters
delta = args.delta                      # Exponent of the maximum value for the XY couplings
J = args.hop                           # Nearest neighbour hopping on x direction
h = args.mu                             # Onsite potential on z direction
gamma = args.dissipation_k              # Dissipation parameter strength of the bath coupling to the system

L = args.system_size                    # Length of the chain
BC = args.bc                            # Boundary conditions
bath = args.bath                        # Type of bath coupling
system_type = args.system               # System type: open, closed

seed = args.seed                        # Seed for the random number generator
N_rep = args.N_rep                      # Number of disorder realizations for each value of delta

show = args.show                        # If True, the disorder averaged local information per scale is shown

t_max = args.t_max                      # Time for the real time evolution
dt = args.dt                            # Time step for the real time evolution
integrator = args.integrator            # 'euler' or 'rk45'
clean = args.clean                      # If True, no disorder is present in the chain. If False, disorder introduced to Z and XX

theta = args.dissipation_th             # Dissipation parameter theta for kappa0 = kappa*cos(theta) for a TBD bath type
kappa = args.dissipation_k              # Dissipation parameter strength of the bath coupling to the system

steady_state = args.ss                  # Set to True if you want to calculate the Steady State as well, which will take more resources

parity = args.block_diag                # Set to True if you want to Block-diagonalize the Lindbladian, which will take more resources

initial_state_type = args.ini_state     # Initial state
epsilon = args.epsilon


# Definitions
if BC == 'periodic':
    N_scales = L // 2 + 1
elif BC == 'open':
    N_scales = L
else:
    raise ValueError('boundary condition must be "periodic" or "open"')

print("In this script, k_eigs is hard coded to be 1!")

info_per_scale = {}


############################################################
'     -making initial state-    '
#Build full rho, applicable for small system sizes

#rho_vec = functions.make_initial_state(H_full=None, Lndbld=None, L=L, initial_state_type=initial_state_type, epsilon = epsilon)
#rho = rho_vec.reshape(2**L, 2**L)
#C_mk_ini, A_mk_dagger_ini = calc_ini_C_mk_A_mk_t0(rho)

#Use to avoid building full rho for large system sizes

C_mk_ini, A_mk_dagger_ini = init_corr_from_blocks(L=L, initial_state_type=initial_state_type, epsilon=epsilon)

#print('C_mk_ini', C_mk_ini)
#print('A_mk_dagger_ini', A_mk_dagger_ini)

###########################################################

#%%

time_steps = int(round(t_max / dt))      # number of timesteps
assert abs(t_max / dt - time_steps) < 1e-9, "t_max must be an integer multiple of dt"

save_stride = 10  # save data only for every 'th step
output_file = f"opdm_t_corr_diss={gamma:g}_J={J:g}_L={L}_init={initial_state_type}_dt={dt:g}.h5"
filters = tb.Filters(complevel=5, complib="blosc:zstd")

n_saved = time_steps // save_stride + 1


info_latt_t = np.zeros((n_saved, L), dtype=np.float32)
tot_info_t = np.zeros(n_saved, dtype=np.float32)
times_saved = np.zeros(n_saved, dtype=np.float64)

step = make_stepper(integrator, gamma, J, h)

C_mk_t = C_mk_ini
A_mk_dagger_t = A_mk_dagger_ini

with tb.open_file(output_file, mode="w") as h5:
    opdm_store = h5.create_earray(
        h5.root,
        "opdm_t_corr",
        atom=tb.ComplexAtom(itemsize=8),
        shape=(0, 2 * L, 2 * L),
        filters=filters,
        expectedrows=n_saved,
    )

    save_idx = 0

    for c in range(time_steps + 1):
        t = c * dt

        if c == 0:
            C_mk_t = C_mk_ini
            A_mk_dagger_t = A_mk_dagger_ini
        else:
            C_mk_t, A_mk_dagger_t = step(C_mk_t, A_mk_dagger_t, dt)


        if c % save_stride == 0:
            generalized_corr_matrix = build_generalized_correlation_matrix(C_mk_t, A_mk_dagger_t)
            info_latt = infoLattice.calc_info_from_corr_matr(generalized_corr_matrix)
            current_info_per_scale = infoLattice.calc_info_per_scale(info_latt, bc=BC)
            info_per_scale[(delta, 0)] = current_info_per_scale
            tot_info = sum(current_info_per_scale.values())
            opdm_store.append(generalized_corr_matrix.astype(np.complex64)[None, :, :])
            times_saved[save_idx] = t
            info_latt_t[save_idx, :len(current_info_per_scale)] = np.array(
                list(current_info_per_scale.values()), dtype=np.float32
            )
            tot_info_t[save_idx] = np.float32(tot_info)
            save_idx += 1

    h5.create_array(h5.root, "times_corr", times_saved[:save_idx])
    h5.create_array(h5.root, "tot_info_t", tot_info_t[:save_idx])
    h5.create_array(h5.root, "info_latt_t", info_latt_t[:save_idx])

    if show:
        plot_local_marker(opdm_store[:], times_saved[:save_idx])

        #if t == dt * np.arange(time_steps)[-1]:
            #plot_info_latt_from_corr(generalized_corr_matrix, L, InfoLatt_norm=2, time=t)
            #plot_info_lattice_from_corr(delta, info_per_scale, N_scales, L, generalized_corr_matrix, info_latt_t, tot_info_t, tot_info, t, c, dt, t_max, InfoLatt_norm=1)

            #np.save("opdm_t_corr.npy", generalized_corr_t)
            #np.save("times_corr.npy", times)
#
            #opdm_t_lindblad = np.load("opdm_t_lindblad.npy")
            #times_lindblad = np.load("times_lindblad.npy")
#
            #opdm_t_corr = np.load("opdm_t_corr.npy")
            #times_corr = np.load("times_corr.npy")
#
            #plot_local_marker_comparison(opdm_t_lindblad, times_lindblad, opdm_t_corr, times_corr,
            #                             lindblad_label='Lindblad evolution', corr_label='Correlation matrix evolution',
            #                             save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
            #                             use_lines=True, truncate_to_common_times=False)

        


    
