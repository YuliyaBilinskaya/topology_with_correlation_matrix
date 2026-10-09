import numpy as np
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from IsingMajorana_class import IsingMajorana, local_marker, spectrum, SuperOperator, SS_degen
import infoLattice
import functions
from parser import parse
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import PchipInterpolator
import os
import tables as tb
import psutil
from datetime import datetime
from scipy.sparse.linalg import expm
from infoLattice import plot_info_lattice, plot_info_latt, plot_info_latt_from_corr
from local_marker_func import plot_local_marker, calc_opdm_from_rho

#%%
start_time = time.time()

args = parse()  # Get parameters from parser


print(args)

# Model parameters
delta = args.delta                      # Exponent of the maximum value for the XY couplings
XX = args.hop                           # Nearest neighbour hopping on x direction
Z = args.mu                             # Onsite potential on z direction
g = args.interaction                    # Interaction strength
XX2 = g                                 # Next-nearest neighbour hopping on x direction
ZZ = g                                  # Interaction on z direction

L = args.system_size                    # Length of the chain
k_eigs = args.k_eigs                              # Number of eigenstates per each parity sector
BC = args.bc                            # Boundary conditions
bath = args.bath                        # Type of bath coupling
system_type = args.system               # System type: open, closed

seed = args.seed                        # Seed for the random number generator
N_rep = args.N_rep                      # Number of disorder realizations for each value of delta

show = args.show                        # If True, the disorder averaged local information per scale is shown

t_max = args.t_max                      # Time for the real time evolution
dt = args.dt                            # Time step for the real time evolution

clean = args.clean                      # If True, no disorder is present in the chain. If False, disorder introduced to Z and XX

theta = args.dissipation_th             # Dissipation parameter theta for kappa0 = kappa*cos(theta) for a TBD bath type
kappa = args.dissipation_k              # Dissipation parameter strength of the bath coupling to the system

steady_state = args.ss                  # Set to True if you want to calculate the Steady State as well, which will take more resources

parity = args.block_diag                # Set to True if you want to Block-diagonalize the Lindbladian, which will take more resources

initial_state_type = args.ini_state     # Initial state



#print("In this script, k_eigs is hard coded to be 1!")


if seed == 42:
    seed = int(1000000 * np.abs(delta) + 10000000 * L + 1000000000 * 0.5 * (1 + np.sign(delta)))

# Definitions
if BC == 'periodic':
    N_scales = L // 2 + 1
elif BC == 'open':
    N_scales = L
else:
    raise ValueError('boundary condition must be "periodic" or "open"')

    
# Definitions for computing local markers (i.e., sanity check)
Id = np.eye(int(L))
sigma_x = np.array([[0, 1], [1, 0]])
S = np.kron(sigma_x, Id)  # Chiral symmetry of the chain (BdG x position space)


info_per_scale = {}

# Model Hamiltonian
chain = IsingMajorana(XX=XX, XX2=XX2, Z=Z, ZZ=ZZ, delta=delta, L=L)
print(chain)


for count, s in enumerate(range(seed, seed + N_rep)):
    print('Disorder realisation: {}/{}'.format(count + 1, N_rep))
    
    if clean == 'False':
        chain.new_disorder_realization(s)
        
    elif clean == 'True':
        chain.clean_realization()
    else:
        raise ValueError('Clean must be "True" or "False"')

    H_full = chain.calc_noparity_Hamiltonian(bc=BC)
    
energies, eigenvectors = spectrum(H_full)
print(energies)

Idt = np.eye(int(2**L)) #Identity operator of the needed dimension to build Lindbladian

if system_type == 'open':
    operator = SuperOperator(L=L, theta=theta, kappa=kappa, bath=bath)
    Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full)))  + operator.calc_L_super(bc=BC, theta=theta, kappa=kappa, bath=bath)
    #Lndbld =  operator.calc_L_super(bc=BC, theta=theta, kappa=kappa, bath=bath) #dissipative-only dynamics
elif system_type == 'closed':
    Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full))) 
else:
    raise ValueError('system type must be "open" or "closed"')


#evals, _ = np.linalg.eig(Lndbld)
#idx = np.argsort(np.abs(evals))[:30]
#print("30 smallest |evals|:", evals[idx])


# non-unitary operator for the time evolution with Lindblad    
non_unitary = expm(Lndbld*dt)

############################################################
'     -setting initial state-    '
rho_vec = functions.make_initial_state(H_full=H_full, Lndbld=Lndbld, L=L, initial_state_type=initial_state_type)

###########################################################


'     -Block-diagonalizing Lindbladian by Parity symmetry-      '
if parity == 'True':
    functions.P_parity_block_diag(Lndbld, L)

'     -Finding the steady state-      '
if steady_state == 'True':
    functions.calc_steady_state(Lndbld, L)

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))

#%%

time_steps = t_max/dt   # number of timesteps

info_latt_t = np.zeros((int(time_steps), L))

tot_info_t = np.zeros(int(time_steps))
opdm_t = np.zeros((int(time_steps), 2 * L, 2 * L), dtype=np.complex128)
times = np.arange(int(time_steps)) * dt

for c in np.arange(time_steps):
    
    t = c * dt    # current time value
    
    #print('t=', t)
    
    save_dir = 'results/clean_{0}/BC_{1}/Syst_{2}/bath_{3}/L{4}/J{5}/mu{6}/seed{7}/kappa{8}/g{9}/delta{10}/t{11}'.format(*(clean, BC, system_type, bath, L, XX, Z, seed, kappa, g, delta, t))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = 'data'
    with tb.File(filename='{}/{}.h5'.format(save_dir, file_name), mode='a') as h5f:  # Open a new file mwe.h5 in append mode
        if not "/isingmajorana/infoperscale" in h5f:
            # Create a new table
            dtypes = [('seed', 'i8'),
                      ('ram', 'f8'),
                      ('time', 'U20')]
            if BC == 'open':
                for l in range(L):
                    dtypes.append(('l_{}'.format(l), 'f8'))
            else:
                for l in range(L // 2 + 1):
                    dtypes.append(('l_{}'.format(l), 'f8'))
            dtypes.append(('tot_info', 'f8'))

            h5table = h5f.create_table(
                where="/isingmajorana",                 # Group name
                name="infoperscale",                    # Dataset name. Total path will be /some/group/trajectory
                title="Information per length scale",
                description=np.dtype(dtypes),           # The dtype of each column on the table
                createparents=True,                     # Create the intermediate groups if they do not exist yet
                chunkshape=None,                        # Give the number of rows that should be treated as a chunk
                expectedrows=N_rep,                     # Give the expected number of rows so that pytables can guess a chunkshape
                filters=tb.Filters(complevel=6,         # Compression level [0-9] (default is 6)
                                   complib="zlib",      # Compression library (zlib is the default and most portable, but others may perform better)
                                   ),
            )
            h5table.attrs.delta = delta
            h5table.attrs.XX = XX
            h5table.attrs.Z = Z
            h5table.attrs.g = g
            h5table.attrs.XX2 = XX2
            h5table.attrs.ZZ = ZZ
            h5table.attrs.L = L
            h5table.attrs.k_eigs = 1
            h5table.attrs.BC = BC
            h5table.attrs.seed = seed
            h5table.attrs.N_rep = N_rep
            h5table.attrs.t = t

    process = psutil.Process()

 
    #############################################
    # Real time evolution of vectorized density by exponentiating the Lindbladian


    if t == 0:
        rho_t = rho_vec
        #print('norm=', np.trace(rho_t.reshape(2**L, 2**L)))
    else:
        rho_t = np.dot(non_unitary,rho_vec)
        #print('norm=', np.trace(rho_t.reshape(2**L, 2**L)))
        rho_vec = rho_t
        np.save('Local_state_L6.npy', rho_vec)

    many_body_rho = rho_t.reshape(2 ** L, 2 ** L)
    opdm_t[int(c)] = calc_opdm_from_rho(many_body_rho)


    
    ##############################################
    
    
    
    with tb.File(filename='{}/{}.h5'.format(save_dir, file_name), mode='a') as h5f:
        # Open the existing table to continue appending
        h5table = h5f.get_node(where="/isingmajorana", name="infoperscale", classname="Table")
    
        
        #calculating info for the Lindbladian
        current_info_latt = infoLattice.calc_info_lndbld(rho_vec)
        current_info_per_scale = infoLattice.calc_info_per_scale(current_info_latt, bc=BC)
        info_per_scale[(delta, count)] = current_info_per_scale
        
        # Saving as hdf5
        # It keeps appending even if the same seed has been already run. Might be useful to implement a row_index
        # to overwrite the existing line with the same seed
        timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
        data_values = [s, process.memory_info().rss / 1024 ** 2, timestamp]
        data_values.extend([val for val in current_info_per_scale.values()])
    
        # Check that the sum of the local info on the full lattice equals L
        tot = 0
        for val in current_info_per_scale.values():
            tot += val
        #print('tot_info', tot)
        tot_info = tot
        data_values.extend([tot])
        info_latt_t[int(c), :len(current_info_per_scale)] = np.array(list(current_info_per_scale.values()))
        tot_info_t[int(c)] = tot_info
        
    
        h5table.append([tuple(data_values)])

    #print("info_per_scale", info_per_scale)

    # PLOTTING
    if show:
        if t == dt * np.arange(time_steps)[-1]:
            #plot_info_latt(rho_vec, L, InfoLatt_norm=2, time=t)
            plot_info_lattice(delta, info_per_scale, N_scales, L, rho_vec, info_latt_t, tot_info_t, tot_info, t,  c, dt, t_max, InfoLatt_norm=1)
            #plot_local_marker(opdm_t, times)

            np.save("opdm_t_lindblad.npy", opdm_t)
            np.save("times_lindblad.npy", times)


        


    
