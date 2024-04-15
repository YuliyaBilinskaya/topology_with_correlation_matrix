import numpy as np
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from IsingMajorana_class import IsingMajorana, local_marker, spectrum, SuperOperator
import infoLattice
from parser import parse
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import PchipInterpolator
import os
import tables as tb
import psutil
from datetime import datetime
from scipy.sparse.linalg import expm

start_time = time.time()

args = parse()  # Get parameters from parser


print(args)

# Model parameters
delta = args.delta                  # Exponent of the maximum value for the XY couplings
XX = args.hop                       # Nearest neighbour hopping on x direction
Z = args.mu                         # Onsite potential on z direction
g = args.interaction                # Interaction strength
XX2 = g                             # Next-nearest neighbour hopping on x direction
ZZ = g                              # Interaction on z direction

L = args.system_size                # Length of the chain
k_eigs = args.k_eigs                          # Number of eigenstates per each parity sector
BC = args.bc                        # Boundary conditions
system_type = args.system           # System type: open, closed

seed = args.seed                    # Seed for the random number generator
N_rep = args.N_rep                  # Number of disorder realizations for each value of delta

show = args.show                    # If True, the disorder averaged local information per scale is shown

t_max = args.t_max                  # Time for the real time evolution
dt = args.dt                        # Time step for the real time evolution

clean = args.clean                  # If True, no disorder is present in the chain. If False, disorder introduced to Z and XX

theta = args.dissipation_th         # Dissipation parameter theta for kappa0 = kappa*cos(theta)
kappa = args.dissipation_k          # Dissipation parameter kappa for kappa0 = kappa*cos(theta)

print("In this script, k_eigs is hard coded to be 1!")


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
    eigvals, psi_eigvec = eigsh(H_full, k=k_eigs, sigma=0, which='LM')

############################################################
'     -setting initial wavefunction-    '

psi_full = np.loadtxt('psi_L6_random.txt')
#psi_full = psi_eigvec

print('psi_full', psi_full)

###########################################################
    

Idt = np.eye(int(2**L)) #Identity operator of the needed dimension to build Lindbladian

if system_type == 'open':
    operator = SuperOperator(L=L, theta=theta, kappa=kappa)
    #Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full))) + operator.calc_L_super(bc=BC, theta=theta, kappa=kappa)   
    Lndbld =  operator.calc_L_super(bc=BC, theta=theta, kappa=kappa) 
elif system_type == 'closed':
    Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full))) 
else:
    raise ValueError('system type must be "open" or "closed"')


'     -Vectorizing rho-      '
# Option 1: by column
# =============================================================================
# rho = np.outer(np.conj(psi_full), psi_full)
# rho_vec = np.transpose(rho).reshape(len(psi_full)**2,1)
# =============================================================================
# Option 1: by row
rho_vec = np.kron(psi_full, np.conj(psi_full))


time_steps = t_max/dt   # number of timesteps

for c in np.arange(time_steps):
    
    t = c * dt    # current time value
    
    print('t=', t)
    
    save_dir = 'results/MBL/BC_{0}/Syst_{1}/L{2}/k_even/seed{3}/N_rep{4}/g{5}/delta{6}/t{7}'.format(*(BC,system_type, L, seed, N_rep, g, delta, t))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = 'data'
    with tb.File(filename=f'{save_dir}/{file_name}.h5', mode='a') as h5f:  # Open a new file mwe.h5 in append mode
        if not "/isingmajorana/infoperscale" in h5f:
            # Create a new table
            dtypes = [('seed', 'i8'),
                      ('ram', 'f8'),
                      ('time', 'U20')]
            if BC == 'open':
                for l in range(L):
                    dtypes.append((f'l_{l}', 'f8'))
            else:
                for l in range(L // 2 + 1):
                    dtypes.append((f'l_{l}', 'f8'))
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

    
     # Option 1 for real time evolution: calculating overlap with the eigenstates,
     # acting with corresponding exp on each eigenstate and summing over all evolved eigenstates      
    
# =============================================================================
#     psi_test_t = np.full_like(psi_even, 0)
#     for j in np.arange(k_eigs):
#         
#         psi_test_t[:,j] = np.dot(psi_even[:,j],psi_test) * np.exp(-1j*E_even[j]*t) * psi_even[:,j]
#     
#     psi_test_t = np.sum(psi_test_t, axis=1) 
#     
#     psi_test_t = 1/np.sqrt(np.sum(abs(psi_test_t)**2))*psi_test_t
#     
#     print('norm=', np.sum(abs(psi_test_t)**2))
#     
#     psi_test = psi_test_t
# =============================================================================
 
    #############################################
    # Real time evolution of vectorized density by exponentiating the Lindbladian

    rho_t = np.dot(expm(Lndbld*dt),rho_vec)
    
    print('norm=', np.trace(rho_t.reshape(2**L, 2**L)))
    
    rho_vec = rho_t    

    
    ##############################################
    
    
    
    with tb.File(filename=f'{save_dir}/{file_name}.h5', mode='a') as h5f:
        # Open the existing table to continue appending
        h5table = h5f.get_node(where="/isingmajorana", name="infoperscale", classname="Table")
    
        
        #calculating info for the Lindbladian
        info_per_scale[(delta, count)] = infoLattice.calc_info_per_scale(infoLattice.calc_info_lndbld(rho_vec), bc=BC)
        
        # Saving as hdf5
        # It keeps appending even if the same seed has been already run. Might be useful to implement a row_index
        # to overwrite the existing line with the same seed
        timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
        data_values = [s, process.memory_info().rss / 1024 ** 2, timestamp]
        data_values.extend([val for val in info_per_scale[(delta, count)].values()])
    
        # Check that the sum of the local info on the full lattice equals L
        tot = 0
        for val in info_per_scale[(delta, count)].values():
            tot += val
        print('tot_info', tot)
        data_values.extend([tot])
        
    
        h5table.append([tuple(data_values)])

    print("info_per_scale", info_per_scale)
    
    ave_info_per_scale_delta = {}
    dcount = {}
    
    for k, v in info_per_scale.items():
        for l in range(1, N_scales + 1):
            ave_info_per_scale_delta[(k[0], l)] = ave_info_per_scale_delta.get((k[0], l), 0) + v[l]
            dcount[(k[0], l)] = dcount.get((k[0], l), 0) + 1
    
    for k, v in ave_info_per_scale_delta.items():
        ave_info_per_scale_delta[k] /= dcount[k]
    
    print("ave_info_per_scale_delta", ave_info_per_scale_delta)
    
    print('Time elapsed: {:.2e} s'.format(time.time() - start_time))
    
    if show:
    # PLOTS
    
        h_fig_size = 6
        golden_ratio = (1. + np.sqrt(5)) / 2
        fig1, ax1 = plt.subplots(1, 1, figsize=(golden_ratio*h_fig_size, h_fig_size), dpi=400)   
        selected_keys = [delta]
    
        for count_k, k in enumerate(selected_keys):
    
            scales = np.arange(N_scales)
            colors = cm.get_cmap('magma', len(selected_keys) + 1)
    
            ave = np.zeros(N_scales)
            for l in range(1, N_scales + 1):
                ave[l - 1] = ave_info_per_scale_delta.get((k, l), 0)
    
            # print("ave", ave)
    
            interp_func = PchipInterpolator(scales, ave)
            interp_scales = np.arange(scales[0], scales[-1] + 0.1, 0.1)
            interp_average = interp_func(interp_scales)
    
            ax1.plot(interp_scales, interp_average, color=colors(count_k), label=k)
            ax1.plot(scales, ave, marker='x', linestyle='None', color='red')
    
            # for count_datum, datum in enumerate(dataPBC[key]):
            # plt.plot(scales, datum, color=colors(count_datum))
    
        ax1.set_title("Ising-Majorana; " + BC +  " BC; " + system_type + " syst (full Lndbl); L=%d; delta=%d; J=%d; h=%d; g=%.1f; dis.real.=%d; t=%.2f" % (L, delta, XX, Z, g, N_rep, t))
        ax1.set_ylabel(r"$i^\ell$")
        ax1.set_ylim([0, L])
        ax1.set_xlabel(r"$\ell$")
        ax1.legend()
        plt.savefig(save_dir + ".pdf", dpi=200)
    
        plt.show()
    
    #plotting the circle diagram    
    #infoLattice.plot_info_latt(infoLattice.calc_info_lndbld(rho_vec))
    



