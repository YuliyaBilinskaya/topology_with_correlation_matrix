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
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

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
bath = args.bath                    # Type of bath coupling
system_type = args.system           # System type: open, closed

seed = args.seed                    # Seed for the random number generator
N_rep = args.N_rep                  # Number of disorder realizations for each value of delta

show = args.show                    # If True, the disorder averaged local information per scale is shown

t_max = args.t_max                  # Time for the real time evolution
dt = args.dt                        # Time step for the real time evolution

clean = args.clean                  # If True, no disorder is present in the chain. If False, disorder introduced to Z and XX

theta = args.dissipation_th         # Dissipation parameter theta for kappa0 = kappa*cos(theta) for a TBD bath type
kappa = args.dissipation_k          # Dissipation parameter strength of the bath coupling to the system

steady_state = args.ss              # Set to True if you want to calculate the Steady State as well, which will take more resources

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
    
    #eigvals, psi_eigvec = eigsh(H_full, k=k_eigs, sigma=0, which='LM')
    eigvals, psi_eigvec = eigh(H_full)

############################################################
'     -setting initial wavefunction-    '
print('eigvals', eigvals)
# Index for the the ground state = 0
idx = int(0)
psi_full = psi_eigvec[:,idx]

#psi_full = np.loadtxt('psi_L6_triv_delta-4.txt')

#print('psi_full', psi_full)
###########################################################
    

Idt = np.eye(int(2**L)) #Identity operator of the needed dimension to build Lindbladian

if system_type == 'open':
    operator = SuperOperator(L=L, theta=theta, kappa=kappa, bath=bath)
    Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full)))  + operator.calc_L_super(bc=BC, theta=theta, kappa=kappa, bath=bath)   
    #Lndbld =  operator.calc_L_super(bc=BC, theta=theta, kappa=kappa, bath=bath) 
elif system_type == 'closed':
    Lndbld = -1j * (np.kron(H_full, Idt) - np.kron(Idt, np.conj(H_full))) 
else:
    raise ValueError('system type must be "open" or "closed"')

#print('diagonal H_full=', np.diag(H_full))
#print('H_full:')
#print(H_full)
#print('diagonal np.kron(H_full, Idt)', np.diag(np.kron(H_full, Idt)))
#print('diagonal np.kron(Idt, np.conj(H_full))', np.diag(np.kron(Idt, np.conj(H_full))))
#print('diagonal unitary part', np.diag(np.kron(H_full, Idt)) - np.diag(np.kron(Idt, np.conj(H_full))))
#print('diagonal Lndbld=', np.diag(Lndbld))
#print('Lndbld:')
#print(Lndbld)


# non-unitary operator for the time evolution with Lindblad    
non_unitary = expm(Lndbld*dt)



'     -Vectorizing rho-      '
# Option 1: by column
# =============================================================================
#rho = np.outer(np.conj(psi_full), psi_full)
#rho_vec = np.transpose(rho).reshape(len(psi_full)**2,1)
# =============================================================================
# Option 2: by row
rho_vec = np.kron(psi_full, np.conj(psi_full))
#rho_vec = np.loadtxt('Rho_Prod+Mixed_L6')

if steady_state == 'True':
    '     -Extracting the steady state-      '
    eigvalues, eigvectors = np.linalg.eig(Lndbld)
    min_idx = min(range(len(eigvalues)), key=abs(eigvalues).__getitem__)
    print('Lndbld eigvalue', eigvalues[min_idx])
    
    # Normalizing by dividing the whole matrix by its trace
    rho_vec_steady = eigvectors[:,min_idx] * 1/np.trace(eigvectors[:,min_idx].reshape(2**L, 2**L)) 
    
    print('norm steady state = ', np.trace(rho_vec_steady.reshape(2**L, 2**L)))
    
    #Calculate steady-state
    info_per_scale_steady = infoLattice.calc_info_per_scale(infoLattice.calc_info_lndbld(rho_vec_steady), bc=BC)
    ave_steady = [info_per_scale_steady[x] for x in np.arange(L)+1]
    
    interp_func_steady = PchipInterpolator(np.arange(L), ave_steady)
    interp_scales_steady = np.arange(np.arange(L)[0], np.arange(L)[-1] + 0.1, 0.1)
    interp_average_steady= interp_func_steady(interp_scales_steady)
    marker_std = '3'



print('Time elapsed: {:.2e} s'.format(time.time() - start_time))



time_steps = t_max/dt   # number of timesteps

info_latt_t = np.zeros((int(time_steps), L))

tot_info_t = np.zeros(int(time_steps))

for c in np.arange(time_steps):
    
    t = c * dt    # current time value
    
    print('t=', t)
    
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

    if t == 0:
        rho_t = rho_vec
        print('norm=', np.trace(rho_t.reshape(2**L, 2**L)))
        
    else:
        rho_t = np.dot(non_unitary,rho_vec)    
        print('norm=', np.trace(rho_t.reshape(2**L, 2**L)))   
        rho_vec = rho_t    

    
    ##############################################
    
    
    
    with tb.File(filename='{}/{}.h5'.format(save_dir, file_name), mode='a') as h5f:
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
        tot_info = tot
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

    
# PLOTTING
    

    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(golden_ratio*h_fig_size, h_fig_size), dpi=400)
    
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=400)

    
    
    #fig1.tight_layout()
    
#    if system_type == 'open' and clean == 'False':
#        fig1.suptitle("Ising-Majorana; clean: " + clean + "; "  + BC +  " BC; " + system_type + " sys;  " + bath + " bath; \n L=%d; delta=%d; J=%d; h=%d; g=%.1f; \n t=%.2f" % (L, delta, XX, Z, g, t))
        
#    elif system_type == 'open' and clean == 'True':
#        fig1.suptitle("Ising-Majorana; clean: " + clean + "; "  + BC +  " BC; " + system_type + " sys;  " + bath + " bath; \n L=%d; J=%d; h=%d; g=%.1f; \n t=%.2f" % (L, XX, Z, g, t))
        
#    elif system_type == 'closed' and clean == 'False':
#        fig1.suptitle("Ising-Majorana; clean: " + clean + "; "  + BC +  " BC; " + system_type + " sys; \n L=%d; delta=%d; J=%d; h=%d; g=%.1f; \n t=%.2f" % (L, delta, XX, Z, g, t))
        
#    elif system_type == 'closed' and clean == 'True':
#        fig1.suptitle("Ising-Majorana; clean: " + clean + "; "  + BC +  " BC; " + system_type + " sys; \n L=%d; J=%d; h=%d; g=%.1f; \n t=%.2f" % (L, XX, Z, g, t))

    
    if show:
    # PLOTS
    
        selected_keys = [delta]
    
        for count_k, k in enumerate(selected_keys):
    
            scales = np.arange(N_scales)
            colors = cm.get_cmap('magma', len(selected_keys) + 1)
    
            ave = np.zeros(N_scales)
            for l in range(1, N_scales + 1):
                ave[l - 1] = ave_info_per_scale_delta.get((k, l), 0)
    
            # print("ave", ave)
            
            # saving the initial state
            if t == 0:
                scales_t0 = scales
                ave_t0 = ave
                interp_func_t0 = PchipInterpolator(scales_t0, ave_t0)
                interp_scales_t0 = np.arange(scales_t0[0], scales_t0[-1] + 0.1, 0.1)
                interp_average_t0 = interp_func_t0(interp_scales_t0)
                marker_ini = 'x'
                
    
            interp_func = PchipInterpolator(scales, ave)
            interp_scales = np.arange(scales[0], scales[-1] + 0.1, 0.1)
            interp_average = interp_func(interp_scales)
            marker_fin = '4'
            
            ax1.plot(interp_scales_t0, interp_average_t0, color='black', label='Initial state')
            ax1.plot(scales_t0, ave_t0, marker = marker_ini, linestyle='None', markersize=8, color='deepskyblue', label='Initial state')
    
            ax1.plot(interp_scales, interp_average, color='black', label='Final state')
            ax1.plot(scales, ave, marker = marker_fin, linestyle='None', markersize=10, color='blue', label='Final state')
    
            # for count_datum, datum in enumerate(dataPBC[key]):
            # plt.plot(scales, datum, color=colors(count_datum))
            
            if steady_state == 'True':
    
                ax1.plot(interp_scales_steady, interp_average_steady, color='black', label='Steady state')
                ax1.plot(scales, ave_steady, marker = marker_std, linestyle='None', markersize=10, color='red', label='Steady state')

    
        ax1.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
        ax1.set_ylim([-0.1, L+0.1])
        ax1.set_xlabel(r"$\ell$", fontsize=16)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)

        #ax1.set_aspect('equal', adjustable='box')
        
        if steady_state == 'True':
            legend_elements = [
                Line2D([0], [0], color='black', marker=marker_ini, markeredgecolor = 'deepskyblue', markersize=8, linestyle='-', label='Initial state'),
                Line2D([0], [0], color='black', marker=marker_fin, markeredgecolor = 'blue', markersize=10, linestyle='-', label='Final state'),
                Line2D([0], [0], color='black', marker=marker_std, markeredgecolor = 'red', markersize=10, linestyle='-', label='Steady state')
                ]
        else:
            legend_elements = [
                Line2D([0], [0], color='black', marker=marker_ini, markeredgecolor = 'deepskyblue', markersize=8, linestyle='-', label='Initial state'),
                Line2D([0], [0], color='black', marker=marker_fin, markeredgecolor = 'blue', markersize=10, linestyle='-', label='Final state'),
                ]
        
        ax1.legend(handles=legend_elements, fontsize=14)
        ax1.text(0.07, 0.97, '({})'.format(chr(97)), transform=ax1.transAxes, fontsize=14, verticalalignment='top', weight='bold')
        
    
    
        #plotting the circle diagram    
        #ax2.plot(infoLattice.plot_info_latt(infoLattice.calc_info_lndbld(rho_vec), save_dir, t, ax2))
        
        color_map = plt.get_cmap("Oranges")
        info_latt = infoLattice.calc_info_lndbld(rho_vec)
        
        print('info_latt', info_latt)
        r = 1/(4*L)
        #fig, ax = plt.subplots(dpi=300)
        for l in info_latt:
            for x in range(len(info_latt[l])):
                ax2.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, color=color_map(info_latt[l][x])))
        ax2.set_xlim([-2*r,1])
        #YB:
        
        x_ticks = []
        y_ticks = []
        for x in range(len(info_latt[1])):
            print("x",x)
            x_ticks.append(x/L+1/(2*L))
            y_ticks.append(x/L+1/(2*L))
    
        
        ax2.set_xticks(x_ticks, range(1,L+1))
        ax2.set_yticks(y_ticks, range(0,L))
        
        ax2.set_xlabel('Site', fontsize=16)
        ax2.set_ylabel('$\ell$', fontsize=16)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        
        ax2.set_ylim([-2*r,1+2*r])
        ax2.text(0.07, 0.97, '({})'.format(chr(97+1)), transform=ax2.transAxes, fontsize=14, verticalalignment='top', weight='bold')
        
        info_latt_t[int(c),:]= ave
        
        x_linspace = np.arange(0, t_max, dt)
        
        tot_info_t[int(c)] = tot_info
                
        markermap = ['s', 'o', 'd', 'p', 'v', '>', '^', '<', 'v', '^']
        #markermap = ['o', '^']
    
        for w in [1, int(L)-1]:
            ax3.plot(x_linspace, info_latt_t[:,w], marker=markermap[w], markerfacecolor='None', color = 'black', linewidth = 1, label = (r"$\mathscr{i}^%d$" % (w)))

        ax3.plot(x_linspace, tot_info_t, linestyle = '--', color='green', label = ("$I_{tot}$"))

        ax3.legend(fontsize=14)
        
        ax3.set_xlabel('t', fontsize=16)
        ax3.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18, labelpad=1)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.tick_params(axis='y', labelsize=14)
        ax3.text(0.07, 0.97, '({})'.format(chr(97+2)), transform=ax3.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    
        
        plt.savefig(save_dir + ".pdf", dpi=200, bbox_inches='tight')
        
        plt.show()
        
if steady_state == 'True':        
    #plotting steady state separately
    
    save_gen_dir = 'results/clean_{0}/BC_{1}/Syst_{2}/bath_{3}/L{4}/J{5}/mu{6}/seed{7}/kappa{8}/g{9}/delta{10}/'.format(*(clean, BC, system_type, bath, L, XX, Z, seed, kappa, g, delta))
    
            
    plt.plot(interp_scales_steady, interp_average_steady, color='black', label='Steady state')
    plt.plot(np.arange(L), ave_steady, marker = marker_std, linestyle='None', markersize=10, color='red', label='Steady state')
    plt.ylim([-0.1, L])
    plt.ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    
    legend_elements = [
        Line2D([0], [0], color='black', marker=marker_std, markeredgecolor = 'red', markersize=10, linestyle='-', label='Steady state')
        ]
    
    plt.legend(handles=legend_elements, fontsize=14)
    
    plt.savefig(save_gen_dir + "SteadyState.pdf", dpi=400, bbox_inches='tight')
    
    plt.show()
    
    np.savetxt(save_gen_dir + "Steady_state_Info_per_scale", ave_steady)
    
#%%

# =============================================================================
# x_linspace = np.arange(0, t_max, dt)[0:int(c)+1]
#         
#         # specify subsystems that are interesting to track
#         level1 = 0   # the level of subsystem 1
#         site1 = 0    # what site on that level is interesting
#         info_latt_t1.append(info_latt[level1+1][site1])
#         n1 = (level1+1)/2 + 0.5 + site1     #calculates the central value of the subsystem 1
#         
#         level2 = 0
#         site2 = 1
#         info_latt_t2.append(info_latt[level2+1][site2])
#         n2 = (level2+1)/2 + 0.5 + site2
#         
#         level3 = 0
#         site3 = 3
#         info_latt_t3.append(info_latt[level3+1][site3])
#         n3 = (level3+1)/2 + 0.5 + site3
#         
#         level4 = 0
#         site4 = 5
#         info_latt_t4.append(info_latt[level4+1][site4])
#         n4 = (level4+1)/2 + 0.5 + site4
#         
#         x_linspace = np.arange(0, t_max, dt)[0:int(c)+1]
#         
#         tot_info_t.append(tot_info/L)
#         
#     #    ax3.plot(np.round(np.linspace(0,t, int(t/dt)+1),2),tot_info, marker = 'o', markersize = 5, color='grey', label = ("$\mathscr{I}$_{tot}"))
#         ax3.plot(x_linspace, info_latt_t1, color='blue', linewidth = 3, label = ("$\mathscr{i}^%d_{%.1f}$" % (level1, n1)))
#         ax3.plot(x_linspace, info_latt_t2, color='orange', linewidth = 3, label = ("$\mathscr{i}^%d_{%.1f}$" % (level2, n2 )))
#         ax3.plot(x_linspace, info_latt_t3, color='green', linewidth = 3, label = ("$\mathscr{i}^%d_{%.1f}$" % (level3, n3 )))
#         ax3.plot(x_linspace, info_latt_t4, color='purple', linewidth = 3, label = ("$\mathscr{i}^%d_{%.1f}$" % (level4, n4 )))
#         ax3.plot(x_linspace, tot_info_t, color='grey', linewidth = 3, label = ("$I_{tot}/L$"))
# =============================================================================
