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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import scipy.optimize as spo
from sympy import var, Matrix, SparseMatrix, nsolve
from numpy import linalg
import sympy as sp
from sympy import lambdify
from sympy import *
from sympy import eye, ImmutableSparseMatrix
from mpmath import mp
from matplotlib.lines import Line2D

#%%
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
#print('eigvals', eigvals)
# Index for the the ground state = 0
#idx = int(2**L/2+6)
idx = int(0)
psi_full = psi_eigvec[:,idx]

#psi_full = psi_eigvec[:,idx+1] + psi_eigvec[:,idx+4] + psi_eigvec[:,idx+7]
#psi_full = psi_full/np.sqrt(sum(i**2 for i in psi_full))
#print('norm psi' , sum(i**2 for i in psi_full))

#psi_full = np.loadtxt('psi_L5_triv.txt')

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



'     -Finding the steady state-      '
if steady_state == 'True':

    eigvalues, eigvectors = np.linalg.eig(Lndbld)
    
    if np.sum((np.real(eigvalues) < 1e12) & (np.real(eigvalues) > -1e-12)) == 1:         #if there is only 1 zero egenvalue
        # For non-degenerate steady state:
        zero_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-1])
        print('Lndbld steady state eigvalue:', eigvalues[zero_idx])
        #print('Lndbld eigvalues', sorted(np.real(eigvalues)))
        
        # Normalizing by dividing the whole matrix by its trace
        rho_vec_steady = eigvectors[:,zero_idx] * 1/np.trace(eigvectors[:,zero_idx].reshape(2**L, 2**L)) 
    
    elif np.sum((np.real(eigvalues) < 1e12) & (np.real(eigvalues) > -1e-12)) == 2:
        # For double-degenerate steady state
        
        # the two zero eigenvalues should be the largest ones as Lindbladian has all eigenvalues non-positive       
        zero1_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-1])
        zero2_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-2])
        
        print('Lndbld steady state eigvalues:', eigvalues[zero1_idx], eigvalues[zero2_idx])
        

        #extracting the two degenerate steady states
        rho_degen1 = eigvectors[:,zero1_idx].reshape(2**L, 2**L)
        rho_degen2 = eigvectors[:,zero2_idx].reshape(2**L, 2**L)

#################### 1st method (arxiv.org/pdf/2101.10236) ###########################          

        #making the density matrices hermitian
        rho_degen1_ho = (rho_degen1 + np.conj(np.transpose(rho_degen1))) 
        rho_degen2_h = (rho_degen2 + np.conj(np.transpose(rho_degen2))) 

        #making the second density matrix orthogonal to the first one        
        rho_degen2_ho = rho_degen2_h - np.trace(rho_degen1_ho * rho_degen2_h)/np.trace(rho_degen1_ho * rho_degen1_ho) * rho_degen1_ho 


        #stacking the two density matrices
        rho_degen_o = np.concatenate((rho_degen1_ho, rho_degen2_ho))
        
        rho_degen_o[(abs(rho_degen_o) > -1e-12) & (abs(rho_degen_o) < 1e-12)] = 0   # setting all negligeble values to zero for faster calculations
        
        print('Calculated rho_steady_o')
        
        var('theta')
        
        # making the rotation matrix with (M^2-M)/2 Euler angles, where M is # of degeneracies
        # for 2 degenraces only 1 euler angle needed, \theta
        U_euler = Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
    
        # acting with the unitary such that the density matrices now depend on theta   
        rho_ksi_o = Matrix(np.dot(np.kron(U_euler, np.eye(int(2**L))), rho_degen_o))
        
        print('Calculated rho_ksi_o')

        # separating the two density matrices
        rho_ksi1 = rho_ksi_o[0:2**L,0:2**L]
        rho_ksi2 = rho_ksi_o[2**L:2*2**L,0:2**L]

        # getting the eigenvalues of each density matrix as a function of the angle \theta
        eigvals1 = list(ImmutableSparseMatrix(rho_ksi1).eigenvals().keys())
        
        print('Calculated eigvals1')
        
        eigvals2 = list(ImmutableSparseMatrix(rho_ksi2).eigenvals().keys())
        
        print('Calculated eigvals2')
            

        all_eigvals = eigvals1 + eigvals2
        start_theta = 1.184186
        
        y = []

        # setting up minimization expression: sum_i ( \lambda(theta)_i - abs(\lambda(theta)_i) ) = 0
        # which means that sum of all eigenvalues minus their absolute values should be equal to zero as a function of unknown theta
        
        for i in all_eigvals:               
            y.append(i - abs(i)) 
            
        fun = np.sum(y)
        
        
################# Solving by minimization ###########################

        # converting symbolic representation of the expression with \theta to a mathematical one
        fun_float = lambdify(theta, -fun)
        
#        plot(fun_float, title='Function for both rhos', xlabel = 'theta', ylabel=None)

        # minimizing the function with the initial guess start_theta

#        result = spo.minimize(-fun_float, start_theta) # multiply the function by a minus as we want to have a maximum instead of minimum
        #print(result)
        
#        if result.success == False:
#            raise ValueError('Steady state minimization failed, try another guess for theta_start')
                    
#        else:
            
#            theta_result = result.x
            
            
################# Solving by finding a zero ###########################

        #change tolerance if needed
        mp.dps = 12
        
        #find zeros on the interval [0, strat_theta]
        theta_result = nsolve(fun, (0, start_theta))
        
        #check if the value of the found zero
#        zero_check = (fun.subs(theta, theta_result)).evalf()
        
        
 ####################################################################       
        
        # constructing the Euler rotation matrix with the found angle theta    
        U_euler_result = [[float(np.cos(theta_result)), float(np.sin(theta_result))], [float(-np.sin(theta_result)), float(np.cos(theta_result))]]
        
        # acting with the rotation matrix on the degenrate states gives positive semi-definite density matrices
        #stacking the two density matrices
        rho_p = np.dot(np.kron(U_euler_result, np.eye(int(2**L))), rho_degen_o)    
        
        # normalizing the density matrices
        rho_steady_rot1 = rho_p[0:2**L,0:2**L]/np.trace(rho_p[0:2**L,0:2**L])
        rho_steady_rot2 = rho_p[2**L:2*2**L,0:2**L]/np.trace(rho_p[2**L:2*2**L,0:2**L])
        
        print('norm rho1=', np.trace(rho_steady_rot1))
        print('norm rho2=', np.trace(rho_steady_rot2))
        
        # build the steady state as the linear superposition of the hermitian, positive semi-definite degenerate density matrices
        rho_vec_steady = (rho_steady_rot1 + rho_steady_rot2)
        rho_vec_steady = (rho_vec_steady / np.trace(rho_vec_steady)).reshape(2**(2*L),)
        
#%%
       

#################### 2nd method with density matrices ###########################  
        #normalizing
        
#        rho_steady1 = rho_steady1_ho/np.trace(rho_steady1_ho)
#        rho_steady2 = rho_steady2_ho/np.trace(rho_steady2_ho)      
        
#        print('norm rho1=', np.trace(rho_steady1))
#        print('norm rho2=', np.trace(rho_steady2))
        
        

#        print('rho_steady1_ho-np.conj(np.transpose(rho_steady1_ho))=',rho_steady1-np.conj(np.transpose(rho_steady1)))
        
#        print('rho_steady2_ho-np.conj(np.transpose(rho_steady2_ho))=',rho_steady2-np.conj(np.transpose(rho_steady2)))
        
#        print('np.trace(rho_steady1_ho*rho_steady2_ho)=',np.trace(rho_steady1*rho_steady2))

        
        # calculation the overlap of the initial state with each degenerate steady state
#        rho_overlap1 =  np.trace(np.matmul(rho_steady1 , rho_vec.reshape(2**L, 2**L) ))
#        rho_overlap2 =  np.trace(np.matmul(rho_steady2 , rho_vec.reshape(2**L, 2**L) ))
        
#        print('rho_overlap1', rho_overlap1)
#        print('rho_overlap2', rho_overlap2)
                
        #normalizing the density vector
#        normalizing_factor_tot = np.sqrt(abs(rho_overlap1)**2 + abs(rho_overlap2)**2)
        
#        normalizing_factor_rho1 = np.sqrt(np.trace(np.matmul(rho_steady1, rho_steady1)))
#        normalizing_factor_rho2 = np.sqrt(np.trace(np.matmul(rho_steady2, rho_steady2)))

#        rho_steady = rho_overlap1  * rho_steady1  + rho_overlap2  * rho_steady2 

        # for positive angle:
        #rho_steady = - rho_overlap1 / normalizing_factor_tot * rho_steady2  + rho_overlap2 / normalizing_factor_tot * rho_steady1 
        
        #theta_test = 0.037659
        
        #theta_test = -1.53808
        

        #rho_steady = np.cos(theta_test) * rho_steady1  + np.sin(theta_test) * rho_steady2         

#        rho_steady = rho_steady / np.trace(rho_steady)
        
#        print('norm rho_steady=', np.trace(rho_steady))
        
        #vectorizing rho
#        rho_vec_steady = rho_steady.reshape(len(psi_full)**2,)
 
#################### 3rd method without hermiticity ###########################  
        #normalizing
        
#        rho_steady1 = rho_degen1/np.trace(rho_degen1)
#        rho_steady2 = rho_degen2/np.trace(rho_degen2)      
        
#        print('norm rho1=', np.trace(rho_steady1))
#        print('norm rho2=', np.trace(rho_steady2))
        
#        rho_steady1_dagger = np.transpose(np.conj(rho_degen1))
#        rho_steady2_dagger = np.transpose(np.conj(rho_degen2))  
        
#        rho_steady1_o = rho_steady1
#        rho_steady2_o = rho_steady2 - np.trace(rho_steady1_o * rho_steady2_dagger)/np.trace(rho_steady1_o * rho_steady1_dagger) * rho_steady1_o 

#        rho_steady1_o_dagger = np.transpose(np.conj(rho_steady1_o))
#        rho_steady2_o_dagger = np.transpose(np.conj(rho_steady2_o))
        

#        print('rho_steady1_o-np.conj(np.transpose(rho_steady1_o))=' , rho_steady1_o - rho_steady1_o_dagger)
        
#        print('rho_steady2_o-np.conj(np.transpose(rho_steady2_o))=' , rho_steady2_o - rho_steady2_o_dagger)
        
#        print('np.trace(rho_steady1_o*rho_steady2_o)=' , np.trace(rho_steady1*rho_steady2_o_dagger))

        
        # calculation the overlap of the initial state with each degenerate steady state
#        rho_overlap1 =  np.trace(np.matmul(rho_vec.reshape(2**L, 2**L) , rho_steady1_o_dagger))
#        rho_overlap2 =  np.trace(np.matmul(rho_vec.reshape(2**L, 2**L) , rho_steady2_o_dagger))
        
#        print('rho_overlap1', rho_overlap1)
#        print('rho_overlap2', rho_overlap2)
                
        #normalizing the density vector
#        normalizing_factor_tot = np.sqrt(abs(rho_overlap1)**2 + abs(rho_overlap2)**2)
        
#        normalizing_factor_rho1 = np.sqrt(np.trace(np.matmul(rho_steady1_o, rho_steady1_o_dagger)))
#        normalizing_factor_rho2 = np.sqrt(np.trace(np.matmul(rho_steady2_o, rho_steady2_o_dagger)))

#        rho_steady = rho_overlap1 * rho_steady1 + rho_overlap2 * rho_steady2 

        #rho_steady = rho_steady + np.conj(np.transpose(rho_steady))

#        rho_steady = rho_steady / np.trace(rho_steady)
        
#        print('norm rho_steady=', np.trace(rho_steady))

#        rho_steady = rho_steady + np.conj(np.transpose(rho_steady))
        
        #vectorizing rho
#        rho_vec_steady = rho_steady.reshape(len(psi_full)**2,)
              

#################### 4th method with vectorized non-hermitian density ###########################  

        # extracting the degenerate vectorized density matrices
#        rho_vec_degen1 = eigvectors[:,zero1_idx].reshape(2**(2*L),)
#        rho_vec_degen2 = eigvectors[:,zero2_idx].reshape(2**(2*L),)
                
        # setting the first vector as orthonormal
#        rho_vec_degen1_on = rho_vec_degen1
        
        # orthogonalizing the second vector relative the first one
#        rho_vec_degen2_o = rho_vec_degen2 - np.dot(np.conj(np.transpose(rho_vec_degen2)), rho_vec_degen1_on) * rho_vec_degen1_on  
        # normalizing the second vector
#        rho_vec_degen2_on = rho_vec_degen2_o / np.sqrt(np.dot(np.conj(np.transpose(rho_vec_degen2_o)), rho_vec_degen2_o))
        
#        print('orthogonality of rho_vec_degen1_on and rho_vec_degen2_on = ', np.dot(np.conj(np.transpose(rho_vec_degen1_on)), rho_vec_degen2_on))
        
#        print('norm rho_vec_degen1_on=', sum(i*np.conj(i)  for i in rho_vec_degen1_on))
#        print('norm rho_vec_degen2_on=', sum(i*np.conj(i)  for i in rho_vec_degen2_on))

        # calculation the overlap of the initial state with each degenerate steady state
#        rho_overlap1 =  np.dot(np.conj(np.transpose(rho_vec_degen1_on)) , rho_vec)
#        rho_overlap2 =  np.dot(np.conj(np.transpose(rho_vec_degen2_on)) , rho_vec)
        
#        print('rho_overlap1', rho_overlap1)
#        print('rho_overlap2', rho_overlap2)

#        rho_vec_steady = rho_overlap1 * rho_vec_degen1_on + rho_overlap2 * rho_vec_degen2_on

        #rho_vec_steady = rho_vec_steady / np.sqrt(sum(i*np.conj(i) for i in rho_vec_steady))
                              
#        rho_vec_steady = rho_vec_steady / np.trace(rho_vec_steady.reshape(2**L,2**L))
        
#        print('norm rho_vec_steady=', sum(i*np.conj(i)  for i in rho_vec_steady))

    else:
        raise ValueError('Number of zero eigenvalues of Lindbladian should be 1 or 2 ')
    

#%%
    
    
    #Calculate steady-state
    info_per_scale_steady = infoLattice.calc_info_per_scale(infoLattice.calc_info_lndbld(rho_vec_steady), bc=BC)
    ave_steady = [info_per_scale_steady[x] for x in np.arange(L)+1]
    
    print("info_per_scale_steady", info_per_scale_steady)
    
    interp_func_steady = PchipInterpolator(np.arange(L), ave_steady)
    interp_scales_steady = np.arange(np.arange(L)[0], np.arange(L)[-1] + 0.1, 0.1)
    interp_average_steady= interp_func_steady(interp_scales_steady)
    marker_std = '3'



print('Time elapsed: {:.2e} s'.format(time.time() - start_time))

#%%

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
    
    
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=400)

    
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
        ax1.set_xticks(np.arange(len(scales)))
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
        
        color_map = plt.get_cmap("Oranges")
        
        colors = color_map(np.linspace(0, 1, 20))
        
        colors[0] = [1, 1, 1, 1]


        custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

        info_latt = infoLattice.calc_info_lndbld(rho_vec)
        
        print('info_latt', info_latt)
        r = 1/(4*L)
        #fig, ax = plt.subplots(dpi=300)
        for l in info_latt:
            for x in range(len(info_latt[l])):
                ax2.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, facecolor=custom_cmap(info_latt[l][x]), edgecolor='black', linewidth = 0.2))
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
                
        markermap = ['o', 'o', '^', '^', 'v', '^', '>', '<', 'v', '^']
    
        for w in [1, L-1]:
            ax3.plot(x_linspace, info_latt_t[:,w], marker=markermap[w], markerfacecolor='None', color = 'black', linewidth = 1, label = (r"$\mathscr{i}^%d$" % (w)))

        ax3.plot(x_linspace, tot_info_t, linestyle = '--', color='green', label = ("$I_{tot}$"))

        ax3.legend(fontsize=14)
        
        ax3.set_xlabel('t', fontsize=16)
#        ax3.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18, labelpad=1)
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
    
