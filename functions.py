import numpy as np
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from functools import reduce
from itertools import product
from scipy.sparse.csgraph import connected_components
from infoLattice import calc_info_from_corr_matr



def P_parity_block_diag(Lndbld, L):

    # ---------------------------
    # 1. DIAGONALIZE THE LINDBLADIAN
    # ---------------------------
    eigvalues, evecs = np.linalg.eig(Lndbld)  # right eigenvectors


##############################################Parity Block-diagonalization#############################################
    # ---------------------------
    # 2. BUILD HILBERT-SPACE PARITY OPERATOR
    # ---------------------------
    # Fock basis: parity = (-1)^(total particle number)

    basis_states = list(product(range(2), repeat=L)) # generates possible Fock states (1000), (0100)...(1111)

    # parity eigenvalues for each basis state
    N_tot = np.array([sum(state) for state in basis_states])
    P = np.diag((-1) ** N_tot)  # Hilbert-space parity operator

    # ---------------------------
    # 3. BUILD PARITY OPERATOR THAT ACTS ON RHO_VEC
    # ---------------------------
    P_L = np.kron(P, P.conj())  # acts on  vectorized rho due to (A rho B) = B^T kron A vec(rho), so (P rho P^dagger) = P kron P^* vec(rho)

    # ---------------------------
    # 4. DIAGONALIZE PARITY OPERATOR
    # ---------------------------
    eigvals_P, eigvecs_P = np.linalg.eig(P_L)

    even_idx = [i for i, val in enumerate(eigvals_P) if np.isclose(val, 1)]
    odd_idx = [i for i, val in enumerate(eigvals_P) if np.isclose(val, -1)]

    # Sort eigenvectors by parity: first even then odd

    V_parity = np.hstack([eigvecs_P[:, even_idx], eigvecs_P[:, odd_idx]])
    size_even = len(even_idx)
    size_odd = len(odd_idx)

    # ---------------------------
    # 5. EXPRESS LINDBLADIAN IN PARITY BASIS
    # ---------------------------
    L_block = V_parity.conj().T @ Lndbld @ V_parity

    L_even = L_block[:size_even, :size_even]
    L_odd = L_block[size_odd:, size_odd:]

    eig_even = np.linalg.eigvals(L_even)
    eig_odd = np.linalg.eigvals(L_odd)

    # Sort Lindbladian eigenvectors by magnitude of real part
    eig_even_sorted = sorted(eig_even, key=lambda x: abs(np.real(x)))
    eig_odd_sorted = sorted(eig_odd, key=lambda x: abs(np.real(x)))

    print("\nEven parity eigenvalues:")
    for val in eig_even_sorted:
        print(val)

    print("\nOdd parity eigenvalues:")
    for val in eig_odd_sorted:
        print(val)

    # ---------------------------
    # 7. PLOT BLOCK-DIAGONAL LINDBLAD
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(L_block), cmap='viridis', origin='upper')
    plt.colorbar(label='|L_block|')
    plt.title('Parity-block-diagonal Lindbladian')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.axhline(size_even - 0.5, color='red', linestyle='--')
    plt.axvline(size_even - 0.5, color='red', linestyle='--')
    plt.show()

    #print('Lindbladian eigvals', sorted(eigvalues, reverse=True))

    # Extract real and imaginary parts
    re = np.real(eigvalues)
    im = np.imag(eigvalues)

    # Plot
    plt.figure()
    plt.scatter(re, im, facecolors='none', edgecolors='k', linewidths=0.2)
    plt.axhline(0)
    plt.axvline(0)

    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Lindbladian spectrum")

    plt.show()



def calc_steady_state(Lndbld, L):
    eigvalues, eigvectors = np.linalg.eig(Lndbld)

#    if np.sum((np.real(eigvalues) < 1e12) & (np.real(eigvalues) > -1e-12)) == 1:         #if there is only 1 zero egenvalue
#        # For non-degenerate steady state:
#        zero_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-1])
#        print('Lndbld steady state eigvalue:', eigvalues[zero_idx])
#        #print('Lndbld eigvalues', sorted(np.real(eigvalues)))
#
#        # Normalizing by dividing the whole matrix by its trace
#        rho_vec_steady = eigvectors[:,zero_idx] * 1/np.trace(eigvectors[:,zero_idx].reshape(2**L, 2**L))
#
#    elif np.sum((np.real(eigvalues) < 1e12) & (np.real(eigvalues) > -1e-12)) == 2:
#        # For double-degenerate steady state
#
#        # the two zero eigenvalues should be the largest ones as Lindbladian has all eigenvalues non-positive
#        zero1_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-1])
#        zero2_idx = np.where(np.real(eigvalues) == sorted(np.real(eigvalues))[-2])
#
#        #print('Lndbld steady state eigvalues:', eigvalues[zero1_idx], eigvalues[zero2_idx])
#
#
#        #extracting the two degenerate steady states
#        rho_degen1 = eigvectors[:,zero1_idx].reshape(2**L, 2**L)
#        rho_degen2 = eigvectors[:,zero2_idx].reshape(2**L, 2**L)
#
    ##################### 1st method (arxiv.org/pdf/2101.10236) ###########################
    #
    #        #making the density matrices hermitian
    #        rho_degen1_ho = rho_degen1 + np.conj(np.transpose(rho_degen1))
    #        rho_degen2_h = rho_degen2 + np.conj(np.transpose(rho_degen2))
    #
    #        #making the second density matrix orthogonal to the first one
    #        rho_degen2_ho = rho_degen2_h - np.trace(rho_degen1_ho * rho_degen2_h)/np.trace(rho_degen1_ho * rho_degen1_ho) * rho_degen1_ho
    #
    #        #stacking the two density matrices
    #        rho_degen_o = np.concatenate((rho_degen1_ho, rho_degen2_ho))
    #
    ##        rho_degen_o[(abs(rho_degen_o) > -1e-12) & (abs(rho_degen_o) < 1e-12)] = 0   # setting all negligeble values to zero for faster calculations
    #
    #
    #        # minimizing the function with the initial guess start_theta
    #
    #        start_theta = 0.3
    #
    ##        theta_result = spo.fsolve(SS_degen, x0 = start_theta, args = (rho_degen1_ho, rho_degen2_ho, L, rho_degen_o))
    #
    #
    #        result = spo.minimize(SS_degen, x0 = start_theta, args = (rho_degen1_ho, rho_degen2_ho, L, rho_degen_o), bounds = [(-np.pi, np.pi)], tol=1e-13)
    #        #print(result)
    #
    #        if result.success == False:
    #            raise ValueError('Steady state minimization failed, try another guess for theta_start')
    #
    #        else:
    #
    #            theta_result = result.x
    # %%

    #        # constructing the Euler rotation matrix with the found angle theta
    #        U_euler_result = [[float(np.cos(theta_result)), float(-np.sin(theta_result))], [float(np.sin(theta_result)), float(np.cos(theta_result))]]
    #
    #        # acting with the rotation matrix on the degenrate states gives positive semi-definite density matrices
    #        #stacking the two density matrices
    #        rho_p = np.dot(np.kron(U_euler_result, np.eye(int(2**L))), rho_degen_o)
    #
    #        # normalizing the density matrices
    #        rho_steady_rot1 = rho_p[0:2**L,0:2**L]/np.trace(rho_p[0:2**L,0:2**L])
    #        rho_steady_rot2 = rho_p[2**L:2*2**L,0:2**L]/np.trace(rho_p[2**L:2*2**L,0:2**L])
    #
    #        print('norm rho1=', np.trace(rho_steady_rot1))
    #        print('norm rho2=', np.trace(rho_steady_rot2))
    #
    #        rho_steady = rho_steady_rot1
    #
    #        rho_vec_steady = (rho_steady / np.trace(rho_steady)).reshape(2**(2*L),)

    # build the steady state as the linear superposition of the hermitian, positive semi-definite degenerate density matrices
    #        rho_overlap1 =  np.trace(np.matmul(rho_steady_rot1 , rho_vec.reshape(2**L, 2**L) ))
    #        rho_overlap2 =  np.trace(np.matmul(rho_steady_rot2 , rho_vec.reshape(2**L, 2**L) ))

    #        rho_steady = rho_overlap1  * rho_steady_rot1  + rho_overlap2  * rho_steady_rot2
    # %%

    #        y = []
    #        for theta in np.arange(-np.pi, np.pi, 0.01):
    #            y.append(np.real(SS_degen(theta, rho_degen1_ho, rho_degen2_ho, L, rho_degen_o)))
    #        plt.plot(np.arange(-np.pi, np.pi, 0.01), y)
    #        plt.ylim([0,10.5])
    #        plt.xlabel('Theta')
    #        plt.show()
    #
    #        y = []
    #        for theta in np.arange(-np.pi/5, np.pi/5, 0.01):
    #            y.append(np.real(SS_degen(theta, rho_degen1_ho, rho_degen2_ho, L, rho_degen_o)))
    #        plt.plot(np.arange(-np.pi/5, np.pi/5, 0.01), y)
    #        plt.ylim([0,0.05])
    #        plt.xlabel('Theta')
    #        plt.show()

    # %%

    #################### 2nd method with density matrices ###########################
    # normalizing

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

    # normalizing the density vector
    #        normalizing_factor_tot = np.sqrt(abs(rho_overlap1)**2 + abs(rho_overlap2)**2)

    #        normalizing_factor_rho1 = np.sqrt(np.trace(np.matmul(rho_steady1, rho_steady1)))
    #        normalizing_factor_rho2 = np.sqrt(np.trace(np.matmul(rho_steady2, rho_steady2)))

    #        rho_steady = rho_overlap1  * rho_steady1  + rho_overlap2  * rho_steady2

    # for positive angle:
    # rho_steady = - rho_overlap1 / normalizing_factor_tot * rho_steady2  + rho_overlap2 / normalizing_factor_tot * rho_steady1

    # theta_test = 0.037659

    # theta_test = -1.53808

    # rho_steady = np.cos(theta_test) * rho_steady1  + np.sin(theta_test) * rho_steady2

    #        rho_steady = rho_steady / np.trace(rho_steady)

    #        print('norm rho_steady=', np.trace(rho_steady))

    # vectorizing rho
    #        rho_vec_steady = rho_steady.reshape(len(psi_full)**2,)

    #################### 3rd method without hermiticity ###########################
    # normalizing

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

    # normalizing the density vector
    #        normalizing_factor_tot = np.sqrt(abs(rho_overlap1)**2 + abs(rho_overlap2)**2)

    #        normalizing_factor_rho1 = np.sqrt(np.trace(np.matmul(rho_steady1_o, rho_steady1_o_dagger)))
    #        normalizing_factor_rho2 = np.sqrt(np.trace(np.matmul(rho_steady2_o, rho_steady2_o_dagger)))

    #        rho_steady = rho_overlap1 * rho_steady1 + rho_overlap2 * rho_steady2

    # rho_steady = rho_steady + np.conj(np.transpose(rho_steady))

    #        rho_steady = rho_steady / np.trace(rho_steady)

    #        print('norm rho_steady=', np.trace(rho_steady))

    #        rho_steady = rho_steady + np.conj(np.transpose(rho_steady))

    # vectorizing rho
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

    # rho_vec_steady = rho_vec_steady / np.sqrt(sum(i*np.conj(i) for i in rho_vec_steady))

    #        rho_vec_steady = rho_vec_steady / np.trace(rho_vec_steady.reshape(2**L,2**L))

    #        print('norm rho_vec_steady=', sum(i*np.conj(i)  for i in rho_vec_steady))

    #    else:
    #        raise ValueError('Number of zero eigenvalues of Lindbladian should be 1 or 2 ')

    # %%

    # Calculate steady-state
#    info_per_scale_steady = infoLattice.calc_info_per_scale(infoLattice.calc_info_lndbld(rho_vec_steady), bc=BC)
#    ave_steady = [info_per_scale_steady[x] for x in np.arange(L) + 1]
#
#    print("info_per_scale_steady", info_per_scale_steady)
#
#    interp_func_steady = PchipInterpolator(np.arange(L), ave_steady)
#    interp_scales_steady = np.arange(np.arange(L)[0], np.arange(L)[-1] + 0.1, 0.1)
#    interp_average_steady = interp_func_steady(interp_scales_steady)
#    marker_std = '3'
#
#    # plotting steady state separately
#
#    save_gen_dir = 'results/clean_{0}/BC_{1}/Syst_{2}/bath_{3}/L{4}/J{5}/mu{6}/seed{7}/kappa{8}/g{9}/delta{10}/'.format(
#        *(clean, BC, system_type, bath, L, XX, Z, seed, kappa, g, delta))
#
#    plt.plot(interp_scales_steady, interp_average_steady, color='black', label='Steady state')
#    plt.plot(np.arange(L), ave_steady, marker=marker_std, linestyle='None', markersize=10, color='red',
#             label='Steady state')
#    plt.ylim([-0.1, L])
#    plt.ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
#    plt.xlabel(r"$\ell$", fontsize=16)
#    plt.tick_params(axis='x', labelsize=14)
#    plt.tick_params(axis='y', labelsize=14)
#
#    legend_elements = [
#        Line2D([0], [0], color='black', marker=marker_std, markeredgecolor='red', markersize=10, linestyle='-',
#               label='Steady state')
#    ]
#
#    plt.legend(handles=legend_elements, fontsize=14)
#
#    plt.savefig(save_gen_dir + "SteadyState.pdf", dpi=400, bbox_inches='tight')
#
#    plt.show()
#
#    np.savetxt(save_gen_dir + "Steady_state_Info_per_scale", ave_steady)

#    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
#    ax1.plot(interp_scales_steady, interp_average_steady, color='black', label='Steady state')
#    ax1.plot(scales, ave_steady, marker=marker_std, linestyle='None', markersize=10, color='red',
#                 label='Steady state')
#
#
#    ax1.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
#    ax1.set_ylim([-0.1, L + 0.1])
#    ax1.set_xlabel(r"$\ell$", fontsize=16)
#    ax1.set_xticks(np.arange(len(scales)))
#    ax1.tick_params(axis='x', labelsize=14)
#    ax1.tick_params(axis='y', labelsize=14)
#
## ax1.set_aspect('equal', adjustable='box')
#
#
#    legend_elements = [
#        Line2D([0], [0], color='black', marker=marker_ini, markeredgecolor='deepskyblue', markersize=8, linestyle='-',
#               label='Initial state'),
#        Line2D([0], [0], color='black', marker=marker_fin, markeredgecolor='blue', markersize=10, linestyle='-',
#               label='Final state'),
#        Line2D([0], [0], color='black', marker=marker_std, markeredgecolor='red', markersize=10, linestyle='-',
#               label='Steady state')
#    ]







def make_initial_state(H_full, Lndbld, L, initial_state_type, epsilon):

    if initial_state_type == 'eigvec_H':
        eigvals, psi_eigvec = eigh(H_full)

        # eigvals, psi_eigvec = eigsh(H_full, k=k_eigs, sigma=0, which='LM')

        # Index for the ground state = 0
        idx = int(0)

        # Or provide another index of the eigstate below
        # idx = int(2**L/2+6)

        psi_full = psi_eigvec[:,idx]

        #Or make a superposition of selected eigenstates
        # psi_full = psi_eigvec[:,idx+1] + psi_eigvec[:,idx+4] + psi_eigvec[:,idx+7]
        # psi_full = psi_full/np.sqrt(sum(i**2 for i in psi_full))
        # print('norm psi' , sum(i**2 for i in psi_full))

        rho_vec = np.kron(psi_full, np.conj(psi_full))
        #np.save('Steady_State_L6_topo_pure.npy', rho_vec)

    elif initial_state_type == 'external':
        psi_full = np.loadtxt('psi_L6_triv.txt')
        rho_vec = np.kron(psi_full, np.conj(psi_full))

    elif initial_state_type == 'mixed_triv':
        prob_up = 1 - epsilon
        site_1 = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
        rho=site_1
        for i in np.arange(0, L-1):
            rho = np.kron(rho, site_1)

        rho_vec = rho.reshape(-1,1)

    elif initial_state_type == 'mixed_Bell':
        bell = (1 / 2) * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        I4 = np.eye(4)
        mixed_bell = (1 - epsilon) * bell + epsilon * I4 / 4  # a weighted sum of the Bell and the maximally mized state (I4)/4
        rho = reduce(np.kron, [mixed_bell] * (L // 2))
        rho_vec = rho.reshape(2 ** (2 * L), 1)

    elif initial_state_type == 'eigvec_L':
        eigvalues_left, evecs_left = np.linalg.eig(Lndbld.conj().T) # left eigenvectors are not actual states

        eigvalues_right, evecs_right = np.linalg.eig(Lndbld)  # right eigenvectors are modes

        # Sorted right side (keep this)
        order_right = np.argsort(np.abs(np.real(eigvalues_right)))
        eigvals_right_sorted = eigvalues_right[order_right]

        # Now build matching for left eigenvalues
        unused = set(range(len(eigvalues_left)))
        order_left = np.empty_like(order_right)

        for i, lam_r in enumerate(eigvals_right_sorted):
            # find closest unused left eigenvalue
            best_j = min(unused, key=lambda j: abs(eigvalues_left[j] - lam_r))
            order_left[i] = best_j
            unused.remove(best_j)


        evecs_left_sorted = evecs_left[:, order_left]
        evecs_right_sorted = evecs_right[:, order_right]

        d = int(2**L)

        # define vectorized rho_0 as the trivial state
        site_1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho_0 = site_1
        for i in np.arange(0, L - 1):
            rho_0 = np.kron(rho_0, site_1)

        rotate_away = np.array(eigvalues_left)
        rotated_rho = rho_0
        for i in range(20):
            rotated_rho =  rotated_rho - np.trace(evecs_left_sorted[:,i].reshape((d, d)) * rho_0) * evecs_right_sorted[:,i].reshape((d, d))

            print('np.trace(evecs_left_sorted[:,i].reshape((d, d)) * evecs_right_sorted[:,i].reshape((d, d))) ', np.trace(evecs_left_sorted[:,i].reshape((d, d)) * evecs_right_sorted[:,i].reshape((d, d))))

        # Enforcing hermiticity
        #rotated_rho = (rotated_rho + rotated_rho.conj().T)
        #rotated_rho /= np.trace(rotated_rho)  # normalize

        # Enforcing non-negativity
        evals, evecs = np.linalg.eigh(rotated_rho)
        print('evals', evals)
        print('trace rho', np.trace(rotated_rho))

        # Vectorized version if needed
        rho_vec = rotated_rho.reshape(-1)
        #print('norm rho_vec' , np.trace(combined_rho))

    return rho_vec


def corr_block_from_density_matrix(rho_block):
    """
    Build C and A^dagger for a small block only.

    rho_block shape:
        (2,2) for one site
        (4,4) for two sites
        in general (2**ell, 2**ell)
    """
    dim = rho_block.shape[0]
    L_block = int(np.log2(dim))

    C_block = np.zeros((L_block, L_block), dtype=np.complex128)
    A_block = np.zeros((L_block, L_block), dtype=np.complex128)

    for m, k in product(range(L_block), range(L_block)):
        ops = calc_opdm_operator(L_block, m, k)
        C_block[m, k] = np.trace(rho_block @ ops["he"][m, k])
        A_block[m, k] = np.trace(rho_block @ ops["ee"][m, k])

    return C_block, A_block

def init_corr_from_blocks(L, initial_state_type, epsilon):
    C_mk = np.zeros((L, L), dtype=np.complex128)
    A_mk_dagger = np.zeros((L, L), dtype=np.complex128)

    if initial_state_type == "mixed_triv":
        prob_up = 1 - epsilon
        rho_site = np.array(
            [[prob_up, 0.0],
             [0.0, 1.0 - prob_up]],
            dtype=np.complex128
        )

        C_block, A_block = corr_block_from_density_matrix(rho_site)

        for i in range(L):
            C_mk[i:i+1, i:i+1] = C_block
            A_mk_dagger[i:i+1, i:i+1] = A_block

    elif initial_state_type == "mixed_Bell":
        if L % 2 != 0:
            raise ValueError("mixed_Bell requires even L")

        bell = 0.5 * np.array(
            [[1, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]],
            dtype=np.complex128
        )
        I4 = np.eye(4, dtype=np.complex128)
        rho_pair = (1 - epsilon) * bell + epsilon * I4 / 4.0

        C_block, A_block = corr_block_from_density_matrix(rho_pair)

        for p in range(L // 2):
            i = 2 * p
            C_mk[i:i+2, i:i+2] = C_block
            A_mk_dagger[i:i+2, i:i+2] = A_block

    elif initial_state_type == "imperfect_mixed_Bell":
        if L % 2 != 0:
            raise ValueError("imperfect_mixed_Bell requires even L")

        # imperfect_bell = sqrt(1/5)|00> + sqrt(4/5)|11>
        imperfect_bell = np.array([
            [1 / 5, 0, 0, np.sqrt(4) / 5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [np.sqrt(4) / 5, 0, 0, 4 / 5]],
            dtype=np.complex128
        )
        I4 = np.eye(4, dtype=np.complex128)
        rho_pair = (1 - epsilon) * imperfect_bell + epsilon * I4 / 4.0

        C_block, A_block = corr_block_from_density_matrix(rho_pair)

        for p in range(L // 2):
            i = 2 * p
            C_mk[i:i+2, i:i+2] = C_block
            A_mk_dagger[i:i+2, i:i+2] = A_block

    elif initial_state_type == "mixed_W":
        if L % 3 != 0:
            raise ValueError("mixed_W requires L divisible by 3")

        # |W> = (|001> + |010> + |100>) / sqrt(3)
        w_vec = np.zeros(8, dtype=np.complex128)
        w_vec[1] = 1 / np.sqrt(3)
        w_vec[2] = 1 / np.sqrt(3)
        w_vec[4] = 1 / np.sqrt(3)

        w_state = np.outer(w_vec, np.conj(w_vec))

        I8 = np.eye(8, dtype=np.complex128)
        rho_block = (1.0 - epsilon) * w_state + epsilon * I8 / 8.0

        C_block, A_block = corr_block_from_density_matrix(rho_block)

        for p in range(L // 3):
            i = 3 * p
            C_mk[i:i + 3, i:i + 3] = C_block
            A_mk_dagger[i:i + 3, i:i + 3] = A_block

    elif initial_state_type == "mixed_locally_cluster":
        if L % 4 != 0:
            raise ValueError("mixed_locally_cluster requires L divisible by 4")

        # 4-qubit linear cluster state on an open chain:
        # start from |++++> and apply CZ on (0,1), (1,2), (2,3)
        psi = np.ones(16, dtype=np.complex128) / 4.0

        for x in range(16):
            b0 = (x >> 3) & 1
            b1 = (x >> 2) & 1
            b2 = (x >> 1) & 1
            b3 = x & 1
            psi[x] *= (-1) ** (b0 * b1 + b1 * b2 + b2 * b3)

        rho_block = np.outer(psi, np.conj(psi))

        # Local Z dephasing on each qubit:
        # rho -> (1-eps) rho + eps Z_q rho Z_q, applied independently for q=0..3
        Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)

        def embed_single_site(op, q, n_sites=4):
            mats = [I2] * n_sites
            mats[q] = op
            out = mats[0]
            for m in mats[1:]:
                out = np.kron(out, m)
            return out

        def apply_local_dephasing(rho_in, p, q):
            Zq = embed_single_site(Z, q)
            return (1.0 - p) * rho_in + p * (Zq @ rho_in @ Zq)

        for q in range(4):
            rho_block = apply_local_dephasing(rho_block, epsilon, q)

        C_block, A_block = corr_block_from_density_matrix(rho_block)

        for p in range(L // 4):
            i = 4 * p
            C_mk[i:i+4, i:i+4] = C_block
            A_mk_dagger[i:i+4, i:i+4] = A_block


    else:
        raise ValueError(
            f"Unsupported initial_state_type '{initial_state_type}'. "
            "Supported types are 'mixed_triv', 'imperfect_mixed_Bell', "
            "'mixed_W', 'mixed_Bell' and 'mixed_locally_cluster'."
        )

    return C_mk, A_mk_dagger



def calc_opdm_operator(L,m,k):
    basic_operators = {
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        "+": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        "-": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        "1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    }

    rho = {'eh': {}, 'hh': {}, 'he': {}, 'ee': {}}

    string_m = ['1'] * L
    string_k = ['1'] * L

    #    print('i', i)
    if m > 0:
        for s in range(m):
            string_m[s] = 'z'
    #    print('string_i', string_i)

    #    print('j', j)
    if k > 0:
        for s in range(k):
            string_k[s] = 'z'
    #    print('string_j', string_j)

    string_m[m] = '+'
    sigma_plus_string_m = string_m.copy()  # sigma_z kron sigma_z kron ... kron sigma_plus kron 1 ... 1
    sigma_plus_m = basic_operators[sigma_plus_string_m[0]]
    for op in sigma_plus_string_m[1:]:
        sigma_plus_m = np.kron(sigma_plus_m, basic_operators[op])

    string_m[m] = '-'
    sigma_minus_string_m = string_m.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_minus_m = basic_operators[sigma_minus_string_m[0]]
    for op in sigma_minus_string_m[1:]:
        sigma_minus_m = np.kron(sigma_minus_m, basic_operators[op])

    string_k[k] = '-'
    sigma_minus_string_k = string_k.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_minus_k = basic_operators[sigma_minus_string_k[0]]
    for op in sigma_minus_string_k[1:]:
        sigma_minus_k = np.kron(sigma_minus_k, basic_operators[op])

    string_k[k] = '+'
    sigma_plus_string_k = string_k.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_plus_k = basic_operators[sigma_plus_string_k[0]]
    for op in sigma_plus_string_k[1:]:
        sigma_plus_k = np.kron(sigma_plus_k, basic_operators[op])

    # print('i= ', i)
    # print('sigma_plus_string_i: ', sigma_plus_string_i)
    # print('sigma_minus_string_i: ', sigma_minus_string_i)
    # print('j= ', j)
    # print('sigma_minus_string_j: ', sigma_minus_string_j)

    rho['eh'][m, k] = sigma_plus_m @ sigma_minus_k
    rho['hh'][m, k] = sigma_minus_m @ sigma_minus_k
    rho['he'][m, k] = sigma_minus_m @ sigma_plus_k
    rho['ee'][m, k] = sigma_plus_m @ sigma_plus_k

    return rho

def calc_ini_C_mk_A_mk_t0(rho_ini):

    L_loc = int(np.log2(len(rho_ini)))

    C_mk_t0 = np.zeros((L_loc, L_loc), dtype=np.complex128)
    A_mk_dagger_t0 = np.zeros((L_loc, L_loc), dtype=np.complex128)

    #print('L_loc', L_loc)
    for m, k in product(range(L_loc), range(L_loc)):
        rho = calc_opdm_operator(L_loc, m, k)
        C_mk = rho_ini @ rho['he'][m, k]
        C_mk_t0[m, k] += np.trace(C_mk)  # sigma_minus_i sigma_plus_j
        A_mk_dagger = rho_ini @ rho['ee'][m, k]  # sigma_plus_i sigma_plus_j
        A_mk_dagger_t0[m, k] += np.trace(A_mk_dagger)  # sigma_plus_i sigma_plus_j

    return C_mk_t0, A_mk_dagger_t0


def build_generalized_correlation_matrix(C_mk_t, A_mk_dagger_t):
    C_mk_t = np.asarray(C_mk_t, dtype=np.complex128)
    A_mk_dagger_t = np.asarray(A_mk_dagger_t, dtype=np.complex128)

    if C_mk_t.shape != A_mk_dagger_t.shape:
        raise ValueError("C_mk_t and A_mk_dagger_t must have the same shape")

    L_loc = C_mk_t.shape[0]
    generalized_corr = np.zeros((2 * L_loc, 2 * L_loc), dtype=np.complex128)
    generalized_corr[:L_loc, :L_loc] = np.eye(L_loc, dtype=np.complex128) - C_mk_t.T
    generalized_corr[:L_loc, L_loc:] = A_mk_dagger_t
    generalized_corr[L_loc:, :L_loc] = -A_mk_dagger_t.conj()
    generalized_corr[L_loc:, L_loc:] =  C_mk_t

    return generalized_corr



def evolve_C_mk_A_mk_delta_t(C_mk_t , A_mk_dagger_t, delta_t, gamma, J, h):

    L_loc = int(len(C_mk_t))

    I = np.eye(L_loc)
    delta_mk =  I

    nu_vec = 2 * np.ones(L_loc, dtype=np.complex128)
    nu_vec[0] = 1
    nu_vec[-1] = 1
    Nu = np.diag(nu_vec)

    s_vec = np.zeros(L_loc, dtype=np.complex128)
    s_vec[0] = 1
    s_vec[-1] = -1
    S = np.diag(s_vec)

    b_vec = np.ones(L_loc, dtype=np.complex128)
    b_vec[0] = 0
    b_vec[-1] = 0
    B = np.diag(b_vec)

    # C_{m,k+1}:
    C_mk_plus_1 = np.roll(C_mk_t, shift=-1, axis=1)
    C_mk_plus_1[:, -1] = 0
    # delta_{m,k+1}:
    delta_mk_plus_1 = np.roll(I, shift=-1, axis=1)
    delta_mk_plus_1[:, -1] = 0

    # C_{m-1,k}:
    C_m_minus_1k = np.roll(C_mk_t, shift=+1, axis=0)
    C_m_minus_1k[0, :] = 0
    # delta_{m-1,k}:
    delta_m_minus_1k = np.roll(I, shift=+1, axis=0)
    delta_m_minus_1k[0, :] = 0

    # C_{m+1,k}:
    C_m_plus_1k = np.roll(C_mk_t, shift=-1, axis=0)
    C_m_plus_1k[-1, :] = 0
    # delta_{m+1,k}:
    delta_m_plus_1k = np.roll(I, shift=-1, axis=0)
    delta_m_plus_1k[-1, :] = 0

    # C_{m,k-1}:
    C_mk_minus_1 = np.roll(C_mk_t, shift=+1, axis=1)
    C_mk_minus_1[:, 0] = 0
    # delta_{m,k-1}:
    delta_mk_minus_1 = np.roll(I, shift=+1, axis=1)
    delta_mk_minus_1[:, 0] = 0

    # A_dagger_{m,k+1}:
    A_dagger_mk_plus_1 = np.roll(A_mk_dagger_t, shift=-1, axis=1)
    A_dagger_mk_plus_1[:, -1] = 0

    # A_dagger_{m-1,k}:
    A_dagger_m_minus_1k = np.roll(A_mk_dagger_t, shift=+1, axis=0)
    A_dagger_m_minus_1k[0, :] = 0

    # A_dagger_{m+1,k}:
    A_dagger_m_plus_1k = np.roll(A_mk_dagger_t, shift=-1, axis=0)
    A_dagger_m_plus_1k[-1, :] = 0

    # A_dagger_{m,k-1}:
    A_dagger_mk_minus_1 = np.roll(A_mk_dagger_t, shift=+1, axis=1)
    A_dagger_mk_minus_1[:, 0] = 0

    #C_mk_t_next = C_mk_t + delta_t * (-1j * J *
    #                                     ( - C_m_minus_1k - C_m_plus_1k + C_mk_minus_1 + C_mk_plus_1 )
    #                                 - gamma * C_mk_t
    #                                 + 1j * J *
    #                                     ( - A_dagger_m_plus_1k + A_dagger_m_minus_1k  + A_dagger_mk_minus_1.conj() - A_dagger_mk_plus_1.conj())
    #                                 - gamma * 0.25 *
    #                                     ( delta_mk_plus_1 + delta_m_plus_1k - 2 * delta_mk )
    #                                  )

    diss_C = (
            0.25 * gamma * (
                    - (Nu @ C_mk_t + C_mk_t @ Nu)
                    - S @ A_mk_dagger_t
                    + A_mk_dagger_t.conj() @ S
                    - delta_mk_plus_1
                    - delta_m_plus_1k
                    + Nu
            )
    )
##
    C_mk_t_next = C_mk_t + delta_t * (
            - 1j * J * (-C_m_minus_1k - C_m_plus_1k + C_mk_minus_1 + C_mk_plus_1)
            + 1j * J * (
                    -A_dagger_m_plus_1k
                    + A_dagger_m_minus_1k
                    + A_dagger_mk_minus_1.conj()
                    - A_dagger_mk_plus_1.conj()
            )
            + diss_C
    )

##
##

    #A_mk_dagger_t_next = A_mk_dagger_t + delta_t * (- 1j * J *
    #                                                    (delta_mk_minus_1 - C_mk_minus_1.conj() - delta_mk_plus_1 + C_mk_plus_1.conj() + C_m_minus_1k - C_m_plus_1k)
    #                                                - 1j * J *
    #                                                     (A_dagger_m_minus_1k +  A_dagger_mk_minus_1 + A_dagger_m_plus_1k + A_dagger_mk_plus_1)
    #                                                - 1j * 4 * h * A_mk_dagger_t
    #                                                - gamma * A_mk_dagger_t
    #                                                - gamma * 0.25 * (  delta_m_plus_1k - delta_m_minus_1k )
    #                                                )

    diss_A_dagger = 0.25 * gamma * (
            - (Nu @ A_mk_dagger_t + A_mk_dagger_t @ Nu)
            - S @ C_mk_t
            + C_mk_t.conj() @ S
            #- np.outer(np.eye(L_loc, dtype=np.complex128)[0], np.eye(L_loc, dtype=np.complex128)[1])
            #+ np.outer(np.eye(L_loc, dtype=np.complex128)[:, 1], np.eye(L_loc, dtype=np.complex128)[:, 0])
            #+ B @ (delta_mk_minus_1 - delta_mk_plus_1) @ B  # + 2 * np.eye(L_loc, dtype=np.complex128)) @ B
            + delta_mk_plus_1
            - delta_mk_minus_1
    )

    A_mk_dagger_t_next = A_mk_dagger_t + delta_t * (
            - 1j * J * (
            delta_mk_minus_1
            - C_mk_minus_1.conj()
            - delta_mk_plus_1
            + C_mk_plus_1.conj()
            + C_m_minus_1k
            - C_m_plus_1k
    )
            - 1j * J * (
                    A_dagger_m_minus_1k
                    + A_dagger_mk_minus_1
                    + A_dagger_m_plus_1k
                    + A_dagger_mk_plus_1
            )
            + 1j * 4 * h * A_mk_dagger_t
            + diss_A_dagger
    )


    # Dissipation-only dynamics:
    #C_mk_t_next = C_mk_t + delta_t * 0.5 * (- 2 * gamma * C_mk_t - gamma * 0.5 * ( delta_mk_plus_1 + delta_m_plus_1k - 2 * delta_mk ) )
    #A_mk_dagger_t_next = A_mk_dagger_t + delta_t * 0.5 * (- 2 * gamma * A_mk_dagger_t - gamma * 0.5 * (  delta_m_plus_1k - delta_m_minus_1k ) )

    return C_mk_t_next, A_mk_dagger_t_next

def time_evo_C_mk_A_mk(rho_ini, delta_t, t_max, J, gamma, h):


    """
    Calculating the d/dt C_mk(t)
    """

    C_mk_t , A_mk_dagger_t = calc_ini_C_mk_A_mk_t0(rho_ini)

    time_steps = t_max / delta_t  # number of timesteps

    for _ in np.arange(time_steps):

        C_mk_t, A_mk_dagger_t = evolve_C_mk_A_mk_delta_t(C_mk_t, A_mk_dagger_t, delta_t, gamma, J, h)

    return C_mk_t, A_mk_dagger_t
