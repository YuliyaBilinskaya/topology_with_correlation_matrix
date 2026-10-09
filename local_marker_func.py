import numpy as np
import time
import itertools
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
from itertools import product
from scipy.sparse.csgraph import connected_components
from infoLattice import calc_info_from_corr_matr
from functions import calc_opdm_operator

def local_marker(L_x, x, P, S, site):
    """
    Calculates the local CS marker for the specified site on a 1d chain in the BdG basis

    ----------
    Parameters:
    L_x, L_y, L_z: Number of lattice sites in each direction
    n_orb : Number of orbitals
    n_sites: Number of lattice sites
    x, y, z: Vectors with the position of the lattice sites
    P : Valence band projector
    S: Chiral symmetry operator of the model
    site: Number of the site we calculate the marker on

    ----------
    Returns:
    Marker vector on each site
    """

    # Position operators  (take the particular site to be as far from the branch cut as possible)

    #This code rotatest the chain such that the site is in the middle of the chain. Effectively mapping the system onto the bulk only and omitting the edge states.
    #If one calculates this way than the average of the local markers gives the value of the chern number in the bulk.
#    half_Lx = np.floor(L_x / 2)
#    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + np.heaviside(half_Lx - x[site], 0) * abs(half_Lx - x[site])
#    x = (x + deltaLx) % L_x                                      # Relabel of the operators

    #calculating the Chern marker with the below will include both the edges and the bulk.
    X = np.concatenate((x, x))                                   # X vector in BdG basis
    X = np.reshape(X, (len(X), 1))                               # Column vector x

    # Marker calculation
    M = P @ S @ (X * P)                                          # Marker operator (BdG x position space)
#    print('M[site, site]', M[site, site])
#    print('M[L_x + site, L_x + site]', M[L_x + site, L_x + site])
    if np.allclose(np.imag(M[site, site]), 0j, atol = 1e-10) and np.allclose(np.imag(M[L_x + site, L_x + site]), 0j, atol = 1e-10):
        marker = -2 * (np.real(M[site, site]) + np.real(M[L_x + site, L_x + site]))    # Trace over each site
    else:
        raise ValueError(f'Imaginary part of the Marker operator should be zero')

    return marker

def parity_projector(many_body_rho, parity):
    L_loc = int(np.log2(len(many_body_rho)))
    L=L_loc
    sigma_z = [[1.0, 0.0], [0.0, -1.0]] # sigma_z
    parity_operator = sigma_z

    for s in range(L - 1):
        parity_operator = np.kron(sigma_z, parity_operator)

    eigvals, eigstates = np.linalg.eigh(parity_operator)

    pos_i = []
    neg_i = []
    positive_projector = np.zeros((2**L, 2**L))
    negative_projector = np.zeros((2**L, 2**L))
    for i in range(len(eigvals)):
        if eigvals[i] == 1:
            positive_projector +=  np.outer(eigstates[i], eigstates[i])
            pos_i.append(i)
        elif eigvals[i] == -1:
            negative_projector += np.outer(eigstates[i], eigstates[i])
            neg_i.append(i)
        else:
            raise ValueError(f'Eigenvalues of parity operator must be +/-1, but we got {eigvals[i]} for i = {i}')

#    print('Degen eigvecs are lin indep:',
#          all(np.allclose(eigstates[i] @ eigstates[j], 0, atol = 1e-10) for i in pos_i for j in pos_i if i != j))

#    print('positive_projector is a projector:', np.allclose(positive_projector, positive_projector @ positive_projector, atol=1e-10))

    if parity == 'positive':
        projected_many_body_rho = positive_projector @ many_body_rho @ positive_projector
    elif parity == 'negative':
        projected_many_body_rho = negative_projector @ many_body_rho @ negative_projector

#    print('trace of rho=', np.trace(projected_many_body_rho))
    projected_many_body_rho = projected_many_body_rho / np.trace(projected_many_body_rho)
#    print('trace of rho=', np.trace(projected_many_body_rho))

    return projected_many_body_rho

def band_flattening(rho):
    values, vectors = np.linalg.eigh(rho)
    idx = np.min(np.where(values > 0.5)[0])
    U = np.zeros(rho.shape, dtype=np.complex128)
    U[:, : idx] = vectors[:, : idx]  # Filling the projector
    rho_flat = U @ np.conj(np.transpose(U))
    return rho_flat

def calc_marker_state(many_body_rho, flatten = False, p_eigvals = False):
    # Single-particle density matrix
    L_loc = int(np.log2(len(many_body_rho)))
    rho_state_opdm = calc_opdm_from_rho(many_body_rho)
    norm_P, ij = calc_opdm_norm(many_body_rho)
    Id = np.eye(int(L_loc), dtype=np.complex128)
    S = np.kron([[0.0, 1.0], [1.0, 0.0]], Id)  # Sigma_x: Chiral symmetry of the chain (BdG x position space)

    if flatten:
        rho_state_opdm = band_flattening(rho_state_opdm)

    # Local chiral marker
    marker = np.zeros((L_loc,))
    #print('P is a projector:', np.allclose(rho_state_opdm, rho_state_opdm @ rho_state_opdm, atol=1e-8))
    #print('P eigenvalues:', np.real(np.linalg.eigvals(rho_state_opdm)))
    #    print('P',rho_state_opdm)
    #    print('P^2', rho_state_opdm@rho_state_opdm)
    for j in range(L_loc):
        marker[j] = local_marker(L_x=L_loc, x=np.arange(L_loc), P=rho_state_opdm, S=S, site=j)

    if p_eigvals:
        eigvals = np.linalg.eigvals(rho_state_opdm)
        if (np.abs(np.imag(eigvals)) > 1e-10).any():
            raise ValueError(f'Imaginary part of the OPDM eigenvalues should be zero')
        else:
            eigvals_subsyst = np.real(np.linalg.eigvals(rho_state_opdm))

    return marker, eigvals_subsyst, norm_P, ij


def calc_marker_state_from_opdm(opdm, flatten=False, p_eigvals=False):
    opdm = np.asarray(opdm, dtype=np.complex128)
    L_loc = opdm.shape[0] // 2
    Id = np.eye(int(L_loc), dtype=np.complex128)
    S = np.kron([[0.0, 1.0], [1.0, 0.0]], Id)

    if flatten:
        opdm = band_flattening(opdm)

    marker = np.zeros((L_loc,))
    for j in range(L_loc):
        marker[j] = local_marker(L_x=L_loc, x=np.arange(L_loc), P=opdm, S=S, site=j)

    eigvals_subsyst = None
    if p_eigvals:
        eigvals = np.linalg.eigvals(opdm)
        if (np.abs(np.imag(eigvals)) > 1e-10).any():
            raise ValueError('Imaginary part of the OPDM eigenvalues should be zero')
        eigvals_subsyst = np.real(eigvals)

    return marker, eigvals_subsyst


def calc_ave_marker_per_t(loaded_dens_mat, loaded_times, get_index=False, max_val=0.7):
    all_markers = defaultdict(dict)
    OPDM_eigvals = defaultdict(dict)
    ave_marker_per_t_subsyst = defaultdict(dict)
    var_marker_per_t_subsyst = defaultdict(dict)
    ave_marker_per_t = defaultdict(dict)
    var_marker_per_t = defaultdict(dict)
    for idx in range(0, int(len(loaded_dens_mat)), 100): #range(0, 5700, 10): ################################ CHOSE timelength and resolution of markers here#############################
        #print('idx', idx)
        d = loaded_dens_mat[idx]
        max_ell = max(k[1] for k in d.keys())
        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k[1] == max_ell]
            for key in max_ell_keys:
                #print('key', key)
                _marker, eigvals_subsyst,  _, _ = calc_marker_state(loaded_dens_mat[idx][key], flatten = False, p_eigvals = True)
                #print('marker', _marker)
                all_markers[idx][key] = _marker  # all markers per time t and and subsystem (n, max_ell)
                OPDM_eigvals[idx][key] = eigvals_subsyst  # eigenvalues of OPDM per time t and and subsystem (n, max_ell)
                #print('OPDM_eigvals', OPDM_eigvals[idx])
                ave_marker_per_t_subsyst[idx][key] = np.average(_marker[1:-1])  # average marker excluding mixed edge sites per t and subsystem
                var_marker_per_t_subsyst[idx][key] = np.var(_marker[1:-1])  # variance of the marker excluding mixed edge sites per t and subsystem
                #print('ave_marker_per_t_subsyst', ave_marker_per_t_subsyst[idx])


            vals_ave = list(ave_marker_per_t_subsyst[idx].values())
            ave_marker_per_t[idx] = np.sum(vals_ave) / len(vals_ave)  # average marker excluding mixed edge sites per t

            vals_var = list(var_marker_per_t_subsyst[idx].values())
            var_marker_per_t[idx] = np.sum(vals_var) / len(vals_var)  # average variance of the marker excluding mixed edge sites per t
            #print('ave_marker_per_t', ave_marker_per_t[idx])


#    last_idx = max(all_markers.keys())
#    print('Marker:', all_markers[50])
#    print('Average per t and subsyst:', ave_marker_per_t_subsyst[last_idx])
#    print('Average per t:', ave_marker_per_t[last_idx])

    if get_index:
        keys = list(ave_marker_per_t.keys())
        first_part = int(len(ave_marker_per_t) / 8)
        keys_beginning = keys[0:first_part]
        keys_end = keys[first_part:-1]
        closest_key_beginning = min(keys_beginning, key=lambda k: abs(ave_marker_per_t[k] - max_val))
        closest_key_end = min(keys_end, key=lambda k: abs(ave_marker_per_t[k] - max_val))
        print('break index:', first_part)
        print(f"Beginning for {max_val}: ave_marker_per_t = {ave_marker_per_t[closest_key_beginning]}, ave_marker_per_t_subsyst = {list(ave_marker_per_t_subsyst[closest_key_beginning].values())}, all_markers = {list(all_markers[closest_key_beginning].values())} (key: {closest_key_beginning})")
        print(f"End for {max_val}: ave_marker_per_t = {ave_marker_per_t[closest_key_end]}, ave_marker_per_t_subsyst = {list(ave_marker_per_t_subsyst[closest_key_end].values())}, all_markers = {list(all_markers[closest_key_end].values())} (key: {closest_key_end})")

        data_beginning = all_markers[closest_key_beginning]

        for key, values in data_beginning.items():
            y = key
            clean_y = (float(y[0]), y[1])
            original_indices = list(range(len(values)))
            slice = 1
            x = values[slice:-slice]
            sliced_indices = original_indices[slice:-slice]
            plt.scatter(x, [str(clean_y)] * len(x), marker='o', edgecolors='green', facecolor='none')
            for i, (x_val, idx) in enumerate(zip(x, sliced_indices)):
                plt.text(x_val, str(clean_y), str(idx), ha='center', va='bottom', fontsize=8)
        plt.xlabel('Topological local marker')
        plt.xlim([-3, 2.5])
        plt.axvline(x=ave_marker_per_t[closest_key_beginning], color='red', linestyle='--', linewidth=1)
        plt.ylabel('Subsystem koordinates', labelpad=0.5)
        plt.title(f"Average topological local marker at time {np.round(loaded_times[closest_key_beginning], 3)} with value {np.round(ave_marker_per_t[closest_key_beginning], 2)}")
        plt.show()

        data_end = all_markers[closest_key_end]

        for key, values in data_end.items():
            y = key
            clean_y = (float(y[0]), y[1])
            original_indices = list(range(len(values)))
            slice = 1
            x = values[slice:-slice]
            sliced_indices = original_indices[slice:-slice]
            plt.scatter(x, [str(clean_y)] * len(x), marker='o', edgecolors='green', facecolor='none')
            for i, (x_val, idx) in enumerate(zip(x, sliced_indices)):
                plt.text(x_val, str(clean_y), str(idx), ha='center', va='bottom', fontsize=8)
        plt.xlabel('Topological local marker')
        plt.xlim([-3, 2.5])
        plt.axvline(x=ave_marker_per_t[closest_key_end], color='red', linestyle='--', linewidth=1)
        plt.ylabel('Subsystem koordinates', labelpad=0.5)
        plt.title(f"Average topological local marker at time {np.round(loaded_times[closest_key_end], 3)} with value {np.round(ave_marker_per_t[closest_key_end], 2)}")
        plt.show()

    #print('OPDM_eigvals', OPDM_eigvals)
    if OPDM_eigvals:
        times = {idx: np.round(loaded_times[idx], 3) for idx in ave_marker_per_t}

        for idx, inner_dict in OPDM_eigvals.items():
            for key, values in inner_dict.items():
                ssyst = (5.5, 5)
                if key == ssyst:
                    y = values
                    plt.scatter([times[idx]] * len(y), y, marker='.', edgecolors='green', facecolor='none', linewidths=0.5, s=1)
                    plt.xlabel('t')
                    plt.ylabel('OPDM eigenvalues', labelpad=0.5)
                    plt.title(f"OPDM eigenvalues for subsystem with coordinates {ssyst}")
        plt.show()

    return ave_marker_per_t, var_marker_per_t


def calc_ave_marker_per_t_from_opdm(opdm_t, loaded_times):
    ave_marker_per_t = {}
    var_marker_per_t = {}

    for idx, opdm in enumerate(opdm_t):
        marker, _ = calc_marker_state_from_opdm(opdm, flatten=False, p_eigvals=False)
        bulk_marker = marker[1:-1] if len(marker) > 2 else marker
        ave_marker_per_t[idx] = np.average(bulk_marker)
        var_marker_per_t[idx] = np.var(bulk_marker)

    return ave_marker_per_t, var_marker_per_t

def calc_opdm_from_rho(many_body_rho):
    """
    Calculates the one-particle-density matrix from a local many-body rho.

    rho_opdm =  [[ Tr(rho_ij @ sigma^+_i @ sigma^-_j) , Tr(rho_ij @ sigma^-_i @ sigma^-_j)],
                 [ Tr(rho_ij @ sigma^+_i @ sigma^+_j),  Tr(rho_ij @ sigma^+_j @ sigma^-_i)]]

    Parameters
    ----------
    rho : numpy arrary
        The state for which we want the opdm.

    Returns
    -------
    rho_opdm : numpy array 2L x 2L
        the opdm.
    """
    L_loc = int(np.log2(len(many_body_rho)))

    rho_opdm = np.zeros((2 * L_loc, 2 * L_loc), dtype=np.complex128)

    #print('L_loc', L_loc)
    for i, j in itertools.product(range(L_loc), range(L_loc)):
        rho = calc_opdm_operator(L_loc, i, j)
        hopp_matrix = many_body_rho @ rho['eh'][i, j]
        rho_opdm[i, j] = np.trace(hopp_matrix)  # 1st quadrant: sigma_plus_i sigma_minus_j
        pair_matrix = many_body_rho @ rho['hh'][i, j]
        rho_opdm[i+ L_loc, j ] = np.trace(pair_matrix)  # 3rd quadrant: sigma_plus_i sigma_plus_j

    rho_opdm[: L_loc, L_loc : 2 * L_loc] = rho_opdm[L_loc : 2 * L_loc, : L_loc].T.conjugate()  # 2nd quadrant: sigma_plus_j sigma_plus_i
    rho_opdm[L_loc: 2 * L_loc, L_loc: 2 * L_loc] = np.eye(L_loc) - rho_opdm[:L_loc, :L_loc].conjugate() # 4th quadrant: sigma_minus_j sigma_plus_i

    return rho_opdm

def plot_local_marker(opdm_t, times):
    ave_marker_per_t, _ = calc_ave_marker_per_t_from_opdm(opdm_t, times)
    plot_times = {idx: np.round(times[idx], 3) for idx in ave_marker_per_t}

    fig, ax1 = plt.subplots()

    ax1.scatter(list(plot_times.values()), list(ave_marker_per_t.values()), marker='.', edgecolors='blue', facecolor='none',
                label='Average')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\sum_{i=2}^{L-1} m_i \,/\, (L-2)$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    #ax1.set_ylim([0,1.1])

    # Title
    plt.title('Average Local Topological Marker over time')

    # Layout adjustment
    fig.tight_layout()

    plt.savefig("/Users/yuliyabilinskaya/Desktop/topo_marker.pdf", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_local_marker_comparison(opdm_t_lindblad, times_lindblad, opdm_t_corr, times_corr,
                                 lindblad_label='Lindblad evolution', corr_label='Correlation matrix evolution',
                                 save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
                                 use_lines=True, truncate_to_common_times=False):
    ave_marker_lindblad, _ = calc_ave_marker_per_t_from_opdm(opdm_t_lindblad, times_lindblad)
    ave_marker_corr, _ = calc_ave_marker_per_t_from_opdm(opdm_t_corr, times_corr)

    lindblad_times = np.array([times_lindblad[idx] for idx in ave_marker_lindblad], dtype=float)
    corr_times = np.array([times_corr[idx] for idx in ave_marker_corr], dtype=float)
    lindblad_values = np.array(list(ave_marker_lindblad.values()), dtype=float)
    corr_values = np.array(list(ave_marker_corr.values()), dtype=float)

    if truncate_to_common_times and len(lindblad_times) > 0 and len(corr_times) > 0:
        t_max_common = min(lindblad_times[-1], corr_times[-1])
        lindblad_mask = lindblad_times <= t_max_common
        corr_mask = corr_times <= t_max_common
        lindblad_times = lindblad_times[lindblad_mask]
        lindblad_values = lindblad_values[lindblad_mask]
        corr_times = corr_times[corr_mask]
        corr_values = corr_values[corr_mask]

    fig, ax1 = plt.subplots()

    if use_lines:
        ax1.plot(lindblad_times, lindblad_values, color='blue', linewidth=1.0)
        ax1.plot(corr_times, corr_values, color='red', linewidth=1.0)

    ax1.scatter(lindblad_times, lindblad_values, marker='o', edgecolors='blue', facecolor='none',
                label=lindblad_label)
    ax1.scatter(corr_times, corr_values, marker='x', color='red',linewidths=0.5, s=10,
                label=corr_label)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\sum_{i=2}^{L-1} m_i \,/\, (L-2)$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend()
    plt.title('Average Local Topological Marker over time')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return lindblad_times, lindblad_values, corr_times, corr_values
