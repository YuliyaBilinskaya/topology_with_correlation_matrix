#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:26:54 2023

@author: claudia
@edited_by: yuliya

"""
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import eigh, eigvalsh
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from scipy.interpolate import PchipInterpolator
from matplotlib import colorbar
from matplotlib.cm import ScalarMappable



######################### calculations from psi ############################


def reshape_psi(psi, n, l):
    '''

    Parameters
    ----------
    psi : numpy array
        Vector of size 2**L where L is the number of sites.
    n : INT
        the leftmost site of the subset of interest A.
    l : INT
        The number of sites in the subset of interest A.

    Returns
    -------
    psi: numpy array (2**l,2**(L-l))
        Reshaped psi in a matrix psi_{(A),(\bar{A}) with (A) the combined
        indices in A, and similar for \bar{A}

    '''

    L = int(np.log2(psi.size))
    # psi = np.transpose(np.reshape(psi,L*[2]))     # transpose since the binary representation is in opposite order in the basis
    psi = np.reshape(psi, L * [2])
    psi = np.moveaxis(psi, list(range(n, n + l)), list(range(0, l)))
    psi = np.reshape(psi, (2 ** l, 2 ** (L - l)))

    return psi

# np.linalg.svd does singular value decomposition of the matrix such that matrix=u * np.diag(s) * v, where u and v unitary
#    the result gives the diagonal matrix with eigvals on the diagonal.
# Calculates entropy of a given subsystem


def calc_entropies(psi):
    '''

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wave function.

    Returns
    -------
    SvN : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l

    '''

    L = int(np.log2(psi.size))
    SvN = {}

    for l in range(1, L + 1):   #Looping over all possible subsystem sizes
        SvN[l] = []
        for n in range(L - l + 1): #The leftmost element of the subsystem
            #print("l",l)
            #print("n",n)
            psi_r = reshape_psi(psi, n, l)
            SvN[l].append(S_vN(psi_r))
        SvN[l] = np.array(SvN[l])
    #print("SvN",SvN)
    return SvN


def S_vN(psi):
    '''

    Parameters
    ----------
    psi : np.array 2**Na x 2**Nb
        The wave function. Assumed to have been reshpaed into a matrix with
        the correct subspace dimensions

    Returns
    -------
    S : FLOAT
        the von Neumann entanglement entropy

    '''

    sv = np.linalg.svd(psi, compute_uv=False)
    sv = sv[sv > 1e-16]
    sv = sv ** 2
    #print("sv", sv)
    return -np.sum(sv * np.log2(sv))


def calc_info(psi):
    '''

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''

    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SvN = calc_entropies(psi)

    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - SvN[l]
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt


def partial_trace(rho_vec,n,l):
    # assume rho is a vectorized density matrix (i.e, if it was a pure state it would be rho = psi x psi)
    # we want to trace over the compliment subsystem on the left (w < n) and right (w > n) of the subsystem we are interested in. 
    L = int(np.log2(np.size(rho_vec)))//2
    rho = np.reshape(rho_vec,(2*L)*[2])  
    for w in range(L-l):
        #print('w=',w)
        #print('rho=', np.shape(rho))
        
        if w < n:
            rho = np.trace(rho, axis1=0, axis2=L-w)
        elif w == n or w > n:
            rho = np.trace(rho, axis1=l, axis2=l+L-w)
    dim = 2**(l)
    #print('l=',l)
    #print('n=',n)
    #print('dim_red=', dim)
    #print('rho=', np.reshape(rho,(dim,dim)))
    return np.reshape(rho,(dim,dim))



def S_vN_lndbld(rho):
    '''

    Parameters
    ----------
    rho : np.array 2**Na x 2**Nb
        Reduced density matrix. Assumed to have been traced over the complement subsystem.

    Returns
    -------
    S : FLOAT
        the von Neumann entanglement entropy

    '''

    eigvals, eigvecs = eigh(rho)
    
    #print('eigvals', eigvals)
    
    # Substituting with 1e-22 eigenvalues which happened to be negative due to numerical error
    # or if the eigvenvalue is exactly zero set it to 1e-22 to be able to calculate log2
    eigvals[(eigvals > -1.5e-12) & (eigvals < 1e-30)] = 1e-22
    
    #print('eigvals', eigvals)
    #print('trace_rho', np.sum(eigvals))
    #print('entropy', -np.sum(eigvals * np.log2(eigvals)))
    
    return -np.sum(eigvals * np.log2(eigvals))


######################### calculations from rho_vec ############################


def calc_entropies_lndbld(rho_vec):
    '''

    Parameters
    ----------
    rho_vec : vectorized density matrix: rho(2**L, 2**L) -> rho(4**L, 1).

    Returns
    -------
    SvN_lndbld : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l for time evolution with Lindbladian

    '''

    L = int(np.log2(rho_vec.size)) // 2
    SvN = {}

    for l in range(1, L + 1):  # Looping over all possible subsystem sizes

        SvN[l] = []
        for n in range(L - l + 1):  # The leftmost element of the subsystem
            # print("l",l)
            # print("n",n)
            rho_red = partial_trace(rho_vec, n, l)
            SvN[l].append(S_vN_lndbld(rho_red))
        SvN[l] = np.array(SvN[l])
    # print('SvN=',SvN)
    return SvN


def calc_info_lndbld(rho_vec):
    '''

    Parameters
    ----------
    rho : Numpy Array 4**L
        Vectorized density matrix.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''
    # Calculating information by I(ρAuB ) − I (ρA) − I (ρB ) + I (ρA∩B )

    # assert len(rho_vec.shape) == 1
    # assert rho_vec.shape[1] == 1
    L = int(np.log2(rho_vec.size)) // 2
    SvN = calc_entropies_lndbld(rho_vec)

    info_latt = {}
    for l in range(1, L + 1):
        # print('l=',l)
        if l == 1:
            info_latt[l] = l - SvN[l]
            # print('SvN[l]', SvN[l])
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
            # print('SvN[l]', SvN[l])
            # print('SvN[l - 1][:-1]', SvN[l - 1][:-1])
            # print('SvN[l - 1][1:]', SvN[l - 1][1:])

            # print('SvN', SvN)
        else:
            # print('SvN[l]', SvN[l])
            # print('SvN[l - 1][:-1]', SvN[l - 1][:-1])
            # print('SvN[l - 1][1:]', SvN[l - 1][1:])
            # print('SvN[l - 2][1:-1]', SvN[l - 2][1:-1])
            # print('SvN[l - 2]', SvN[l - 2])
            # print('SvN', SvN)
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt


######################### calculations from correlation matrix ############################




def partial_trace_corr_mat(corr_mat, n, l):
    corr_mat = np.asarray(corr_mat)

    L = corr_mat.shape[0] // 2
    reduced_corr_mat = np.zeros((2 * l, 2 * l), dtype=corr_mat.dtype)
    reduced_corr_mat[:l, :l] = corr_mat[n:n + l, n:n + l]
    reduced_corr_mat[:l, l:] = corr_mat[n:n + l, L + n:L + n + l]
    reduced_corr_mat[l:, :l] = corr_mat[L + n:L + n + l, n:n + l]
    reduced_corr_mat[l:, l:] = corr_mat[L + n:L + n + l, L + n:L + n + l]

    return reduced_corr_mat


def s_vN_corr_matrix(corr_mat):
    eigvals = eigvalsh(corr_mat)

    if not np.isfinite(corr_mat).all():
        raise ValueError("Generalized correlation matrix contains NaN or inf")

    tol = 1e-10

    if not np.allclose(corr_mat, corr_mat.conj().T, atol=tol, rtol=0):
        raise ValueError("Generalized correlation matrix is not Hermitian")

    #if np.min(eigvals) < -tol or np.max(eigvals) > 1 + tol:
    #    raise ValueError(
    #        f"Generalized correlation matrix has eigenvalues outside [0, 1]: "
    #        f"[{np.min(eigvals)}, {np.max(eigvals)}]"
    #     )

    #print('eigvals', eigvals)
    # print('trace_rho', np.sum(eigvals))
    # print('entropy', -np.sum(eigvals * np.log2(eigvals)))

    mask = (eigvals > 1e-12) & (eigvals < 1 - 1e-12)
    eigvals = eigvals[mask]

    #if using only C_mk block: -np.sum(eigvals * np.log2(eigvals) + (1 - eigvals) * np.log2(1 - eigvals))

    return -np.sum(eigvals * np.log2(eigvals))



def calc_entropies_corr_mat(corr_mat):
    '''

    Parameters
    ----------
    rho_vec : vectorized density matrix: rho(2**L, 2**L) -> rho(4**L, 1).

    Returns
    -------
    SvN_lndbld : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l for time evolution with Lindbladian

    '''

    L = int(len(corr_mat) // 2)
    SvN = {}

    for l in range(1, L + 1):  # Looping over all possible subsystem sizes

        SvN[l] = []
        for n in range(L - l + 1):  # The leftmost element of the subsystem
            # print("l",l)
            # print("n",n)
            reduced_corr_mat = partial_trace_corr_mat(corr_mat, n, l)
            SvN[l].append(s_vN_corr_matrix(reduced_corr_mat))
        SvN[l] = np.array(SvN[l])
    # print('SvN=',SvN)
    return SvN


def calc_info_from_corr_matr(corr_mat):
    '''

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''

    L = int(len(corr_mat) // 2)
    SvN = calc_entropies_corr_mat(corr_mat)

    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - SvN[l]
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt



######################### plotting ############################

def plot_info_latt(rho_vec, L, InfoLatt_norm, time):
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 5), dpi=400)


    color_map = plt.get_cmap("Oranges")
    colors = color_map(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]  # sets the firs color to white
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLatt_norm)
    # print('info_latt', info_latt)
    r = 1 / (4 * L)
    # fig, ax = plt.subplots(dpi=300)
    info_latt =  calc_info_lndbld(rho_vec)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            value = info_latt[l][x]
            ax1.add_artist(
                plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, facecolor=custom_cmap(norm(value)),
                           edgecolor='black', linewidth=0.2))
    ax1.set_xlim([-2 * r, 1])
    # YB:
    x_ticks = []
    y_ticks = []
    for x in range(len(info_latt[1])):
        x_ticks.append(x / L + 1 / (2 * L))
        y_ticks.append(x / L + 1 / (2 * L))
    ax1.set_xticks(x_ticks, range(1, L + 1))
    ax1.set_yticks(y_ticks, range(0, L))
    ax1.set_xlabel('Site', fontsize=10)
    ax1.set_ylabel('$\ell$', fontsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_ylim([-2 * r, 1 + 2 * r])


    ax1.set_title(f'Info lattice at time t={time}', fontsize=10)


    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    bbox = ax1.get_position()

    # define gap and width for the colorbar
    gap = 0.01  # horizontal space between the axis and the colorbar
    cbar_width = 0.02  # width of the colorbar

    # create an axes for the colorbar
    cax = fig1.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.y1 - bbox.y0])

    # create the colorbar in this axes
    cbar = fig1.colorbar(sm, cax=cax, orientation="vertical")

    # label and tick size
    cbar.set_label('$i^\ell_n$')
    cbar.ax.tick_params(labelsize=8)

    plt.savefig("/Users/yuliyabilinskaya/Desktop/topo_diss.pdf", dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig1)


def plot_info_lattice(delta, info_per_scale, N_scales, L, rho_vec, info_latt_t, tot_info_t, tot_info, t, c, dt, t_max, InfoLatt_norm):
    ave_info_per_scale_delta = {}
    dcount = {}

    for k, v in info_per_scale.items():
        for l in range(1, N_scales + 1):
            ave_info_per_scale_delta[(k[0], l)] = ave_info_per_scale_delta.get((k[0], l), 0) + v[l]
            dcount[(k[0], l)] = dcount.get((k[0], l), 0) + 1

    for k, v in ave_info_per_scale_delta.items():
        ave_info_per_scale_delta[k] /= dcount[k]

    # print("ave_info_per_scale_delta", ave_info_per_scale_delta)

    # print('Time elapsed: {:.2e} s'.format(time.time() - start_time))

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=400)

    scales_t0 = np.arange(N_scales)
    ave_t0 = info_latt_t[0, :N_scales]
    interp_func_t0 = PchipInterpolator(scales_t0, ave_t0)
    interp_scales_t0 = np.arange(scales_t0[0], scales_t0[-1] + 0.1, 0.1)
    interp_average_t0 = interp_func_t0(interp_scales_t0)
    marker_ini = 'x'

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
        marker_fin = '4'
        ax1.plot(interp_scales_t0, interp_average_t0, color='black', label='Initial state')
        ax1.plot(scales_t0, ave_t0, marker=marker_ini, linestyle='None', markersize=8, color='deepskyblue',
                 label='Initial state')
        ax1.plot(interp_scales, interp_average, color='black', label='Final state')
        ax1.plot(scales, ave, marker=marker_fin, linestyle='None', markersize=10, color='blue', label='Final state')
        # for count_datum, datum in enumerate(dataPBC[key]):
        # plt.plot(scales, datum, color=colors(count_datum))
    ax1.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
    ax1.set_ylim([-0.1, L + 0.1])
    ax1.set_xlabel(r"$\ell$", fontsize=16)
    ax1.set_xticks(np.arange(len(scales)))
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    # ax1.set_aspect('equal', adjustable='box')
    legend_elements = [
        Line2D([0], [0], color='black', marker=marker_ini, markeredgecolor='deepskyblue', markersize=8,
               linestyle='-', label='Initial state'),
        Line2D([0], [0], color='black', marker=marker_fin, markeredgecolor='blue', markersize=10, linestyle='-',
               label='Final state'),
    ]
    ax1.legend(handles=legend_elements, fontsize=14)
    ax1.text(0.07, 0.97, '({})'.format(chr(97)), transform=ax1.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')
    # plotting the circle diagram
    color_map = plt.get_cmap("Oranges")
    colors = color_map(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]  # sets the firs color to white
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLatt_norm)
    info_latt = calc_info_lndbld(rho_vec)
    # print('info_latt', info_latt)
    r = 1 / (4 * L)
    # fig, ax = plt.subplots(dpi=300)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            value = info_latt[l][x]
            ax2.add_artist(
                plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, facecolor=custom_cmap(norm(value)),
                           edgecolor='black', linewidth=0.2))
    ax2.set_xlim([-2 * r, 1])
    # YB:
    x_ticks = []
    y_ticks = []
    for x in range(len(info_latt[1])):
        #print("x", x)
        x_ticks.append(x / L + 1 / (2 * L))
        y_ticks.append(x / L + 1 / (2 * L))
    ax2.set_xticks(x_ticks, range(1, L + 1))
    ax2.set_yticks(y_ticks, range(0, L))
    ax2.set_xlabel('Site', fontsize=16)
    ax2.set_ylabel('$\ell$', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_ylim([-2 * r, 1 + 2 * r])
    ax2.text(0.07, 0.97, '({})'.format(chr(97 + 1)), transform=ax2.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')

    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    bbox = ax3.get_position()
    gap = 0.01
    cbar_width = 0.02
    cax = fig1.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.y1 - bbox.y0])
    cbar = fig1.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label('$i^\\ell_n$')
    cbar.ax.tick_params(labelsize=8)

    info_latt_t[int(c), :] = ave
    x_linspace = np.arange(0, t_max, dt)
    tot_info_t[int(c)] = tot_info
    markermap = ['o', 'o', '^', '^', 'v', '^', '>', '<', 'v', '^']
    #for w in [1, L - 1]:
        #ax3.plot(x_linspace, info_latt_t[:, w], marker=markermap[w], markerfacecolor='None', color='black', linewidth=1, label=(r"$\mathscr{i}^%d$" % (w)))
    ax3.plot(x_linspace, tot_info_t, linestyle='--', color='green', label=("$I_{tot}$"))
    ax3.legend(fontsize=14)
    ax3.set_xlabel('t', fontsize=16)
    #        ax3.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18, labelpad=1)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set_ylim([0, L ])
    ax3.text(0.07, 0.97, '({})'.format(chr(97 + 2)), transform=ax3.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')
    #plt.savefig(save_dir + ".pdf", dpi=200, bbox_inches='tight')
    plt.savefig("/Users/yuliyabilinskaya/Desktop/topo_diss.pdf", dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig1)

def plot_info_latt_from_corr(corr_mat, L, InfoLatt_norm, time):
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 5), dpi=400)


    color_map = plt.get_cmap("Oranges")
    colors = color_map(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]  # sets the firs color to white
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLatt_norm)
    # print('info_latt', info_latt)
    r = 1 / (4 * L)
    # fig, ax = plt.subplots(dpi=300)
    info_latt =  calc_info_from_corr_matr(corr_mat)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            value = info_latt[l][x]
            ax1.add_artist(
                plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, facecolor=custom_cmap(norm(value)),
                           edgecolor='black', linewidth=0.2))
    ax1.set_xlim([-2 * r, 1])
    # YB:
    x_ticks = []
    y_ticks = []
    for x in range(len(info_latt[1])):
        x_ticks.append(x / L + 1 / (2 * L))
        y_ticks.append(x / L + 1 / (2 * L))
    ax1.set_xticks(x_ticks, range(1, L + 1))
    ax1.set_yticks(y_ticks, range(0, L))
    ax1.set_xlabel('Site', fontsize=10)
    ax1.set_ylabel('$\ell$', fontsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_ylim([-2 * r, 1 + 2 * r])


    ax1.set_title(f'Info lattice at time t={time}', fontsize=10)


    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    bbox = ax1.get_position()

    # define gap and width for the colorbar
    gap = 0.01  # horizontal space between the axis and the colorbar
    cbar_width = 0.02  # width of the colorbar

    # create an axes for the colorbar
    cax = fig1.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.y1 - bbox.y0])

    # create the colorbar in this axes
    cbar = fig1.colorbar(sm, cax=cax, orientation="vertical")

    # label and tick size
    cbar.set_label('$i^\ell_n$')
    cbar.ax.tick_params(labelsize=8)

    #plt.savefig(f"/Users/yuliyabilinskaya/Desktop/corr_info_latt.pdf", dpi=200, bbox_inches='tight')

    #plt.savefig(save_dir + ".pdf", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig1)


def plot_info_lattice_from_corr(delta, info_per_scale, N_scales, L, corr_mat, info_latt_t, tot_info_t, tot_info, t, c, dt, t_max, InfoLatt_norm):
    ave_info_per_scale_delta = {}
    dcount = {}

    for k, v in info_per_scale.items():
        for l in range(1, N_scales + 1):
            ave_info_per_scale_delta[(k[0], l)] = ave_info_per_scale_delta.get((k[0], l), 0) + v[l]
            dcount[(k[0], l)] = dcount.get((k[0], l), 0) + 1

    for k, v in ave_info_per_scale_delta.items():
        ave_info_per_scale_delta[k] /= dcount[k]

    # print("ave_info_per_scale_delta", ave_info_per_scale_delta)

    # print('Time elapsed: {:.2e} s'.format(time.time() - start_time))

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=400)

    scales_t0 = np.arange(N_scales)
    ave_t0 = info_latt_t[0, :N_scales]
    interp_func_t0 = PchipInterpolator(scales_t0, ave_t0)
    interp_scales_t0 = np.arange(scales_t0[0], scales_t0[-1] + 0.1, 0.1)
    interp_average_t0 = interp_func_t0(interp_scales_t0)
    marker_ini = 'x'

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
        marker_fin = '4'
        ax1.plot(interp_scales_t0, interp_average_t0, color='black', label='Initial state')
        ax1.plot(scales_t0, ave_t0, marker=marker_ini, linestyle='None', markersize=8, color='deepskyblue',
                 label='Initial state')
        ax1.plot(interp_scales, interp_average, color='black', label='Final state')
        ax1.plot(scales, ave, marker=marker_fin, linestyle='None', markersize=10, color='blue', label='Final state')
        # for count_datum, datum in enumerate(dataPBC[key]):
        # plt.plot(scales, datum, color=colors(count_datum))
    ax1.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18)
    ax1.set_ylim([-0.1, L + 0.1])
    ax1.set_xlabel(r"$\ell$", fontsize=16)
    ax1.set_xticks(np.arange(len(scales)))
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    # ax1.set_aspect('equal', adjustable='box')
    legend_elements = [
        Line2D([0], [0], color='black', marker=marker_ini, markeredgecolor='deepskyblue', markersize=8,
               linestyle='-', label='Initial state'),
        Line2D([0], [0], color='black', marker=marker_fin, markeredgecolor='blue', markersize=10, linestyle='-',
               label='Final state'),
    ]
    ax1.legend(handles=legend_elements, fontsize=14)
    ax1.text(0.07, 0.97, '({})'.format(chr(97)), transform=ax1.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')
    # plotting the circle diagram
    color_map = plt.get_cmap("Oranges")
    colors = color_map(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]  # sets the firs color to white
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLatt_norm)
    info_latt = calc_info_from_corr_matr(corr_mat)
    # print('info_latt', info_latt)
    r = 1 / (4 * L)
    # fig, ax = plt.subplots(dpi=300)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            value = info_latt[l][x]
            ax2.add_artist(
                plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, facecolor=custom_cmap(norm(value)),
                           edgecolor='black', linewidth=0.2))
    ax2.set_xlim([-2 * r, 1])
    # YB:
    x_ticks = []
    y_ticks = []
    for x in range(len(info_latt[1])):
        x_ticks.append(x / L + 1 / (2 * L))
        y_ticks.append(x / L + 1 / (2 * L))
    ax2.set_xticks(x_ticks, range(1, L + 1))
    ax2.set_yticks(y_ticks, range(0, L))
    ax2.set_xlabel('Site', fontsize=16)
    ax2.set_ylabel('$\ell$', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_ylim([-2 * r, 1 + 2 * r])
    ax2.text(0.07, 0.97, '({})'.format(chr(97 + 1)), transform=ax2.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')

    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    bbox = ax3.get_position()
    gap = 0.01
    cbar_width = 0.02
    cax = fig1.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.y1 - bbox.y0])
    cbar = fig1.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label('$i^\\ell_n$')
    cbar.ax.tick_params(labelsize=8)

    info_latt_t[int(c), :] = ave
    x_linspace = np.arange(0, t_max, dt)
    tot_info_t[int(c)] = tot_info
    markermap = ['o', 'o', '^', '^', 'v', '^', '>', '<', 'v', '^']
    #for w in [1, L - 1]:
        #ax3.plot(x_linspace, info_latt_t[:, w], marker=markermap[w], markerfacecolor='None', color='black', linewidth=1, label=(r"$\mathscr{i}^%d$" % (w)))
    ax3.plot(x_linspace, tot_info_t, linestyle='--', color='green', label=("$I_{tot}$"))
    ax3.legend(fontsize=14)
    ax3.set_xlabel('t', fontsize=16)
    #        ax3.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=18, labelpad=1)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set_ylim([0, L])
    ax3.text(0.07, 0.97, '({})'.format(chr(97 + 2)), transform=ax3.transAxes, fontsize=14, verticalalignment='top',
             weight='bold')
    plt.savefig("/Users/yuliyabilinskaya/Desktop/topo_diss.pdf", dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig1)


######################### miscellaneous ############################

def S_vN_Anderson(arr):
    ent = 0.
    for x in arr:
        if x > 1e-14 and x < 1 - 1e-14:
            ent -= x * np.log2(x)
    return ent


def get_entanglement_spectrum_from_correlation(correlation, subsystem):
    '''

        Parameters
        ----------
        correlation : Numpy Array 2*L x 2*L
            The correlation matrix.

        subsystem : list of physical sites in the subsystem

        Returns
        -------
        SvN : dictionary of np.arrays
            SvN[l] is the set of entropies of subspaces with size l

    '''

    L = int(0.5 * correlation.shape[0])
    l = len(subsystem)
    first_el = subsystem[0]
    last_el = subsystem[-1] + 1
    small_c = np.zeros((2*l, 2*l))
    small_c[0:l, 0:l] = correlation[first_el:last_el, first_el:last_el]
    small_c[0:l, l:] = correlation[first_el:last_el, first_el+L:last_el+L]
    small_c[l:, 0:l] = correlation[first_el+L:last_el+L, first_el:last_el]
    small_c[l:, l:] = correlation[first_el+L:last_el+L, first_el+L:last_el+L]

    entanglement_spectrum, _ = eigh(small_c)
    print("subsystem", subsystem)
    print("entanglement_spectrum", entanglement_spectrum)

    return entanglement_spectrum



def calc_entropies_Anderson(correlation):
    '''

    Parameters
    ----------
    correlation : Numpy Array 2*L x 2*L
        The correlation matrix.

    Returns
    -------
    SvN : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l

    '''

    L = int(0.5 * correlation.shape[0])
    SvN = {}

    for l in range(1, L + 1):
        SvN[l] = []
        for n in range(L - l + 1):
            subsystem = [x for x in range(n, n + l)]
            subsystem = np.array(subsystem)
            SvN[l].append(S_vN_Anderson(get_entanglement_spectrum_from_correlation(correlation, subsystem)))
        SvN[l] = np.array(SvN[l])
    #print("SvN",SvN)
    return SvN




def calc_info_Anderson(correlation):
    '''

    Parameters
    ----------
    correlation : Numpy Array 2*L x 2*L
        The correlation matrix.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''

    L = int(0.5 * correlation.shape[0])
    SvN = calc_entropies_Anderson(correlation)

    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - SvN[l]
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]
    #print("info_latt", info_latt)
    return info_latt


def calc_info_per_scale(info_latt, bc):
    '''


    Parameters
    ----------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    Returns
    -------
    info_per_scale: np.array
        info_per_scale[l] is the info summed over all sites on scale l

    '''

    L = len(info_latt)
    info_per_scale = {}

    if bc == 'periodic':
        for l in range(1, L//2 + 2):
            if l == 1:
                info_per_scale[l] = np.sum(info_latt[l])    
            else:
                if L % 2 == 0 and l == L//2 + 1:            
                    info_per_scale[l] = np.sum(info_latt[l])    
                else:
                    info_per_scale[l] = np.sum(info_latt[l]) + np.sum(info_latt[L-l+2])
    elif bc == 'open':
        for l in info_latt:
            info_per_scale[l] = np.sum(info_latt[l])

    else:
        raise ValueError('boundary condition must be "periodic" or "open"')

    return info_per_scale


def singlet(L):
    assert L%2 == 0
    if L//2 == 1:
        up = np.array([1,0])
        down = np.array([0,1])
        return (np.kron(up,down)-np.kron(down,up))/np.sqrt(2)
    else:
        return np.kron(singlet(L-2),singlet(2))
