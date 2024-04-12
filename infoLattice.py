#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:26:54 2023

@author: claudia
@edited_by: yuliya

"""
import numpy as np
from scipy.linalg import eigh
from scipy.stats import unitary_group
import matplotlib.pyplot as plt



def partial_trace(rho_vec,n,l):
    # assume rho is a vectorized density matrix (i.e, if it was a pure state it would be rho = psi x psi)
    # we want to trace over the compliment subsystem on the left (w < n) and right (w > n) of the subsystem we are interested in. 
    L = int(np.log2(np.size(rho_vec)))//2
    rho = np.reshape(rho_vec,(2*L)*[2])  
    for w in range(L-l):
        print('w=',w)
        #print('rho=', np.shape(rho))
        
        if w < n:
            rho = np.trace(rho, axis1=0, axis2=L-w)
        elif w == n or w > n:
            rho = np.trace(rho, axis1=l, axis2=l+L-w)
    dim = 2**(l)
    print('l=',l)
    print('n=',n)
    print('dim_red=', dim)
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
    
    print('eigvals', eigvals)
    
    # Substituting with 1e-22 eigenvalues which happened to be negative due to numerical error
    # or if the eigvenvalue is exactly zero set it to 1e-22 to be able to calculate log2
    eigvals[(eigvals > -1e-14) & (eigvals < 1e-30)] = 1e-22
    
    print('eigvals', eigvals)
    print('trace_rho', np.sum(eigvals))
    print('entropy', -np.sum(eigvals * np.log2(eigvals)))
    
    return -np.sum(eigvals * np.log2(eigvals))



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

    L = int(np.log2(rho_vec.size))//2
    SvN = {}

    for l in range(1, L + 1):   #Looping over all possible subsystem sizes
        
        SvN[l] = []
        for n in range(L - l + 1): #The leftmost element of the subsystem
            #print("l",l)
            #print("n",n)
            rho_red = partial_trace(rho_vec,n,l)
            SvN[l].append(S_vN_lndbld(rho_red))
        SvN[l] = np.array(SvN[l])
    print('SvN=',SvN)
    return SvN



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


# Calculating information by I(ρAuB ) − I (ρA) − I (ρB ) + I (ρA∩B ) 

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

    #assert len(rho_vec.shape) == 1
    assert rho_vec.shape[1] == 1
    L = int(np.log2(rho_vec.size))//2
    SvN = calc_entropies_lndbld(rho_vec)
    
    info_latt = {}
    for l in range(1, L + 1):
        print('l=',l)
        if l == 1:
            info_latt[l] = l - SvN[l]
            print('SvN[l]', SvN[l])
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
            print('SvN[l]', SvN[l])
            print('SvN[l - 1][:-1]', SvN[l - 1][:-1])
            print('SvN[l - 1][1:]', SvN[l - 1][1:])

            print('SvN', SvN)
        else:
            print('SvN[l]', SvN[l])
            print('SvN[l - 1][:-1]', SvN[l - 1][:-1])
            print('SvN[l - 1][1:]', SvN[l - 1][1:])
            print('SvN[l - 2][1:-1]', SvN[l - 2][1:-1])
            print('SvN[l - 2]', SvN[l - 2])
            print('SvN', SvN)
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt


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


def plot_info_latt(info_latt):
    color_map = plt.get_cmap("Oranges")
    L = max(info_latt.keys())
    print("info_latt.keys()", info_latt.keys())
    print("tuple", L)
    r = 1/(4*L)
    fig, ax = plt.subplots(dpi=300)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            ax.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, color=color_map(info_latt[l][x])))
    plt.xlim([-2*r,1])
    #YB:
    print("range(len(info_latt[1]))", range(len(info_latt[1])))
    x_ticks = []
    y_ticks = []
    for x in range(len(info_latt[1])):
        print("x",x)
        x_ticks.append(x/L+1/(2*L))
        y_ticks.append(x/L+1/(2*L))

    
    plt.xticks(x_ticks, range(1,L+1))
    plt.yticks(y_ticks, range(0,L))
    
    plt.xlabel('Sites')
    plt.ylabel('Levels (l)')
    
    plt.ylim([-2*r,1+2*r])
    ax.set_aspect('equal')
    #plt.axis('off')
    plt.show()


def singlet(L):
    assert L%2 == 0
    if L//2 == 1:
        up = np.array([1,0])
        down = np.array([0,1])
        return (np.kron(up,down)-np.kron(down,up))/np.sqrt(2)
    else:
        return np.kron(singlet(L-2),singlet(2))
