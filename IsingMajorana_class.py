#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:48:54 2022

@author: jensba

@updated_by: yubi
"""
import numpy as np
from dataclasses import dataclass
from scipy import sparse
from scipy.stats import unitary_group
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# YB: bin() converts to binary numbers, zfill makes a string of zeros

def bin_to_n(s, L):
    """
    Given an integer s representing a basis state in an L site system,
    return the occupation number representation as an array.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    s : integer
    L : number of sites

    Returns
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1

    Example
    ----------
    # bin_to_n(10,4)
    array([0, 1, 0, 1])

    """

    return np.array([int(x) for x in reversed(bin(s)[2:].zfill(L))])

#YB: int( ,2) gives integers in base 2

def n_to_bin(n):
    """
    Given an occupation number representation of a state n,
    return the integer corresponding to its binary representation.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1

    Returns
    ----------
    s : integer

    Example
    ----------
    # n_to_bin([0,1,0,1])
    10

    """

    return int(''.join(str(x) for x in reversed(n)), 2)



def onsite_field(n, i):
    """
    Calculates the phase alpha = +- 1 where
    (2 n_i - 1) |a> = \alpha |a>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L-1}]
    i : int

    Returns
    -------
    alpha for any state |a>

    """
    if n[i] == 1:
        return 1
    else:
        return -1


def hopping(n, i, j):
    """
    Calculates the binary r, and phase alpha where
    c^{\dagger}_i c_j |a> = \alpha |r>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L-1}].
    i : int
    j : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """
    
    #print('i=', i)
    #print('j=', j)
    #print('binary=', n)
    if i == j and n[i] == 1: 
        a = n_to_bin(n) 
        #print('exponent', 1)
        #print('a', bin_to_n(a, L))
        return a, 1
    elif n[i] == 0 and n[j] == 1:
        a = n_to_bin(n) 
        exponent = np.sum(n[min(i, j) + 1: max(i, j)]) 
        #print('exponent', exponent)
        #print('a', bin_to_n(a + 2 ** i - 2 ** j, L))
        return a + 2 ** i - 2 ** j, (-1) ** exponent 
    else:
        return None


def pairing(n, i, j):
    """
    Calculates the binary r, and phase alpha where
    c_i c_j |a> = \alpha |r>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L-1}].
    i : int
    j : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """
    if i != j and (n[i] == 1 and n[j] == 1):
        a = n_to_bin(n)
        exponent = np.sum(n[min(i, j):max(i, j)]) 
        #print('i=', i)
        #print('j=', j)
        #print('n starting state:', n)
        #print('hopping exponent', exponent)
        #print('new state:', bin_to_n(a - 2 ** i - 2 ** j, 6))
        return a - 2 ** i - 2 ** j, np.sign(i - j) * (-1) ** exponent 
    else:
        return None

# this bath acts on each pair of sites of the chain
def jump_TBD(n, i, j, theta, L):
    """
    Jump term of the form c^+_i + c^+_i+1 - c_i+1 + c_i = c^+_i + c^+_j - c_j + c_i
    
    Calculates the binary r, and phase alpha where
    (c^+_i + c^+_j - c_j + c_i) |a> = \alpha_i |r_i>,  \alpha_j |r_j>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L-1}].
    i : int
    j : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """

    if i != j:
        #print('i=', i)
        #print('j=', j)
        #print('binary=', n)
        
        if n[i] == 1:
            a = n_to_bin(n)
            exponent_i = np.sum(n[0:i])    
            #print('exponent_i', exponent_i)
            a_i =  a - 2 ** i
            #print('a_i', bin_to_n(a_i, L))
            #print('a_i', a_i)
            phase_i = (-1) ** exponent_i * np.cos(theta)
        
        elif n[i] == 0:
            a = n_to_bin(n)
            exponent_i = np.sum(n[0:i])            
            #print('exponent_i', exponent_i)
            a_i =  a + 2 ** i 
            #print('a_i', bin_to_n(a_i, L))
            #print('a_i', a_i)
            phase_i = (-1) ** exponent_i * np.sin(theta)
    
        
        if n[j] == 1:
            a = n_to_bin(n)
            exponent_j = np.sum(n[0:j])      
            #print('exponent_j', exponent_j)
            a_j =  a - 2 ** j
            #print('a_j', bin_to_n(a_j, L))
            #print('a_j', a_j)
            phase_j = - (-1) ** (exponent_j) * np.sin(theta)       #adding a minus in front of a_i+1 as in Topol Quantum wire paper (DOI: 10.1038/NPHYS2106)
        
        elif n[j] == 0:
            a = n_to_bin(n)
            exponent_j = np.sum(n[0:j]) 
            #print('exponent_j', exponent_j)
            a_j =  a + 2 ** j
            #print('a_j', bin_to_n(a_j, L))
            #print('a_j', a_j)
            phase_j =  (-1) ** (exponent_j) * np.cos(theta)
        else:
            return None  
        
        return a_i, phase_i, a_j, phase_j

        


def jump_TBD_prod(n, i, j, L):
    """
    Jump term of the form (c^+_i + c^+_i+1) (- c_i+1 + c_i) = (c^+_i + c^+_j) (- c_j + c_i) = 
    
    = (c^+_i c_i + c^+_j c_i - c^+_i c_j - c^+_j c_j)
    
    Calculates the binary r, and phase alpha.
    
    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L-1}].
    i : int
    j : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """

    #print('i=', i)
    #print('j=', j)
    #print('binary=', n)
    
    if n[i] == 1 and n[j] == 1:
        a = n_to_bin(n)
        
        # c^+_i c_i - c^+_j c_j term
        a_i = a                         # c^+_i c_i term
        a_j = a                         # c^+_j c_j term
        phase_i = 1                     # c^+_i c_i term
        phase_j = -1                    # - c^+_j c_j term
        #print('a_i', bin_to_n(a_i, L))
        #print('binary a_i', a_i)
        #print('phase_i', phase_i)
        #print('a_j', bin_to_n(a_j, L))
        #print('binary a_j', a_j)
        #print('phase_j', phase_j)
    
    elif n[i] == 1 and n[j] == 0:
        a = n_to_bin(n)
        
        # c^+_j c_i + c^+_i c_i term
        exponent_j = np.sum(n[min(i, j) + 1: max(i, j)])
        #print('exponent_i', exponent_i)
        a_j = a - 2 ** i + 2 ** j       # c^+_j c_i term
        a_i = a                         # c^+_i c_i term
        phase_i = 1                     # c^+_i c_i term
        phase_j = (-1) ** (exponent_j)  # c^+_j c_i term
        #print('a_i', bin_to_n(a_i, L))
        #print('binary a_i', a_i)
        #print('phase_i', phase_i)
        #print('a_j', bin_to_n(a_j, L))
        #print('binary a_j', a_j)
        #print('phase_j', phase_j)

    
    
    elif n[i] == 0 and n[j] == 1:
        a = n_to_bin(n)
        
        # - c^+_i c_j - c^+_j c_j term
        exponent_i = np.sum(n[min(i, j) + 1: max(i, j)])
        #print('exponent_j', exponent_j)
        a_i =  a - 2 ** j + 2 ** i      # c^+_i c_j term
        a_j =  a                        # c^+_j c_j term      
        phase_i = - (-1) ** (exponent_i)# - c^+_i c_j term
        phase_j = - 1                   # - c^+_j c_j term
        #print('a_i', bin_to_n(a_i, L))
        #print('binary a_i', a_i)
        #print('phase_i', phase_i)
        #print('a_j', bin_to_n(a_j, L))
        #print('binary a_j', a_j)
        #print('phase_j', phase_j)
        
    else:
        return None  
    
    return a_i, phase_i, a_j, phase_j




# this bath acts on each site of the chain    
def jump_EachSite(n, i, L):
    """
    Jump term of the form c^+_i + c_i = c^+_i + c_i
    
    Calculates the binary r, and phase alpha where
    (c^+_i + c_i) |a> = \alpha_i |r_i>,  \alpha_j |r_j>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L}].
    i : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """

#    print('i=', i)
#    print('binary=', n)
  
    if n[i] == 1:
        a = n_to_bin(n)
        exponent_i = np.sum(n[0:i])    
        #print('exponent_i', exponent_i)
        a_i =  a - 2 ** i
        #print('a_i', bin_to_n(a_i, L))
        #print('a_i', a_i)
        phase_i = (-1) ** exponent_i 
    
    elif n[i] == 0:
        a = n_to_bin(n)
        exponent_i = np.sum(n[0:i])            
        #print('exponent_i', exponent_i)
        a_i =  a + 2 ** i 
        #print('a_i', bin_to_n(a_i, L))
        #print('a_i', a_i)
        phase_i = (-1) ** exponent_i 
        
    else:
        return None 
        
    return a_i, phase_i

# this bath acts only on the first and last sites of the chain
def jump_EdgeSites(n, i, L):
    """
    Jump term of the form c^+_0 + c_0 + c^+_L + c_L 
    
    Calculates the binary r, and phase alpha where
    (c^+_0 + c_0 + c^+_L + c_L) |a> = \alpha_i |r_i>,  \alpha_j |r_j>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L}].
    i : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """

    print('i=', i)
    print('binary=', n)
    
    if (i == 0 or i == L-1) and n[i] == 1:
        a = n_to_bin(n)
        exponent_i = np.sum(n[0:i])    
        print('exponent_i', exponent_i)
        a_i =  a - 2 ** i
        print('a_i', bin_to_n(a_i, L))
        #print('a_i', a_i)
        phase_i = (-1) ** exponent_i 
    
    elif (i == 0 or i == L-1) and n[i] == 0:
        a = n_to_bin(n)
        exponent_i = np.sum(n[0:i])            
        print('exponent_i', exponent_i)
        a_i =  a + 2 ** i 
        print('a_i', bin_to_n(a_i, L))
        #print('a_i', a_i)
        phase_i = (-1) ** exponent_i 
    else:
        return None    
          
    return a_i, phase_i

# this bath acts only on the first site of the chain
def jump_OneEdgeSite(n, i, j, theta, L):
    """
    Jump term of the form c^+_0 + c_0  
    
    Calculates the binary r, and phase alpha where
    (c^+_0 + c_0 ) |a> = \alpha_i |r_i>,  \alpha_j |r_j>
    with a obtained from the state n = [n0,...,n_{L-1}]

    Parameters
    ----------
    n : list of integers
        the binary representation of the state [n0,n1,...,n_{L}].
    i : int

    Returns
    -------
    r and alpha if the state |a> is not annihilated,
    otherwise return None.

    """


    if i != j:
        print('i=', i)
        print('j=', j)
        #print('binary=', n)
        
        if (i == 0) and n[i] == 1:
            a = n_to_bin(n)
            exponent_i = np.sum(n[0:i])    
            #print('exponent_i', exponent_i)
            a_i =  a - 2 ** i
            print('a_i', bin_to_n(a_i, L))
            print('a_i', a_i)
            phase_i = (-1) ** exponent_i * np.cos(theta)
        
        elif (i == 0) and n[i] == 0:
            a = n_to_bin(n)
            exponent_i = np.sum(n[0:i])            
            #print('exponent_i', exponent_i)
            a_i =  a + 2 ** i 
            print('a_i', bin_to_n(a_i, L))
            print('a_i', a_i)
            phase_i = (-1) ** exponent_i * np.sin(theta)
    
        
        if (i == 0) and n[j] == 1:
            a = n_to_bin(n)
            exponent_j = np.sum(n[0:j])      
            #print('exponent_j', exponent_j)
            a_j =  a - 2 ** j
            print('a_j', bin_to_n(a_j, L))
            print('a_j', a_j)
            phase_j = - (-1) ** (exponent_j) * np.sin(theta)       #adding a minus in front of a_i+1 as in Topol Quantum wire paper (DOI: 10.1038/NPHYS2106)
        
        elif (i == 0) and n[j] == 0:
            a = n_to_bin(n)
            exponent_j = np.sum(n[0:j]) 
            #print('exponent_j', exponent_j)
            a_j =  a + 2 ** j
            print('a_j', bin_to_n(a_j, L))
            print('a_j', a_j)
            phase_j = (-1) ** (exponent_j) * np.cos(theta)
        else:
            return None  
        
        return a_i, phase_i, a_j, phase_j

    
 

@dataclass
class SuperOperator:

    """ Generates the Jump operator:
     L_jump_i = 1/2 (c^+_i + c_i - c_i+1 + c^+_i+1)  
     
     and the Superoperator:
     
    L_super = \sum_i L_jump_i tensor L_jump_i^* - 1/2(L_jump_i^+L_jump_i) tensor 1 - 1/2 1 tensor (L_jump_i^+L_jump_i)^*  
     
     """

    def __init__(self, L, theta, kappa, bath):
        self.L = L              # Number of sites
        self.theta = theta
        self.kappa = kappa
        self.bath = bath
               
    
    def calc_L_super(self, bc, theta, kappa, bath):    
        
        """
        Calculating L_jump_i, then building L_super_i and finally summing over all L_super_i's

        """             
        L = self.L

        dmns = 4**L
        L_super = np.zeros((L,dmns, dmns))
        
        # Boundary conditions
        if bc == 'periodic':
            Lhop = L
        elif bc == 'open' and (bath == 'TBD' or bath == 'TBD_prod'):
            Lhop = L - 1
        elif (bc == 'periodic' or bc == 'open') and (bath == 'Each_site' or bath == 'Edge_sites'  or bath == 'OneEdge_site'):
            Lhop = L
        else:
            raise ValueError('boundary condition must be "periodic" or "open" and bath must be "TBD","TBD_prod", "Each_site" or "Edge_sites" or "OneEdge_site" ')
        
        Id = np.eye(int(2**L))
        
        dim = 2 ** L 
        
        # Nearest-neighbor terms
        for i in range(Lhop):
            #print('i=', i)
            L_jump = np.zeros((dim, dim))
            
            if bc == 'periodic':
                j = np.mod(i + 1, Lhop)  # j is either i+1 or 0
            else:
                j = i + 1

            for r in range(dim): 
                #print('r=', r)
                n = bin_to_n(r, L)
                #YB: buiding a single jump operator for site i
                #print('r=', r)
                #print('n=', n)

                # OBS factor 1/2 is from the Quantum wire paper
                try:
                   # get the binary state and the phase of the jump operator
                   
                   if bath == 'TBD':
                       a_i, h_i, a_j, h_j  = jump_TBD(n, i, j, theta, L)

                       #print('a_i', a_i)
                       #print('a_j', a_j)
                       #print('h_i', h_i)
                       #print('h_j', h_j)
                       
                       # assign values to the correct location in the Hilbert space
                       L_jump[a_i, r] += 1/np.sqrt(2) * h_i 
                       L_jump[a_j, r] += 1/np.sqrt(2) * h_j 
    
                       
                       #print('L_jump[a_i, r]', L_jump[a_i, r])
                       #print('L_jump[a_j, r]', L_jump[a_j, r]) 


                   elif bath == 'TBD_prod':
                       a_i, h_i, a_j, h_j  = jump_TBD_prod(n, i, j, L)

                       #print('a_i', a_i)
                       #print('a_j', a_j)
                       #print('h_i', h_i)
                       #print('h_j', h_j)
                       
                       # assign values to the correct location in the Hilbert space
                       L_jump[a_i, r] += 1/4 * h_i 
                       L_jump[a_j, r] += 1/4 * h_j 
    
                       
                       #print('L_jump[a_i, r]', L_jump[a_i, r])
                       #print('L_jump[a_j, r]', L_jump[a_j, r]) 
                       
                       
                   
                   elif bath == 'Each_site':
                       a_i, h_i  = jump_EachSite(n, i, L)

                       #print('a_i', a_i)
                       #print('h_i', h_i)
                       
                       # assign values to the correct location in the Hilbert space
                       L_jump[a_i, r] +=  h_i
                       
                       #print('L_jump[a_i, r]', L_jump[a_i, r])
                       
                   elif bath == 'Edge_sites':
                       a_i, h_i  = jump_EdgeSites(n, i, L)

                       #print('a_i', a_i)
                       #print('h_i', h_i)
                    
                       # assign values to the correct location in the Hilbert space
                       L_jump[a_i, r] +=  h_i
                    
                       #print('L_jump[a_i, r]', L_jump[a_i, r])
                       
                   elif bath == 'OneEdge_site':
                       a_i, h_i, a_j, h_j  = jump_OneEdgeSite(n, i, j, theta, L)
    
                       #print('a_i', a_i)
                       #print('h_i', h_i)
                     
                       # assign values to the correct location in the Hilbert space
                       L_jump[a_i, r] += 1/np.sqrt(2) * h_i
                       L_jump[a_j, r] += 1/np.sqrt(2) * h_j 
                     
                       #print('L_jump[a_i, r]', L_jump[a_i, r])
                       
                       
                       

                except TypeError as e:
                    #print(f"TypeError occurred: {e}")
                    #print(f"n: {n}")
                    #print(f"i: {i}")
                    #print(f"j: {j}")
                    pass
                
            
            
# =============================================================================
#            plt.matshow(np.real(L_jump))
#            cbar = plt.colorbar()
#            plt.title('L_jump$_{}$'.format(i), fontsize=6)
#            print('L_jump=',np.real(L_jump))
#            cbar.ax.tick_params(labelsize=6)
            
# =============================================================================
            #print('i=',i)
            #print('L_jump=', L_jump)
            
             # building the super-operator for site i               
            L_jump_dagger = np.conj(np.transpose(L_jump))
            
            #term1 = np.kron(np.transpose(L_jump_dagger), L_jump)
            
            #term2 = 1/2*np.kron(Id, np.matmul(L_jump_dagger,L_jump))
            
            #term3 = 1/2*np.kron(np.transpose(np.matmul(L_jump_dagger,L_jump)), Id)
            
            L_super[i,:,:] = np.kron(L_jump, np.conj(L_jump)) - 1/2*np.kron(np.dot(L_jump_dagger,L_jump), Id) - 1/2*np.kron(Id, np.conj(np.dot(L_jump_dagger,L_jump)))
            
            #L_super[i,:,:] = term1 - term2 - term3
        
# =============================================================================
#            plt.matshow(np.real(L_super[i,:,:]))
#            cbar = plt.colorbar()
#            plt.title('L_super$_{}$'.format(i), fontsize=6)
#            plt.tick_params(axis='both', labelsize=6)
#            cbar.ax.tick_params(labelsize=6)
# =============================================================================
            
            #print('L_super_i', L_super[i,:,:])
            
        #Summing over all i to get the final Super-operator
# =============================================================================
#        plt.matshow(np.real(np.sum(L_super, axis=0)))
#        cbar = plt.colorbar()
#        plt.title('L_super', fontsize=6)
#        plt.tick_params(axis='both', labelsize=6)
#        cbar.ax.tick_params(labelsize=6)
# =============================================================================
        
        
        # Multiplying the super-operator with \kappa0 as in Topol by dissipartion paper (DOI: 10.1038/NPHYS2106)
        L_super_tot = kappa * np.sum(L_super, axis=0)
        print('L_super shape:', np.shape(L_super))
        
        #print('L_super=', L_super_tot)
                
    
        return L_super_tot


#####################################################################


@dataclass
class IsingMajorana:

    """ Generates the Hamiltonian for a disordered Ising-Majorana chain
     whose Hamiltonian is
     H = \sum_i J_i X_i X_i+1 + h_i Z_i + g Z_i Z_i+1 + g X_i X_i+2
       = \sum_i J_i (c^+_i+1 c_i + hc) + J_i (c^+_i c^+_i+1 + hc) + h_i (2 n_i - 1) - g (2 n_i - 1) (2n_i+1 - 1)
       - g (c^+_i+2 c_i + hc) - g (c^+_i c^+_i+2 + hc) """

    def __init__(self, XX, XX2, Z, ZZ, delta, L):
        self.XX = XX            # Strength of nearest-neighbor hopping in x direction
        self.XX2 = XX2          # Strength of next-nearest-neighbor hopping in x direction
        self.Z = Z              # Strength of onsite field in z direction
        self.ZZ = ZZ            # Strength of spin interaction in z direction
        self.delta = delta      # Strength of the disorder
        self.L = L              # Number of sites
        self.toprint = ['XX', 'XX2', 'Z', 'ZZ', 'delta', 'L' ]

    def __str__(self):
        """
        Prints out all the model parameters and their values.

        Params:
        ------
        self: Parameters of the model

        Returns:
        -------
        Model parameters and their values
        """

        out = ''
        for key in self.toprint:
            out += key + ':\t' + str(self.__dict__[key]) + '\n'
        return out

    def new_disorder_realization(self, seed):
        """
        Extracts a new disorder realization

        Params:
        ------
        self: Parameters of the model
        seed: seed for the random number generator

        Returns:
        -------
        None
        """
        
        #setting up the specific string of random numbers
        self.disorder = np.random.RandomState(seed)
        # random.uniform(low=0.0, high=1.0, size=None): all values between [0,1) are equally likely
        self.disXX = self.XX * self.disorder.uniform(0, np.exp(0.5 * self.delta), size=self.L)
        self.disZ = self.Z * self.disorder.uniform(0, np.exp(-0.5 * self.delta), size=self.L)
        
        # special case to reproduce results of Topol by dissipartion paper (DOI: 10.1038/NPHYS2106)
        # disporder is only in \mu and has to be \mu << pi/4
# =============================================================================
#        self.disXX = self.XX * np.ones(self.L)
#        self.disZ = self.Z * self.disorder.uniform(0, np.exp(-0.5 * self.delta), size=self.L) * np.ones(self.L)
# =============================================================================


    def clean_realization(self):
        """
        Extracts a clean realization (i.e., constant hopping and onsite potential)

        Params:
        ------
        self: Parameters of the model

        Returns:
        -------
        None
        """

        self.disXX = self.XX * np.ones(self.L)
        self.disZ = self.Z * np.ones(self.L) 



    #Calculating Hamiltonian without parity
    
    def calc_noparity_Hamiltonian(self, bc):
        """
        Calculates the Hamiltonian without parity consideration.

        Parameters
        ----------
        bc :     {string, optional} Sets the boundary condition. Can be either 'periodic' or 'open'

        Returns
        -------
        H :  {np.array} The Hamiltonian with given boundary conditions.
        """

        L = self.L
        dim = 2 ** L
        H = np.zeros((dim, dim))
        
        print('self.disXX=', self.disXX)
        print('disZ=', self.disZ)

        # Boundary conditions
        if bc == 'periodic':
            Lhop = L
            Lhop2 = L
        elif bc == 'open':
            Lhop = L - 1
            Lhop2 = L - 2
        
        else:
            raise ValueError('boundary condition must be "periodic" or "open"')

        # Hamiltonian
        for r in range(dim):      
            n = bin_to_n(r, L)
            
#YB: np.roll cycles through the array by 1 step to the left

            # Diagonal term from interactions: - g (2n_i - 1) (2n_i+1 - 1)
            two_n_minus_one = 2 * n - 1
            H[r, r] += self.ZZ * np.dot(two_n_minus_one[:Lhop], np.roll(two_n_minus_one, -1)[:Lhop])



            # Diagonal term from onsite random field: h_i(2n_i - 1)
            for i in range(L):
                H[r, r] -= self.disZ[i] * onsite_field(n, i)
                #print('i', i)
                #print('n=', n)
                #print('onsite_field', onsite_field(n, i))
                #print('disZ', self.disZ[i])
                #print('H[r,r] \mu', H[r, r])
            # Nearest-neighbor terms
            for i in range(Lhop):
                if bc == 'periodic':
                    j = np.mod(i + 1, Lhop)  # j is either i+1 or 0
                elif bc == 'open':
                    j = i + 1
                else:
                    raise ValueError('boundary condition must be "periodic" or "open"')

#YB: change to the parity basis, calculate hopping/pairing value ht/hp, add it to the corresponding off-diagonal in the Hamiltonian
#    and its conjugate to the opposite side of the diagonal.
                #print('i:', i)
                #print('j:', j)
                # Hopping term
                try:
                    b, h = hopping(n, i, j)
                    ht = h * self.disXX[i]
                    #print('h', h)
                    #print('ht', ht)
                    H[b, r] += ht
                    H[r, b] += np.conjugate(ht)
                    #print('b', b)
                    #print('r', r)
                    #print('H[b, r] hop', H[b, r])
                    #print('H[r, b] hop', H[r, b])
                except TypeError as e:
                    #print(f"TypeError occurred: {e}")
                    #print(f"n: {n}")
                    #print(f"i: {i}")
                    #print(f"j: {j}")
                    pass

                # Pairing term
                try:
                    b, h = pairing(n, i, j)
                    hp = h * self.disXX[i] 
                    #print('hp', hp)
                    H[b, r] += hp
                    H[r, b] += np.conjugate(hp)
                    #print('b', b)
                    #print('r', r)
                    #print('H[b, r] pair', H[b, r])
                    #print('H[r, b] pair', H[r, b])
                except TypeError:
                    pass

            # Next-nearest-neighbor terms
            for i in range(Lhop2):
                if bc == 'periodic':
                    k = np.mod(i + 1, Lhop)   # k is either i+1 or 0
                    j = np.mod(i + 2, Lhop2)  # j is either i+2 or 0 or 1
                else:
                    k = i + 1
                    j = i + 2

                # Hopping term
                try:
                    b, h = hopping(n, i, j)
                    ht = - h * self.XX2
                    phase = onsite_field(n, k)
                    H[b, r] += (phase * ht)
                    H[r, b] += np.conjugate(phase * ht)
                    
                except TypeError:
                    pass

                # Pairing term
                try:
                    b, h = pairing(n, i, j)
                    hp = - h * self.XX2
                    phase = onsite_field(n, k)
                    H[b, r] += (phase * hp)
                    H[r, b] += np.conjugate(phase * hp)
                except TypeError:
                    pass
        return H

    def calc_opdm_operator(self, parity='even'):
        """
        Calculates the matrix rho_ij = <r|c_i^dagger c_j|s>
        and <r|c_i c_j|s> in subspace of even or odd parity.
        rho['even/odd'][i,j] = rho_ij
        with rho_ij a sparse matrix.

        Parameters
        ----------
        parity : string, optional
            The parity of the basis for rho. The default is 'even'.


        Returns
        -------
        None. But sets self.rho to rho.

        """
        L = self.L
        dim = 2 ** (L - 1)
        rho = {'eh': {}, 'hh': {}}


        # Initializing rho_eh operator as a lil_matrix
        for i, j in itertools.product(range(L), range(L)):
            rho['eh'][i, j] = sparse.lil_matrix((dim, dim))
            rho['hh'][i, j] = sparse.lil_matrix((dim, dim))

        # Parity sector
        if parity == 'even':
            r_to_a = self.rp2a
            a_to_r = self.a2rp
        elif parity == 'odd':
            r_to_a = self.rm2a
            a_to_r = self.a2rm
        else: raise ValueError('parity must be "even" or "odd"')

        # Calculation of the matrix rho in the parity subspace
        for r in range(dim):

            # State r in the different basis
            a = r_to_a[r]                                            # Binary
            n = bin_to_n(a, L)                                       # Number representation

            # Iterate over the possible ij combinations
            for i, j in itertools.product(range(L), range(L)):

                # Electron-hole block
                try:
                    b, alpha = hopping(n, i, j)                      # c_i^\dagger c_j /r>
                    rho['eh'][i, j][a_to_r[b], r] = alpha            # <b/ c^\dagger c/ r>
                except TypeError:
                    pass

                # Hole-hole block
                try:
                    b, alpha = pairing(n, i, j)                      # c_i c_j /r>
                    rho['hh'][i, j][a_to_r[b], r] = alpha            # <b/ c^\dagger c/ r>
                except TypeError:
                    pass

        # Convert sparse lil matrices to csr
        self.rho = {'eh': {}, 'hh': {}}
        for key in rho['eh']:
            self.rho['eh'][key] = rho['eh'][key].tocsr()
            self.rho['hh'][key] = rho['hh'][key].tocsr()

        return None


#YB: rho_opdm calculated by parts for rho_eh and rho_hh separately. 

    def calc_opdm_from_psi(self, psi, parity='even'):
        """
        Calculates the one-particle-density matrix from a state psi.

        rho_opdm =  [[ <psi|c^\dagger_i c_j|psi> , <psi|c^\dagger_i c^\dagger_j|psi>],
                     [          <psi|c_i c_j|psi>,   <psi|c_i c^\dagger_j|psi> ]]

        Parameters
        ----------
        psi : numpy arrary
            The state for which we want the opdm.
        parity : TYPE, optional
            the parity of the subspace in which the state lives. The default is 'even'.

        Returns
        -------
        rho_opdm : numpy array 2L x 2L
            the opdm.
        """

        if not hasattr(self, 'rho'):
            self.calc_opdm_operator(parity)

        L = self.L
        rho_opdm = np.zeros((2 * L, 2 * L))

        for i, j in itertools.product(range(L), range(L)):
            rho_opdm[i, j] = np.dot(psi.conjugate(), (self.rho['eh'][i, j] * psi))
            rho_opdm[i + L, j] = np.dot(psi.conjugate(), (self.rho['hh'][i, j] * psi))
        rho_opdm[L: 2 * L, L: 2 * L] = np.eye(L) - rho_opdm[:L, :L]  # .T
        rho_opdm[:L, L:2 * L] = rho_opdm[L: 2 * L, :L].T.conjugate()

        return rho_opdm

    def expand_psi_to_full_space(self, psi, parity='even'):
        """
        Transform the state psi from one parity sector to the full Hilbert space.

        Parameters
        ----------
        psi : numpy arrary
            The state for which we want the opdm.
        parity : TYPE, optional
            the parity of the subspace in which the state lives. The default is 'even'.

        Returns
        -------
        psi_full : numpy array 2L.
        """

        r_to_a, a_to_r = self.get_ar(parity)
        psi_full = np.zeros(2 ** self.L, dtype=psi.dtype)

        for r in range(2 ** (self.L - 1)):
            psi_full[r_to_a[r]] = psi[r]

        return psi_full



# Other (maybe useful??) functions
def Ham_bdg(L, t, delta, mu):
    """
    Calculates the single-particle Hamiltonian

    Params:
    ------
    L:         {int} System size
    t:       {float} Hopping amplitude
    delta:   {float} Pairing amplitude
    mu:      {float} Chemical potential

    Returns:
    H:    {np.array} Single-particle hamiltonian
    """

    sigma = [np.array([[1, 0], [0, 1]]),        # Pauli 0
             np.array([[0, 1], [1, 0]]),        # Pauli x
             np.array([[0, -1j], [1j, 0]]),     # Pauli y
             np.array([[1, 0], [0, -1]])]       # Pauli z

    # Real space basis
    vector = np.zeros(L); basis = []
    for index in range(L):
        vector[index] = 1
        basis.append(vector)
        vector = np.zeros(L)

    # Real space projectors
    n_offdiag = []                               # |n><n+1|
    n_diag = []                                  # |n><n|
    for i in range(L):
        j = np.mod(i + 1, L)
        n_offdiag.append(np.outer(basis[i], basis[j]))
        n_diag.append(np.outer(basis[i], basis[i]))

    # Hamiltonian construction
    n_offdiag_tot = sum(n_offdiag)               # sums all n_ matrices as the tensor product is linear
    n_diag_tot = sum(n_diag)
    tau = -(t * sigma[3] + 1j * delta * sigma[2])
    H = np.kron(tau, n_offdiag_tot)
    H = H + np.conj(H.T) - mu * np.kron(sigma[3], n_diag_tot)

    return 0.5 * H

def ManyBodySpectrum(E):
    """
    Calculates the many body spectrum provided the single-particle levels

    Params:
    ------
    E:  {np.array} Single-particle energy levels

    Returns:
    -------
    energy:  {np.array} Many-body levels
    """

    spectrum = []
    combinations = itertools.product([-1, 1], repeat=len(E))
    for i in combinations: spectrum.append(np.dot(E, np.array(i)))

    energy = np.array(spectrum)
    idx = energy.argsort()
    energy = energy[idx]
    return energy

def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates

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
    half_Lx = np.floor(L_x / 2)
    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + np.heaviside(half_Lx - x[site], 0) * abs(half_Lx - x[site])
    x = (x + deltaLx) % L_x                                      # Relabel of the operators
    X = np.concatenate((x, x))                                   # X vector in BdG basis
    X = np.reshape(X, (len(X), 1))                               # Column vector x

    # Marker calculation
    M = P @ S @ (X * P)                                          # Marker operator (BdG x position space)
    marker = -2 * (M[site, site] + M[L_x + site, L_x + site])    # Trace over each site

    return marker