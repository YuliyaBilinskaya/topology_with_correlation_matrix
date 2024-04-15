from argparse import ArgumentParser
from argparse import BooleanOptionalAction
import numpy as np

def parse():
    parser = ArgumentParser(description='IsingMajorana')

    parser.add_argument('-d', '--delta', type=float, help='Disorder strength', default=-1.0)
    parser.add_argument('--hop', type=float, help='Hopping and onsite potential strength', default=1.0)
    parser.add_argument('--mu', type=float, help='Onsite chemical potential strength', default=1.0)
    parser.add_argument('-p', '--prefactor', type=float, help='Disorder prefactor common to both disorder distributions', default=0.0)
    parser.add_argument('-g', '--interaction', type=float, help='Interaction strength', default=0.0)
    parser.add_argument('-theta', '--dissipation_th', type=float, help='Dissipation parameter', default=0)
    parser.add_argument('-kappa', '--dissipation_k', type=float, help='Dissipation parameter', default=0.1)
    parser.add_argument('--system', type=str, help='System type', choices=['open', 'closed'], default='open')
    parser.add_argument('-L', '--system_size', type=int, help='System size', default=6)
    parser.add_argument('--k_eigs', type=int, help='Number of eigenstates per parity sector', default=1)
    parser.add_argument('--bc', type=str, help='Boundary condition', choices=['open', 'periodic'], default='open')
    #YB: keeping seed constant will generate same string of random numbers every time the code is run. Changing the seed values will change the string of random numbers.
    parser.add_argument('-s', '--seed', type=int, help='Seed for the random number generator', default=42)
    parser.add_argument('-N', '--N_rep', type=int, help='Number of disorder realizations', default=1)
    parser.add_argument('--offset', type=int, help='Offset for the seed (useful in case more jobs with the same parameters are running in parallel)', default=0)
    parser.add_argument('--clean', type=str, help='If True the Hamiltonian is clean and translation invariant', choices=['True', 'False'], default='False')
    parser.add_argument('--show', help='If True the disorder averaged local information per scale is shown', default=True)
    parser.add_argument('--t_max', '--real_t_evo', type=float, help='Time value for the real time evolution', default=7.02)
    parser.add_argument('--dt', '--dt', type=float, help='Time step for the real time evolution', default=0.02)

    args, unknown = parser.parse_known_args()

    return args



