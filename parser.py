from argparse import ArgumentParser
from argparse import BooleanOptionalAction
import numpy as np

def parse():
    parser = ArgumentParser(description='IsingMajorana')

    parser.add_argument('-d', '--delta', type=float, help='Disorder strength', default= 10.0)
    parser.add_argument('--hop', type=float, help='Hopping and onsite potential strength', default=4.0)
    parser.add_argument('--mu', type=float, help='Onsite chemical potential strength', default=0.0)
    parser.add_argument('-p', '--prefactor', type=float, help='Disorder prefactor common to both disorder distributions', default=0.0)
    parser.add_argument('-g', '--interaction', type=float, help='Interaction strength', default=2.0)
    parser.add_argument('-theta', '--dissipation_th', type=float, help='Dissipation parameter', default = np.pi / 4)
    parser.add_argument('-kappa', '--dissipation_k', type=float, help='Dissipation parameter', default=0.2)
    parser.add_argument('--system', type=str, help='System type', choices=['open', 'closed'], default='open')
    parser.add_argument('-L', '--system_size', type=int, help='System size', default=6)
    parser.add_argument('--k_eigs', type=int, help='Number of eigenstates per parity sector', default=1)
    parser.add_argument('--bc', type=str, help='Boundary condition', choices=['open', 'periodic'], default='open')
    parser.add_argument('--bath', type=str, help='The type of bath coupling', choices=['TBD', 'Each_site', 'Edge_sites', 'OneEdge_site'], default='TBD')
    #YB: keeping seed constant will generate same string of random numbers every time the code is run. Changing the seed values will change the string of random numbers.
    parser.add_argument('-s', '--seed', type=int, help='Seed for the random number generator', default=42)
    parser.add_argument('-N', '--N_rep', type=int, help='Number of disorder realizations', default=1)
    parser.add_argument('--offset', type=int, help='Offset for the seed (useful in case more jobs with the same parameters are running in parallel)', default=0)
    parser.add_argument('--clean', type=str, help='If True the Hamiltonian is clean and translation invariant', choices=['True', 'False'], default='False')
    parser.add_argument('--ss', type=str, help='Calculate steady state?', choices=['True', 'False'], default='False')
    parser.add_argument('--show', help='If True the disorder averaged local information per scale is shown', default=True)
    parser.add_argument('--t_max', '--real_t_evo', type=float, help='Time value for the real time evolution', default=15.5)
    parser.add_argument('--dt', '--dt', type=float, help='Time step for the real time evolution', default=0.5)

    args, unknown = parser.parse_known_args()

    return args



