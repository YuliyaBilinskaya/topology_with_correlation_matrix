"""
Time integrators for the correlation-matrix evolution (stand-alone module).

'euler' : calls the existing evolve_C_mk_A_mk_delta_t unchanged (bit-identical to the current code).
'rk45'  : the same explicit Runge-Kutta scheme as LITE's RungeKuttaSolver.solve, with a fixed step:
          same Butcher tableau, same stage construction, same order of summation, advanced with the
          5th-order weights.
"""
import numpy as np
from functions import evolve_C_mk_A_mk_delta_t

# Copied expression-by-expression from LITE: local_information/core/runge_kutta_solvers/runge_kutta_parameters.py
# (function RK45_parameters)  ->  identical float64 values.
RK45_C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
RK45_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [3 / 32, 9 / 32, 0, 0, 0, 0],
        [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
    ]
)
RK45_B5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])  # b_higher in LITE (used to advance)
RK45_B4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])           # b_lower in LITE (error estimate only)


def corr_rhs(C, A, gamma, J, h):
    """F(C, A_dagger) = d/dt (C, A_dagger), obtained from the existing Euler routine with delta_t = 1."""
    C1, A1 = evolve_C_mk_A_mk_delta_t(C, A, delta_t=1.0, gamma=gamma, J=J, h=h)
    return C1 - C, A1 - A


def rk45_step(C, A, dt, gamma, J, h, return_error=False):
    """One fixed-size step; mirrors RungeKuttaSolver.solve (stage loop and summations)."""
    n_stages = len(RK45_B5)

    fC, fA = corr_rhs(C, A, gamma, J, h)
    kC = [dt * fC]
    kA = [dt * fA]

    for j in range(1, n_stages):
        addC = RK45_A[j, 0] * kC[0]
        addA = RK45_A[j, 0] * kA[0]
        for l in range(1, j):
            addC = addC + RK45_A[j, l] * kC[l]
            addA = addA + RK45_A[j, l] * kA[l]
        fC, fA = corr_rhs(C + addC, A + addA, gamma, J, h)
        kC.append(dt * fC)
        kA.append(dt * fA)

    totC = RK45_B5[0] * kC[0]
    totA = RK45_B5[0] * kA[0]
    for j in range(1, n_stages):
        totC = totC + RK45_B5[j] * kC[j]
        totA = totA + RK45_B5[j] * kA[j]

    C_next = C + totC
    A_next = A + totA

    if return_error:
        errC = RK45_B5[0] * kC[0] - RK45_B4[0] * kC[0]
        errA = RK45_B5[0] * kA[0] - RK45_B4[0] * kA[0]
        for j in range(1, n_stages):
            errC = errC + (RK45_B5[j] - RK45_B4[j]) * kC[j]
            errA = errA + (RK45_B5[j] - RK45_B4[j]) * kA[j]
        return C_next, A_next, max(np.abs(errC).max(), np.abs(errA).max())
    return C_next, A_next


def make_stepper(integrator, gamma, J, h):
    """Returns step(C, A, dt) -> (C_next, A_next) for the chosen integrator."""
    if integrator == "euler":
        def step(C, A, dt):
            return evolve_C_mk_A_mk_delta_t(C, A, delta_t=dt, gamma=gamma, J=J, h=h)
    elif integrator == "rk45":
        def step(C, A, dt):
            return rk45_step(C, A, dt, gamma, J, h)
    else:
        raise ValueError(f"unknown integrator '{integrator}' (use 'euler' or 'rk45')")
    return step