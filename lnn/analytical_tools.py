# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import numpy as np
from scipy.integrate import odeint, solve_ivp
import torch
from torch.func import vmap, jacrev, hessian
from torchdiffeq import odeint as tor_odeint
from functools import partial


def get_qdotdot(total_state, analytical_solution):
    '''
    total_state is a tensor of generalised coords and velocities,
    returns q double dot.
    '''
    dims = total_state.shape[1]
    if dims%2 != 0:
        raise ValueError("The input total_state must have an even number of columns.")
    return analytical_solution(total_state[:, :dims//2], total_state[:, dims//2:])


# Returns q_dot and q_dotdot 

#Want to make these functions as general as possible. so that they can be used for any system.

def analytically_differentiated_state(x, t, analytical_solution):
    d = torch.zeros_like(x)
    
    #Remove 2's, make it general
    n_cols = x.shape[1]
    if n_cols % 2 != 0:
        raise ValueError("The input x must have an even number of columns.")
    d[:, :n_cols//2] = x[:, n_cols//2:]
    d[:, n_cols//2:] = get_qdotdot(x, analytical_solution)
    #print('DIAGNOSE VELS:',d[:, :n_cols//2] )
    #print('DIAGNOSE ACCS:',d[:, n_cols//2:])
    return d


def ode_solve_analytic(q0, qt0, t, analytical_solution):

    x0 = np.append(q0, qt0)
    def f_analytic(x, t):
        d = np.zeros_like(x)
        n_cols = x.shape[0]
        d[:n_cols//2] = x[n_cols//2:]
        d[n_cols//2:] = np.squeeze(get_qdotdot(np.expand_dims(x, axis=0), analytical_solution))
        # print(x, d)
        return d
    
    return odeint(f_analytic, x0, t, rtol=1e-10, atol=1e-10) # scipy's ode (This is being used for both demo's analytic and test's analytic solution.)

def get_diff_state_lagr(lagrangian, t, x):
    """
    Compute state derivatives using modern torch.func approach with solve instead of pinv.
    Based on the updated LNN.py implementation.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    B, D = x.shape
    n = D // 2
    
    # Ensure x requires grad for derivative computation
    x = x.requires_grad_(True)
    qdot = x[:, n:]
    
    # Compute Jacobian using vmap and jacrev
    def single_jac(xi):
        return jacrev(lagrangian)(xi)
    
    jac = vmap(single_jac)(x)  # [B, D]
    
    # Compute Hessian using vmap
    def single_hes(xi):
        return hessian(lagrangian)(xi)
    
    hes = vmap(single_hes)(x)  # [B, D, D]
    
    # Extract relevant matrices for Euler-Lagrange equations
    A = hes[:, n:, n:]  # Mass matrix [B, n, n]
    B_mat = hes[:, n:, :n]  # [B, n, n] 
    C = jac[:, :n].unsqueeze(-1)  # [B, n, 1]
    
    # Right-hand side: C - B_mat @ qdot
    q = qdot.unsqueeze(-1)  # [B, n, 1]
    rhs = C - torch.matmul(B_mat, q)  # [B, n, 1]
    
    # Solve A @ qddot = rhs using torch.linalg.solve
    try:
        qddot = torch.linalg.solve(A, rhs).squeeze(-1)  # [B, n]
    except torch.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        print("Warning: Using pseudo-inverse fallback due to singular matrix")
        qddot = torch.matmul(torch.linalg.pinv(A), rhs).squeeze(-1)
    
    # Return [qdot, qddot]
    result = torch.cat([qdot, qddot], dim=1)
    
    return result

# This takes the starting coordinates and f fills the first argument of get_diff_state_lagr with "lagrangian",
# so that this can directly go into tor_odeint function, which is similar to the previous one.
def torch_solve_ode(x0, t, lagrangian):
    def f(t, x):
        return get_diff_state_lagr(lagrangian, t, x)
    return tor_odeint(f, x0, t, rtol=1e-6, atol=1e-8)


