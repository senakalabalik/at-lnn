# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import torch
from typing import Sequence, Optional
from torch.utils.data import TensorDataset, DataLoader
from analytical_tools import ode_solve_analytic, analytically_differentiated_state
import numpy as np


def create_test_data(particle, number_of_tests, time_bounds, time_step, dimension, position_tuples, 
                                                    velocity_tuples, seed, dtype=torch.float32, device=None, test_states=None):
    state_list = []
    diff_state_list = []
    t_test = torch.arange(time_bounds[0], time_bounds[1], time_step)

    
    if test_states is not None:
        number_of_tests = len(test_states)
        print('Number of states given for test_states is:', number_of_tests)

    if device == None:
        device = torch.device('cpu')
    for i in range(number_of_tests):
        if test_states is not None:
            start_state = torch.as_tensor(test_states[i]).unsqueeze(0) 
            start_pos_vector = start_state.squeeze()[:dimension]
            start_vel_vector = start_state.squeeze()[dimension:]

            #q0_cpu = start_pos_vector.detach().cpu().numpy()
            #qt0_cpu = start_vel_vector.detach().cpu().numpy()
            #t_test = t_test.to(torch.float64)
            #start_pos_vector = start_pos_vector.to(torch.float64)
            #start_vel_vector = start_vel_vector.to(torch.float64)

        else:
            pos_log_scale_indices = getattr(particle, 'position_log_scale_indices', None)
            vel_log_scale_indices = getattr(particle, 'velocity_log_scale_indices', None)
            
            start_pos_vector = torch.squeeze(random_initialize(position_tuples, dimension, 1, seed+dimension*i, log_scale_indices=pos_log_scale_indices))
            start_vel_vector = torch.squeeze(random_initialize(velocity_tuples, dimension, 1, seed+dimension*(number_of_tests+i), log_scale_indices=vel_log_scale_indices))
            # Combine into a single state vector with a batch dimension of 1
            start_state = torch.cat([start_pos_vector, start_vel_vector]).unsqueeze(0)

        # Enforce constraints if the method exists
        if hasattr(particle, 'enforce_constraints'):
            # Check whether any element in particle.scale is other than 1, if so, give a warning
            if any(s != 1.0 for s in particle.scale):
                print("Warning: Scaling other than 1 detected while calling enforce_constraints.")
            # The method expects a batch, so we pass [1, 6] and get [1, 6] back
            corrected_start_state = particle.enforce_constraints(start_state)
            # Split back into position and velocity for the ODE solver
            start_pos_vector = corrected_start_state.squeeze()[:dimension]
            start_vel_vector = corrected_start_state.squeeze()[dimension:]

        solved = ode_solve_analytic(start_pos_vector, start_vel_vector, t_test, particle.solve_acceleration)
        if torch.is_tensor(solved):
            q_qdot = solved.clone().detach()
        else:
            q_qdot = torch.tensor(solved)
            qdot_qdotdot = analytically_differentiated_state(q_qdot, t_test, particle.solve_acceleration) # Isn't this better to take out of the if-else?

#        for j in angle_indices:
#            q_qdot[:,j] = q_qdot[:,j] % 2*np.pi
        if (device != None):
            q_qdot = q_qdot.to(dtype=dtype, device=device)
            qdot_qdotdot = qdot_qdotdot.to(dtype=dtype, device=device)
            
        state_list.append(q_qdot)
        diff_state_list.append(qdot_qdotdot)
    
    return [state_list, diff_state_list, t_test]

def normalize_training_data(training_data, dimension, scale_factor=None, time_scaling=False): # still needs some work

    if time_scaling==True:
        time_scale_k=torch.max(torch.abs(training_data[0][:,dimension:]), dim=0).values # max of velocities
        training_data[0][:,dimension:] = training_data[0][:,dimension:]/time_scale_k # velocities
    
        training_data[1][:,:dimension] = training_data[1][:,:dimension]/time_scale_k # velocities
        training_data[1][:,dimension:] = training_data[1][:,dimension:]/(time_scale_k)**2 # accelerations, 
        maximum = time_scale_k.tolist()

    else: 
        if scale_factor is None:
            maximum = torch.max(torch.abs(training_data[0][:,:dimension]), dim=0).values
            maximum = torch.max(torch.max(torch.abs(training_data[0][:,dimension:]), dim=0).values, maximum) #making this [0] -> [1] looks simply like a solution, but that scales time uncontrolledly
        else:
            maximum = torch.Tensor(scale_factor).to(training_data[0].device)
        
        training_data[0][:,:dimension] = training_data[0][:,:dimension]/maximum # positions
        training_data[0][:,dimension:] = training_data[0][:,dimension:]/maximum # velocities
        
        training_data[1][:,:dimension] = training_data[1][:,:dimension]/maximum # velocities
        training_data[1][:,dimension:] = training_data[1][:,dimension:]/maximum # accelerations
    
    return maximum.tolist(), training_data
        
def normalize_testing_data(testing_data, dimension, maximum):
    for i in range(len(testing_data[0])):
        testing_data[0][i][:,:dimension] = testing_data[0][i][:,:dimension]/maximum
        testing_data[0][i][:,dimension:] = testing_data[0][i][:,dimension:]/maximum
        testing_data[1][i][:,:dimension] = testing_data[1][i][:,:dimension]/maximum
        testing_data[1][i][:,dimension:] = testing_data[1][i][:,dimension:]/maximum
    return testing_data

def random_initialize(
    tup_lst: Sequence[tuple],
    dim: int,
    n: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device = None,
    log_scale_indices: Optional[Sequence[int]] = None
):
    """
    Random initialization with optional per-dimension log-uniform sampling.

    Parameters
    ----------
    tup_lst : sequence of (start, end) pairs
        Length must equal `dim`. Each pair gives the sampling interval for that coordinate.
    dim : int
        Number of dimensions (must match len(tup_lst)).
    n : int
        Number of samples (rows) to generate.
    seed : int or None
        If provided, used for reproducible sampling. The original per-dimension reseeding
        behavior is preserved (seed, seed+1, ...).
    dtype : torch.dtype
    device : torch.device or None
    log_scale_indices : sequence of ints or None
        Indices (0-based) of dimensions to sample log-uniformly. Default: [] (i.e. none).

    Returns
    -------
    torch.Tensor of shape (n, dim)
    """
    if len(tup_lst) != dim:
        raise ValueError(f"The length of the tuple list should match the dimension: {dim}")
    if device is None:
        device = torch.device('cpu')

    # Normalize log-scale index list
    if log_scale_indices is None:
        log_scale_indices = []
    else:
        # ensure it's a list of unique ints
        try:
            log_scale_indices = list(log_scale_indices)
        except Exception:
            raise ValueError("log_scale_indices must be a sequence of integers (or None).")
    # validate indices
    for idx in log_scale_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= dim:
            raise ValueError(f"log_scale_indices must contain integers in [0, {dim-1}]. Got: {idx}")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # First dimension i=0
    start, end = tup_lst[0]
    if 0 in log_scale_indices:
        if start <= 0 or end <= 0:
            raise ValueError("Log-scale sampling requires start>0 and end>0 for all log dimensions.")
        log_start = float(torch.log(torch.tensor(start, dtype=torch.float64)))
        log_end = float(torch.log(torch.tensor(end, dtype=torch.float64)))
        x = torch.rand((n, 1), generator=gen).to(dtype=dtype, device=device) * (log_end - log_start) + log_start
        result = torch.exp(x)
    else:
        result = torch.rand((n, 1), generator=gen).to(dtype=dtype, device=device) * (end - start) + start

    # Remaining dimensions
    for i in range(1, dim):
        if seed is not None:
            gen.manual_seed(seed + i)
        start, end = tup_lst[i]
        if i in log_scale_indices:
            if start <= 0 or end <= 0:
                raise ValueError("Log-scale sampling requires start>0 and end>0 for all log dimensions.")
            log_start = float(torch.log(torch.tensor(start, dtype=torch.float64)))
            log_end = float(torch.log(torch.tensor(end, dtype=torch.float64)))
            x = torch.rand((n, 1), generator=gen).to(dtype=dtype, device=device) * (log_end - log_start) + log_start
            col = torch.exp(x)
        else:
            col = torch.rand((n, 1), generator=gen).to(dtype=dtype, device=device) * (end - start) + start
        result = torch.cat([result, col], dim=1)

    return result



def create_training_data(particle, n, dimension, position_tuples, velocity_tuples, seed, data_path='', dtype=torch.float32, device=None):
    # If data_path is provided and exists, load the existing data
    if data_path and data_path.strip():
        import os
        if os.path.exists(data_path):
            print(f"Loading existing training data from {data_path}")
            try:
                data = torch.load(data_path, map_location=device if device else 'cpu')
                # Ensure data is in correct format [state, diff_state]
                if isinstance(data, list) and len(data) == 2:
                    state = data[0].to(dtype=dtype, device=device) if device else data[0].to(dtype=dtype)
                    diff_state = data[1].to(dtype=dtype, device=device) if device else data[1].to(dtype=dtype)
                    print(f"Loaded {state.shape[0]} samples from {data_path}")
                    return [state, diff_state]
                elif isinstance(data, list) and len(data) == 1: # correct format: [state]. Easier to use for already defined physical systems.
                    state = data[0].to(dtype=dtype, device=device) if device else data[0].to(dtype=dtype)
                    diff_state = analytically_differentiated_state(state, 0, particle.solve_acceleration).clone().detach()
                    print(f"Loaded {state.shape[0]} samples from {data_path} and computed the accelerations inside the workflow.")
                    #print('diff_state DIAGNOSE:', diff_state)
                    return [state, diff_state]
                else:
                    print(f"Warning: Data format unexpected in {data_path}. Generating new data.")
            except Exception as e:
                print(f"Error loading data from {data_path}: {e}. Generating new data.")
        else:
            print(f"Warning: Data path {data_path} does not exist. Generating new data.")
    
    # Make a torch tensor with n random starting positions and velocities using uniform distribution:
    # Merge them into a single tensor.
    pos_log_scale_indices = getattr(particle, 'position_log_scale_indices', None)
    vel_log_scale_indices = getattr(particle, 'velocity_log_scale_indices', None)
    
    state = torch.cat([random_initialize(position_tuples, dimension, n, seed, dtype, device, log_scale_indices=pos_log_scale_indices), 
                       random_initialize(velocity_tuples, dimension, n, seed*2, dtype, device, log_scale_indices=vel_log_scale_indices)], dim=1)
    if hasattr(particle, 'enforce_constraints'):
        if any(s != 1.0 for s in particle.scale):
            print("Warning: Scaling other than 1 detected while calling enforce_constraints.")
        state = particle.enforce_constraints(state)
    
    diff_state = analytically_differentiated_state(state, 0, particle.solve_acceleration).clone().detach()
    
    if (device != None):
        return [state.to(dtype=dtype, device=device), diff_state.to(dtype=dtype, device=device)]
    
    return [state, diff_state]

def create_validation_data(particle, dtype=torch.float32, device=None, 
                          position_bounds_override=None, 
                          velocity_bounds_override=None,
                          num_samples_override=None, validation_data_path=''):
    """
    Create validation data with optional custom bounds and sample count.
    
    Args:
        particle: Physical system instance
        dtype: Data type for tensors
        device: Device to place tensors on
        position_bounds_override: Optional custom position bounds
        velocity_bounds_override: Optional custom velocity bounds  
        num_samples_override: Optional custom number of samples
    
    Returns:
        List of [state_tensor, diff_state_tensor] for validation
    """
    # Use override bounds if provided, otherwise training bounds
    position_bounds = position_bounds_override or particle.train_hyperparams['position_bounds']
    velocity_bounds = velocity_bounds_override or particle.train_hyperparams['velocity_bounds']
    
    # Standard validation size (20% of training) unless overridden
    train_samples = particle.train_hyperparams['num_samples']
    num_samples = num_samples_override or max(100, train_samples // 5)  # At least 100 samples, or 20% of training
    
    # Different seed from training
    val_seed = particle.train_hyperparams['train_seed'] + 999
    
    dimension = particle.dof // 2

    scale_factor = particle.scale
    particle.scale_constants([1.0 for _ in range(particle.dof//2)])

    r = create_training_data(particle, num_samples, dimension,
                               position_bounds, velocity_bounds, seed=val_seed,
                               dtype=dtype, device=device, data_path=validation_data_path) # updated for custom validation
    
    particle.scale_constants(scale_factor)

    return r