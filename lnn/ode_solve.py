# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import numpy as np
from scipy.integrate import odeint, solve_ivp
import torch
import torchode as to

def nn_solve_ode(particle, model, x0, t, dtype=torch.float32, device=None):
    if device == None:
        device = torch.device('cpu')
    x0 = x0.cpu().detach().numpy()
    
    # Store original model dtype for restoration
    original_dtype = next(model.parameters()).dtype
    
    # Temporarily convert model to specified dtype for ODE integration
    model = model.to(dtype=dtype)
    
    def f(x, t):
        for j in particle.angle_indices:
            x[j] = x[j] % (2*np.pi/particle.scale[j])
        #print(f'Input:{x} ', end="")
        x_tor = torch.tensor(np.expand_dims(x, 0), requires_grad=True, dtype=dtype)
        out = np.squeeze(model(x_tor.to(device, dtype)).cpu().detach().numpy(), axis=0)
        #print(f'Output:{out}')
        return out
    
    result = odeint(f, x0, t, rtol=1e-10, atol=1e-10)
    
    # Restore original model dtype
    model = model.to(dtype=original_dtype)
    
    return result 

def solve_with_lnn_robust(particle, model, x0, t_span, t_eval, dtype=torch.float32, device=None):
    if device == None:
        device = torch.device('cpu')
    
    # Check GPU usage
    is_cuda = device.type == 'cuda'
    if is_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
        print(f"Model is on device: {next(model.parameters()).device}")
    
    x0 = x0.cpu().detach().numpy()
    
    # Store original model dtype for restoration
    original_dtype = next(model.parameters()).dtype
    model = model.to(device=device, dtype=dtype)  # Ensure model is on correct device
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    # Pre-allocate tensors on GPU to avoid repeated transfers
    x_buffer = torch.empty(x0.shape, dtype=dtype, device=device)
    out_buffer = np.empty(x0.shape, dtype=np_dtype)

    angle_idx = torch.tensor(particle.angle_indices, dtype=torch.long, device=device)
    scale_t = torch.tensor(particle.scale, dtype=dtype, device=device)
    modulus_t = 2 * torch.pi / scale_t

    def f(t, x):
        # Transfer to GPU once
        x_buffer.copy_(torch.from_numpy(x).to(device, dtype=dtype))

        # Process on GPU
        with torch.no_grad():
            x_buffer[angle_idx] = torch.remainder(x_buffer[angle_idx], modulus_t[angle_idx])
        
        # Model inference on GPU
        model_output = model(x_buffer.unsqueeze(0)).squeeze(0)
        
        # Single transfer back to CPU
        out_buffer[:] = model_output.cpu().detach().numpy()
        return out_buffer

    if is_cuda:
        print(f"GPU memory before ODE solving: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")

    # Use solve_ivp with a stiff solver like 'Radau' or 'BDF'
    sol = solve_ivp(
        fun=f,
        t_span=t_span,
        y0=x0,
        method='Radau',
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-8
    )
    
    if is_cuda:
        print(f"GPU memory after ODE solving: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MB")
    
    # solve_ivp returns the solution in a different format
    if sol.success:
        result = sol.y.T
    else:
        print(f"Solver failed: {sol.message}")
        result = np.zeros((len(t_eval), len(x0)))
    
    # Restore original model dtype
    model = model.to(dtype=original_dtype)
    
    if is_cuda:
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    
    return result



def solve_with_lnn_torchode(particle, model, x0, t_span, t_eval, dtype=torch.float32, device=None, max_batch_size=None, num_time_splits=None):
    if device is None:
        device = torch.device('cpu')
    original_dtype = next(model.parameters()).dtype
    
    # Check GPU usage
    is_cuda = device.type == 'cuda'
    if is_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
        print(f"Model is on device: {next(model.parameters()).device}")
   
    # Convert inputs to torch tensors on device
    x0_torch = x0.to(device=device, dtype=dtype)
    t_eval_torch = t_eval.to(device=device, dtype=dtype)
    
    # Handle batch dimensions properly
    if x0_torch.dim() == 1:
        # Single initial condition - add batch dimension
        x0_torch = x0_torch.unsqueeze(0)
        batch_size = 1
    else:
        # Multiple initial conditions (batch processing)
        batch_size = x0_torch.shape[0]
    
    # Apply time-splitting if specified from the start
    if num_time_splits is not None and num_time_splits > 1:
        print(f"Time-splitting approach: dividing time into {num_time_splits} segments")
        
        # Split time points into segments
        n_time_points = len(t_eval_torch)
        points_per_segment = n_time_points // num_time_splits
        
        all_results = []
        current_x0 = x0_torch.clone()
        
        for split_idx in range(num_time_splits):
            # Define time segment
            start_idx = split_idx * points_per_segment
            if split_idx == num_time_splits - 1:
                # Last segment gets all remaining points
                end_idx = n_time_points
            else:
                end_idx = (split_idx + 1) * points_per_segment + 1  # +1 for overlap
                
            t_segment = t_eval_torch[start_idx:end_idx]
            t_span_segment = (t_segment[0].item(), t_segment[-1].item())
            
            print(f"Solving time segment {split_idx + 1}/{num_time_splits}: t = {t_span_segment[0]:.3f} to {t_span_segment[1]:.3f}")
            print(f"Time points: {len(t_segment)}, Batch size: {batch_size}")
            
            # Clear GPU memory before each segment
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print(f"GPU memory before segment {split_idx + 1}: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
            
            # Solve this time segment (recursive call without time splitting)
            segment_result = solve_with_lnn_torchode(particle, model, current_x0, 
                                                   t_span_segment, t_segment, 
                                                   dtype=dtype, device=device, 
                                                   max_batch_size=max_batch_size, num_time_splits=None)
            
            # Convert to torch tensor for processing
            if not torch.is_tensor(segment_result):
                segment_result = torch.tensor(segment_result, device=device, dtype=dtype)
            
            # Store result (exclude overlap point except for first segment)
            if split_idx == 0:
                # First segment: store all points
                all_results.append(segment_result)
            else:
                # Subsequent segments: skip first point to avoid duplication (it's the same as last point of previous segment)
                all_results.append(segment_result[:, 1:, :])
            
            # Update initial conditions for next segment (final state of current segment)
            current_x0 = segment_result[:, -1, :].clone()  # Take final state of all batches
            
            print(f"Segment {split_idx + 1} completed successfully")
            print(f"Segment result shape: {segment_result.shape}")
            print(f"Next initial condition shape: {current_x0.shape}")
            
            # Clear memory after each segment
            if device.type == 'cuda':
                del segment_result
                torch.cuda.empty_cache()
                print(f"GPU memory after segment {split_idx + 1}: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
        
        # Concatenate all segments along time dimension
        result = np.concatenate([seg.cpu().numpy() for seg in all_results], axis=1)
        print(f"Time-splitting completed. Final result shape: {result.shape}")
        
        # Restore original model dtype
        model = model.to(dtype=original_dtype)
        return result
    
    # Apply max_batch_size limit if specified
    if max_batch_size is not None and batch_size > max_batch_size:
        print(f"Chunking {batch_size} test cases into chunks of {max_batch_size}")
        # Process in chunks
        all_results = []
        for i in range(0, batch_size, max_batch_size):
            chunk_end = min(i + max_batch_size, batch_size)
            chunk_x0 = x0_torch[i:chunk_end]
            
            print(f"Processing chunk {i//max_batch_size + 1}/{(batch_size + max_batch_size - 1)//max_batch_size}: tests {i+1}-{chunk_end}")
            
            # Recursively call with chunk (will not trigger chunking again since chunk_size <= max_batch_size)
            chunk_result = solve_with_lnn_torchode(particle, model, chunk_x0, t_span, t_eval, 
                                                  dtype, device, max_batch_size=None, num_time_splits=num_time_splits)
            all_results.append(chunk_result)
            
            # Clear GPU cache between chunks
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate all chunk results
        result = np.concatenate(all_results, axis=0)
        
        # Restore original model dtype
        model = model.to(dtype=original_dtype)
        return result
    
    # t_eval should be replicated for each batch item: (batch_size, n_steps)
    if t_eval_torch.dim() == 1:
        t_eval_torch = t_eval_torch.unsqueeze(0).expand(batch_size, -1)
   
    # Store original model dtype for restoration
    
    model = model.to(device=device, dtype=dtype)
   
    # Pre-compute particle properties on GPU
    angle_idx = torch.tensor(particle.angle_indices, dtype=torch.long, device=device)
    scale_t = torch.tensor(particle.scale, dtype=dtype, device=device)
    modulus_t = 2 * torch.pi / scale_t
   
    def ode_func(t, x):
        """ODE function for torchode - must return torch tensor"""
        # x has shape (batch_size, state_dim)
        x_processed = x.clone()
        if len(angle_idx) > 0:
            with torch.no_grad():
                x_processed[:, angle_idx] = torch.remainder(x_processed[:, angle_idx],
                                                       modulus_t[angle_idx].unsqueeze(0))
        
        # Model inference on GPU
        model_output = model(x_processed)
           
        return model_output
   
    if is_cuda:
        print(f"GPU memory before ODE solving: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
   
    # Set up torchode solver
    term = to.ODETerm(ode_func)
    step_method = to.Dopri5(term=term)  # Back to Dopri5 for better speed
    
    # Use PIDController for stiff dynamics
    step_size_controller = to.PIDController(
        atol=1e-8,   # Relaxed from 1e-12 but still tight
        rtol=1e-6,   # Relaxed from 1e-10 for better speed
        pcoeff=0.2,  # Standard value
        icoeff=0.4,  # Slightly increased
        dcoeff=0.0,  # Removed for speed
        term=term
    )
   
    # Create solver and compile it
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    
    try:
        jit_solver = torch.compile(solver)
        print("Using torch.compile for solver optimization")
    except Exception as e:
        print(f"torch.compile failed, using regular solver: {e}")
        jit_solver = solver
   
    # Set up the initial value problem
    problem = to.InitialValueProblem(
        y0=x0_torch,
        t_eval=t_eval_torch
    )
   
    try:
        # Solve with compiled solver
        sol = jit_solver.solve(problem)
        
        if is_cuda:
            print(f"GPU memory after ODE solving: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
            print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MB")
        
        print(f"Solver stats: {sol.stats}")
        
        # Handle batch dimensions in output
        result_torch = sol.ys  # Shape: (batch_size, n_steps, state_dim)
        
        # Always keep batch dimension for consistency with test.py expectations
        result = result_torch.cpu().detach().numpy()
        
        print("TorchODE solver succeeded")
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU memory error: {e}")
            print("Attempting time-splitting fallback...")
            
            # Clear GPU memory before attempting fallback
            if device.type == 'cuda':
                print("Clearing GPU cache before fallback...")
                # Delete solver objects to free memory
                try:
                    del solver, jit_solver, problem, term, step_method, step_size_controller
                except:
                    pass
                # Delete local variables that might hold GPU memory
                try:
                    del x0_torch, t_eval_torch, angle_idx, scale_t, modulus_t
                except:
                    pass
                # Clear all cached GPU memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection multiple times
                import gc
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"GPU memory after aggressive cleanup: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
                
                # Wait for automatic cleanup
                print("Waiting 10 seconds for automatic memory cleanup...")
                import time
                time.sleep(10)
                torch.cuda.empty_cache()
                print(f"GPU memory after 10s wait: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
            
            # Try time-splitting fallback
            try:
                # Restore original model dtype before time-splitting
                model = model.to(dtype=original_dtype)
                return solve_with_lnn_torchode(particle, model, x0, t_span, t_eval, 
                                             dtype, device, max_batch_size=max_batch_size, 
                                             num_time_splits=2)
            except Exception as split_error:
                print(f"Time-splitting fallback also failed: {split_error}")
                raise split_error
        
        print(f"TorchODE solver failed: {e}")
        raise e
   
    # Restore original model dtype
    model = model.to(dtype=original_dtype)
   
    if is_cuda:
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
   
    return result
