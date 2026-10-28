# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import numpy as np
import torch
import os
import datetime
import time
import json
import pandas as pd


def save_particle(particle, save_path, overwrite=False):
    """
    Save particle parameters to JSON format.
    Saves all attributes including physical parameters, hyperparameters, and test parameters.
    """
    os.makedirs(save_path, exist_ok=True)
    
    json_filename = os.path.join(save_path, "particle.json")
    

    # Check if particle.json already exists and handle enumeration
    if not overwrite and os.path.exists(json_filename):
        print(f"Warning: Particle file {json_filename} already exists. Saving with enumeration.")
        print("Note: Load functions will always load the default 'particle.json', not enumerated versions.")
        counter = 1
        base_name = os.path.join(save_path, "particle")
        while os.path.exists(f"{base_name}_{counter}.json"):
            counter += 1
        json_filename = f"{base_name}_{counter}.json"
        print(f"Note: File enumerated and saved as {json_filename}")

    print(f"Saving particle as {json_filename}")

    # Create dictionary with all particle attributes
    particle_data = {
        'class_name': particle.__class__.__name__,
        'dof': particle.dof,
        'scale': particle.scale,
    }
    
    # Physical parameters - get from param_names
    physical_params = {}
    for param_name in particle.param_names:
        physical_params[param_name] = getattr(particle, param_name)
    
    particle_data['physical_params'] = physical_params
    
    # Convert numpy/torch tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Save hyperparameters and test parameters
    if hasattr(particle, 'train_hyperparams'):
        # Handle activation function separately since it's not JSON serializable
        train_params = particle.train_hyperparams.copy()
        particle_data['train_hyperparams'] = convert_for_json(train_params)
    
    if hasattr(particle, 'model_params'):
        model_params = particle.model_params.copy()
        # Convert activation function to string representation
        if 'activation_fn' in model_params:
            model_params['activation_fn'] = str(model_params['activation_fn'])
        particle_data['model_params'] = convert_for_json(model_params)
    
    if hasattr(particle, 'test_params'):
        # Skip toy data parameters - they can be regenerated
        test_params_to_save = {}
        skip_params = {'toy_position', 'toy_velocity', 'toy_time_dataset'}
        for key, value in particle.test_params.items():
            if key not in skip_params:
                test_params_to_save[key] = value
        particle_data['test_params'] = convert_for_json(test_params_to_save)
    
    # Save angle_indices if available
    if hasattr(particle, 'angle_indices'):
        particle_data['angle_indices'] = particle.angle_indices
    
    # Save to JSON file
    with open(json_filename, 'w') as f:
        json.dump(particle_data, f, indent=2)
    
    print(f"Particle parameters saved to {json_filename}")
    return json_filename


def load_particle(particle, load_path):
    """
    Load particle parameters from JSON format.
    Updates the existing particle object with loaded parameters.
    """
    if os.path.isdir(load_path):
        # It's a directory, look for particle.json inside
        json_filename = os.path.join(load_path, "particle.json")
    else:
        # It's a file path, use as is
        json_filename = load_path
    
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"Particle file {json_filename} not found")
    
    # Load JSON data
    with open(json_filename, 'r') as f:
        particle_data = json.load(f)
    
    # Update physical parameters
    if 'physical_params' in particle_data:
        for param_name, param_value in particle_data['physical_params'].items():
            if hasattr(particle, param_name):
                setattr(particle, param_name, param_value)
                print(f"Loaded {param_name} = {param_value}")
    
    # Update scale
    if 'scale' in particle_data:
        particle.scale = particle_data['scale']
    
    # Update hyperparameters
    if 'train_hyperparams' in particle_data:
        for key, value in particle_data['train_hyperparams'].items():
            if key in particle.train_hyperparams:
                particle.train_hyperparams[key] = value
    
    if 'model_params' in particle_data:
        for key, value in particle_data['model_params'].items():
            if key in particle.model_params and key != 'activation_fn':  # Skip activation_fn as it's not directly loadable
                particle.model_params[key] = value
    
    if 'test_params' in particle_data:
        for key, value in particle_data['test_params'].items():
            if key in particle.test_params:
                particle.test_params[key] = value
    
    # Update angle_indices if available
    if 'angle_indices' in particle_data:
        particle.angle_indices = particle_data['angle_indices']
    
    print(f"Particle parameters loaded from {json_filename}")
    return particle

def save_model(model, save_path, overwrite=False):
    os.makedirs(save_path, exist_ok=True)

    pt_filename = os.path.join(save_path, "model.pt")
    
    # Check if model.pt already exists and handle enumeration
    if not overwrite and os.path.exists(pt_filename):
        print(f"Warning: Model file {pt_filename} already exists. Saving with enumeration.")
        print("Note: Load functions will always load the default 'model.pt', not enumerated versions.")
        counter = 1
        base_name = os.path.join(save_path, "model")
        while os.path.exists(f"{base_name}_{counter}.pt"):
            counter += 1
        pt_filename = f"{base_name}_{counter}.pt"
        print(f"Note: File enumerated and saved as {pt_filename}")
    
    print(f"Saving model as {pt_filename}")
    
    return torch.save(model, pt_filename)


def load_model(load_path, device, dtype, initialized_model=None):
    """
    Load model from .pt file or directory containing model.pt
    """
    if os.path.isdir(load_path):
        # It's a directory, look for model.pt inside
        model_filename = os.path.join(load_path, "model.pt")
    else:
        # It's a file path, use as is
        model_filename = load_path
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file {model_filename} not found")
    
    try:
        model = torch.load(model_filename, map_location=device, weights_only=False)
        model.eval()
    except:
        if initialized_model is not None:
            state_dict = torch.load(model_filename, map_location=device, weights_only=True)
            initialized_model.load_state_dict(state_dict)
            model = initialized_model
        else:
            raise ValueError("No initialized model provided and failed to load model from path.")
    return model.to(device, dtype)

def save_trajectory_test_results(test_data, nn_test_list, q_qdot_test_list, t_test, save_dir): # qdot_qdotdot_test_list,
    """
    Save test results and ground truth to a file for later plotting.
    Saves as .npz (numpy compressed) with a timestamp in the filename.
    """
    # Create trajectory_test_results subdirectory
    trajectory_dir = os.path.join(save_dir, "trajectory_test_results")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"trajectory_test_results_{now}.npz"
    filepath = os.path.join(trajectory_dir, filename)

    # Convert tensors to numpy arrays for saving
    test_data_np = test_data.detach().cpu().numpy() if torch.is_tensor(test_data) else np.array(test_data)
    t_test_np = t_test.detach().cpu().numpy() if torch.is_tensor(t_test) else np.array(t_test)
    nn_test_np = [x if isinstance(x, np.ndarray) else x.detach().cpu().numpy() for x in nn_test_list]
   # qd_qdotdot_test_np = [x if isinstance(x, np.ndarray) else x.detach().cpu().numpy() for x in qdot_qdotdot_test_list]
    gt_test_np = [x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x) for x in q_qdot_test_list]

    np.savez_compressed(
        filepath,
        test_data=test_data_np,
        t_test=t_test_np,
        nn_test_list=nn_test_np,
        q_qdot_test_list=gt_test_np,
     #   qdot_qdotdot_test_list=qd_qdotdot_test_np
    )
    print(f"Test results saved to {filepath}")

def load_trajectory_test_results(load_path):
    """
    Load test results from saved files. Compatible with directory structure.
    
    Args:
        load_path (str): Path to directory containing trajectory results or direct path to .npz file
        
    Returns:
        tuple: (q_qdot_test_list, nn_test_list, qdot_qdotdot_test_list, t_test)
    """
    import glob
    
    npz_file_path = None
    
    if os.path.isdir(load_path):
        # It's a directory, look for trajectory results inside
        # First check for trajectory_test_results subdirectory
        trajectory_dir = os.path.join(load_path, "trajectory_test_results")
        if os.path.exists(trajectory_dir):
            # Look for .npz files in the trajectory_test_results subdirectory
            npz_files = glob.glob(os.path.join(trajectory_dir, "trajectory_test_results_*.npz"))
            if npz_files:
                # Use the most recent file (sorted by filename which includes timestamp)
                npz_file_path = sorted(npz_files)[-1]
            else:
                # Look for any .npz file in trajectory_test_results
                npz_files = glob.glob(os.path.join(trajectory_dir, "*.npz"))
                if npz_files:
                    npz_file_path = npz_files[0]
        
        # If not found in subdirectory, look in the main directory
        if npz_file_path is None:
            npz_files = glob.glob(os.path.join(load_path, "trajectory_test_results_*.npz"))
            if npz_files:
                npz_file_path = sorted(npz_files)[-1]
            else:
                # Look for any .npz file
                npz_files = glob.glob(os.path.join(load_path, "*.npz"))
                if npz_files:
                    npz_file_path = npz_files[0]
    elif load_path.endswith('.npz'):
        # It's a direct path to an .npz file
        npz_file_path = load_path
    else:
        # Try appending .npz extension
        npz_candidate = load_path + '.npz'
        if os.path.exists(npz_candidate):
            npz_file_path = npz_candidate
    
    if npz_file_path is None or not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"No trajectory test results found in {load_path}")
    
    print(f"Loading trajectory data from: {npz_file_path}")
    
    # Load the .npz file
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Extract data - handle both old and new formats
    if 'test_data' in data:
        # Old format compatibility
        test_data = data['test_data']
        t_test = data['t_test']
        nn_test_list = list(data['nn_test_list'])
        q_qdot_test_list = list(data['q_qdot_test_list'])
        
        # Keep data as numpy arrays - no torch conversion
        
        # Create derivatives placeholder (qdot_qdotdot_test_list)
        qdot_qdotdot_test_list = [np.zeros_like(trajectory) for trajectory in q_qdot_test_list]
        
        print(f"Loaded {len(q_qdot_test_list)} trajectory test cases from npz format")
        print(f"Trajectory shape: {q_qdot_test_list[0].shape}")
        print(f"Time steps: {len(t_test)}")
        
        return q_qdot_test_list, nn_test_list, qdot_qdotdot_test_list, t_test
    
    else:
        # Try to handle other possible formats in the npz file
        keys = list(data.keys())
        print(f"Available keys in npz file: {keys}")
        
        # Try to intelligently map keys to our expected format
        q_qdot_test_list = []
        nn_test_list = []
        qdot_qdotdot_test_list = []
        t_test = None
        
        for key in keys:
            value = data[key]
            if 'time' in key.lower() or 't_' in key.lower():
                t_test = value  # Keep as numpy array
            elif 'analytical' in key.lower() or 'ground_truth' in key.lower() or 'gt' in key.lower():
                q_qdot_test_list = list(value)  # Keep as numpy arrays
            elif 'predicted' in key.lower() or 'nn' in key.lower():
                nn_test_list = list(value)  # Keep as numpy arrays
            elif 'deriv' in key.lower():
                qdot_qdotdot_test_list = list(value)  # Keep as numpy arrays
        

        if not qdot_qdotdot_test_list and q_qdot_test_list:
            qdot_qdotdot_test_list = [np.zeros_like(traj) for traj in q_qdot_test_list]
        
        if not q_qdot_test_list:
            raise ValueError(f"Could not extract trajectory data from {npz_file_path}")
        
        print(f"Loaded {len(q_qdot_test_list)} trajectory test cases (generic npz format)")
        return q_qdot_test_list, nn_test_list, qdot_qdotdot_test_list, t_test

def save_model_and_particle(particle, model, save_dir, overwrite=False):
    """
    Save the model and particle parameters to the specified directory.
    """
    save_model(model, save_dir, overwrite=overwrite)
    save_particle(particle, save_dir, overwrite=overwrite)

def save_history(history, save_path, overwrite=False):
    """
    Save training history to CSV format.
    Automatically finds and saves best epoch and best loss.
    """
    os.makedirs(save_path, exist_ok=True)
    
    csv_filename = os.path.join(save_path, "history.csv")
    
    # Check if history.csv already exists and handle enumeration
    if not overwrite and os.path.exists(csv_filename):
        print(f"Warning: History file {csv_filename} already exists. Saving with enumeration.")
        print("Note: Load functions will always load the default 'history.csv', not enumerated versions.")
        counter = 1
        base_name = os.path.join(save_path, "history")
        while os.path.exists(f"{base_name}_{counter}.csv"):
            counter += 1
        csv_filename = f"{base_name}_{counter}.csv"
        print(f"Note: File enumerated and saved as {csv_filename}")

    print(f"Saving history as {csv_filename}")

    # Convert history to DataFrame
    df_data = {}
    
    # Add epoch numbers
    if 'train_loss' in history and len(history['train_loss']) > 0:
        epochs = list(range(1, len(history['train_loss']) + 1))
        df_data['epoch'] = epochs
    
    # Add all history data
    for key, values in history.items():
        if isinstance(values, (list, np.ndarray, torch.Tensor)):
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            df_data[key] = values
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Find best epoch and loss - prioritize validation loss if available, otherwise use training loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        # Determine which loss to use for best epoch selection
        if 'val_loss' in history and len(history['val_loss']) > 0:
            # Use validation loss for best epoch determination
            losses_for_best = history['val_loss']
            best_loss = min(losses_for_best)
            best_epoch = losses_for_best.index(best_loss) + 1
            loss_type = 'validation'
        else:
            # Fall back to training loss
            losses_for_best = history['train_loss']
            best_loss = min(losses_for_best)
            best_epoch = losses_for_best.index(best_loss) + 1
            loss_type = 'training'
        
        print(f"Best epoch: {best_epoch}, Best {loss_type} loss: {best_loss:.6f}")
        
        # Save metadata to a separate file
        metadata = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_loss_type': loss_type,
            'total_epochs': len(history['train_loss']),
            'save_timestamp': datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        
        metadata_filename = csv_filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_filename}")
    
    # Save to CSV file
    df.to_csv(csv_filename, index=False)
    
    print(f"Training history saved to {csv_filename}")
    return csv_filename


def load_history(load_path):
    """
    Load training history from CSV format.
    
    Args:
        load_path (str): Path to directory containing history.csv or direct path to CSV file
        
    Returns:
        dict: Dictionary containing training history with keys like 'train_loss', 'lr', 'best_epoch', etc.
    """
    if os.path.isdir(load_path):
        # It's a directory, look for history.csv inside
        csv_filename = os.path.join(load_path, "history.csv")
        metadata_filename = os.path.join(load_path, "history_metadata.json")
    else:
        # It's a file path, use as is
        csv_filename = load_path
        metadata_filename = csv_filename.replace('.csv', '_metadata.json')
    
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"History file {csv_filename} not found")
    
    # Load CSV data
    df = pd.read_csv(csv_filename)
    
    # Convert DataFrame back to dictionary
    history_data = {}
    for column in df.columns:
        if column != 'epoch':  # Skip epoch column as it's just sequential
            history_data[column] = df[column].tolist()
    
    # Load metadata if available
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
        history_data.update(metadata)
        
        print(f"Training history loaded from {csv_filename}")
        print(f"Best epoch: {metadata['best_epoch']}, Best loss: {metadata['best_loss']:.6f}")
        print(f"Total epochs trained: {metadata['total_epochs']}")
    else:
        print(f"Training history loaded from {csv_filename} (no metadata found)")
    
    return history_data