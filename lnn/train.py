# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import torch
import importlib
from torch.utils.data import TensorDataset, DataLoader
from dataset_creation import create_training_data, normalize_training_data, create_validation_data
from variance_preserving_initializations import initialize_model_weights
def get_lnn_class(experiment_name=None):
    """
    Dynamically import the appropriate LNN class based on experiment name.
    
    Args:
        experiment_name: Name of experiment (e.g., "DyT", "standard", etc.)
        
    Returns:
        LNN class from the appropriate module
    """
    if experiment_name is None or experiment_name.lower() == "lnn":
        module_name = "LNN"
    else:
        module_name = f"{experiment_name}_LNN"
    
    # First try to import from architectures folder
    try:
        arch_module_name = f"architectures.{module_name}"
        module = importlib.import_module(arch_module_name)
        return getattr(module, 'LNN')
    except (ImportError, AttributeError):
        pass
    
    # Fall back to root directory
    try:
        module = importlib.import_module(module_name)
        return getattr(module, 'LNN')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import LNN from {module_name} or architectures.{module_name}, falling back to standard LNN")
        from LNN import LNN
        return LNN


def train_from_scratch(particle, save_path, lambda_val, dtype=torch.float32, device=None, training_data_path='', validation_data_path='',
                      validation_position_bounds=None, validation_velocity_bounds=None, normalize=True,
                      experiment_name=None):
    # Get the appropriate LNN class based on experiment name
    LNN = get_lnn_class(experiment_name)
    
    # Override normalize flag if experiment name contains "unnormalized"
    if experiment_name and "unnormalized" in experiment_name.lower():
        normalize = False
        print(f"Setting normalize=False for experiment: {experiment_name}")
    
    total_data_points = particle.train_hyperparams['num_samples']
    position_start_end = particle.train_hyperparams['position_bounds']
    velocity_start_end = particle.train_hyperparams['velocity_bounds']
    train_seed = particle.train_hyperparams['train_seed']
    training_data = create_training_data(particle, total_data_points, particle.dof//2, 
                                        position_start_end, velocity_start_end, seed=train_seed, data_path=training_data_path,
                                        dtype=torch.float32, device=device)
    
    eps = particle.train_hyperparams['total_epochs']
    batch_size = particle.train_hyperparams['minibatch_size']
    num_layers = particle.model_params['num_layers']
    activation_fn = particle.model_params['activation_fn']
    hidden_dim = particle.model_params['hidden_dim']
    lr = particle.train_hyperparams['lr']
    if normalize:
        scale_factor = None
    if not normalize:
        scale_factor = particle.scale
    scale_factor, training_data = normalize_training_data(training_data, particle.dof//2, scale_factor=scale_factor) # this scale_factor is coming from the bounds, so i gotta change that.
    particle.scale_constants(scale_factor)
    q_qdot_train, qdot_qdotdot_train = training_data

    num_particles = particle.dof // 4
    enforce_mass_matrix = getattr(particle, "enforce_mass_matrix", False)
    #masses = [particle.m] * num_particles if enforce_mass_matrix else None
    
    if enforce_mass_matrix:
        # physical mass per particle
        m = particle.m
        num_particles = particle.dof // 4   # //4 for usual 2D 3BODY (x,y,vx,vy) per particle → 12 // 4 = 3

        # expand physical masses to per-velocity
        # [m1, m1, m2, m2, m3, m3]
        m_per_vel = torch.tensor(
            [m for _ in range(num_particles) for _ in range(2)], # 2 for usual 2D 3BODY
            dtype=torch.float32, device=device,
        )

        # velocity scale vector (length = 6)
        s = torch.tensor(particle.scale, dtype=torch.float32, device=device)

        # bake scale^2 into masses (normalized-coordinate mass matrix)
        masses = (m_per_vel * s**2).tolist()
    else:
        masses = None

    # NOTE:
    # masses are in normalized velocity coordinates:
    #   M_norm = m_phys * scale^2
    print("Mass target (normalized):", masses)
    print("Expected analytic diag:", torch.tensor(particle.scale)**2 * particle.m)


    model = LNN(particle.dof, num_layers, activation_fn, hidden_dim, device, dtype, enforce_mass_matrix=enforce_mass_matrix, masses=masses) ## TODO: add the particle.constants here for mass enforcement?

    # Apply custom initialization if experiment name contains "initializ"
    if experiment_name and "initializ" in experiment_name.lower():
        # Choose your preferred pass type here. 'backward' is your reasoned choice
        # for the Hessian-based network.
        chosen_pass_type = 'backward' 
        
        initialize_model_weights(model, pass_type=chosen_pass_type)
        print(f"Finished custom dynamic initialization for experiment: {experiment_name}")
    
    #if torch.cuda.device_count() > 1:
    #    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)
    
    model.to(device, dtype)
    
    train_ds = TensorDataset(q_qdot_train, qdot_qdotdot_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Create validation data and loader
    validation_data = create_validation_data(
        particle, dtype, device,
        position_bounds_override=validation_position_bounds,
        velocity_bounds_override=validation_velocity_bounds,
        num_samples_override=None,
        validation_data_path=validation_data_path
    )
    
    # Apply same normalization to validation data
    _, validation_data_normalized = normalize_training_data(validation_data, particle.dof//2, scale_factor=scale_factor)
    q_qdot_val, qdot_qdotdot_val = validation_data_normalized
    
    val_ds = TensorDataset(q_qdot_val, qdot_qdotdot_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    #optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-6, foreach=False) (Float64 fix)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-6, foreach=True)
    eta_min = particle.train_hyperparams['eta_min']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eps//10, eta_min=eta_min)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    #print('val_loader is intentionally being set to None') # TEMPORARY EDIT
    #val_loader = None
    history = model.fit(eps, train_loader, optimizer, scheduler, val_loader=val_loader, ckpt_path=save_path, lambda_penalty=lambda_val)
    return model, history

def train_from_imported(particle, imported_model, lambda_val, save_path, dtype, device, training_data_path='', has_scheduler=True,
                       validation_position_bounds=None, validation_velocity_bounds=None):

    total_data_points = particle.train_hyperparams['num_samples']
    position_start_end = particle.train_hyperparams['position_bounds']
    velocity_start_end = particle.train_hyperparams['velocity_bounds']
    train_seed = particle.train_hyperparams['train_seed'] + 111
    scale_factor = particle.scale
    particle.scale_constants([1.0 for _ in range(particle.dof//2)])
    training_data = create_training_data(particle, total_data_points, particle.dof//2, position_start_end, velocity_start_end, seed=train_seed, data_path=training_data_path, dtype=dtype, device=device)
    _, training_data = normalize_training_data(training_data, particle.dof//2, scale_factor=scale_factor)
    particle.scale_constants(scale_factor)

    q_qdot_train, qdot_qdotdot_train = training_data

    lr = particle.train_hyperparams['lr']
    eps = particle.train_hyperparams['total_epochs']

    batch_size = particle.train_hyperparams['minibatch_size']
    train_ds = TensorDataset(q_qdot_train, qdot_qdotdot_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Create validation data and loader
    validation_data = create_validation_data(
        particle, dtype, device,
        position_bounds_override=validation_position_bounds,
        velocity_bounds_override=validation_velocity_bounds,
        num_samples_override=None
    )
    
    # Apply same normalization to validation data
    _, validation_data_normalized = normalize_training_data(validation_data, particle.dof//2, scale_factor=scale_factor)
    q_qdot_val, qdot_qdotdot_val = validation_data_normalized
    
    val_ds = TensorDataset(q_qdot_val, qdot_qdotdot_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.RAdam(imported_model.parameters(), lr=lr, weight_decay=1e-6, foreach=True)
    if has_scheduler:
        eta_min = particle.train_hyperparams['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eps//10, eta_min=eta_min)
    else:
        scheduler = None
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("Scale factor {}".format(scale_factor))
    history = imported_model.fit(eps, train_loader, optimizer, scheduler, val_loader=val_loader, ckpt_path=save_path, lambda_penalty=lambda_val)
    return imported_model, history

