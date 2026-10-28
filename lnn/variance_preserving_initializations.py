# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.

import numpy as np
import torch
import torch.nn as nn
import scipy.special as sp
import importlib # We need this for the dynamic import

# --- Part 1: Numerical Integration Core ---
# (This part is unchanged)
def gaussian_expectation(g, n=200):
    x, w = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * x
    vals = g(z)
    return (w @ vals) / np.sqrt(np.pi)

# --- Part 2: Activation Functions and Their Derivatives (in NumPy) ---
# (This part is unchanged)
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def d_softplus(x): return 1.0 / (1.0 + np.exp(-x))
def gelu_exact_fn(x): return 0.5 * x * (1.0 + sp.erf(x / np.sqrt(2.0)))
def d_gelu_exact_fn(x):
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + sp.erf(x / np.sqrt(2.0)))
    return cdf + x * pdf
def xtanhkx(x, k=1.0): return x * np.tanh(k * x)
def d_xtanhkx(x, k=1.0):
    t = np.tanh(k * x)
    return t + k * x * (1.0 - t**2)
def quadratic(x): return x**2
def d_quadratic(x): return 2.0 * x

# --- Part 3: On-the-Fly Gain Calculator ---
# (This part is unchanged)
def calculate_init_gain(activation_name: str, pass_type: str = 'forward', k: float = None) -> float:
    activation_name = activation_name.lower()
    if activation_name == 'linear': return 1.0
    func_map = {
        'softplus': (softplus, d_softplus),
        'gelu': (gelu_exact_fn, d_gelu_exact_fn),
        'xtanhkx': (lambda z: xtanhkx(z, k), lambda z: d_xtanhkx(z, k)),
        'quadratic': (quadratic, d_quadratic),
    }
    if activation_name not in func_map: raise ValueError(f"Unknown activation function '{activation_name}'.")
    f, df = func_map[activation_name]
    if pass_type == 'forward': c = gaussian_expectation(lambda z: f(z)**2)
    elif pass_type == 'backward': c = gaussian_expectation(lambda z: df(z)**2)
    else: raise ValueError(f"pass_type must be 'forward' or 'backward', but got '{pass_type}'")
    return 1.0 / np.sqrt(c + 1e-8)

# --- Part 4: Main Initializer Function (NEW DYNAMIC VERSION) ---

def initialize_model_weights(model: nn.Module, pass_type: str = 'forward'):
    """
    Initializes all linear layers by dynamically discovering the model's
    custom activation functions and structure.
    """
    print(f"Applying dynamic '{pass_type}' initialization...")

    # --- Step A: Find the core network module and its defined classes ---
    # This handles your DyT vs standard LNN structure
    if hasattr(model.network, 'combined_network'):
        network_to_init = model.network.combined_network # DyT case
    else:
        network_to_init = model.network # Standard case

    # Dynamically discover the module where custom activations are defined
    network_module_name = network_to_init.__class__.__module__
    try:
        network_module = importlib.import_module(network_module_name)
        # Safely get the custom classes if they exist in that module
        QuadraticSmoothActivation = getattr(network_module, 'QuadraticSmoothActivation', None)
        QuadraticActivation = getattr(network_module, 'QuadraticActivation', None)
    except ImportError:
        print(f"Warning: Could not import module {network_module_name}. Custom activations may not be found.")
        QuadraticSmoothActivation, QuadraticActivation = None, None

    # --- Step B: Build the activation map with the discovered classes ---
    activation_map = {
        nn.GELU: 'gelu',
        nn.Softplus: 'softplus',
    }
    if QuadraticSmoothActivation: activation_map[QuadraticSmoothActivation] = 'xtanhkx'
    if QuadraticActivation: activation_map[QuadraticActivation] = 'quadratic'
    
    with torch.no_grad():
        submodules_to_init = {}
        
        # Add MLP if it exists
        if hasattr(network_to_init, 'mlp'):
            submodules_to_init["MLP"] = network_to_init.mlp
        
        # Add QMLP if it exists (check for both qmlp and its nested qmlp)
        if hasattr(network_to_init, 'qmlp') and hasattr(network_to_init.qmlp, 'qmlp'):
            submodules_to_init["QMLP"] = network_to_init.qmlp.qmlp
        for name, submodule in submodules_to_init.items():
            if not isinstance(submodule, nn.Sequential): 
                continue
            last_activation = ""
            last_k = None
            for i, layer in enumerate(submodule):
                if isinstance(layer, nn.Linear):
                    # Check if there's a next layer to avoid IndexError
                    if i + 1 < len(submodule):
                        next_layer = submodule[i + 1]
                        activation_class = type(next_layer)
                        if activation_class in activation_map:
                            activation_name = activation_map[activation_class]
                            last_activation = activation_name
                            k = getattr(next_layer, 'k', None)
                            last_k = k
                            print(f"  - Initializing {name}[{i}] for a following '{activation_name}' (k={k})")
                        else:
                            activation_name, k = 'linear', None
                            print(f"  - WARNING: Unknown activation {activation_class} after {name}[{i}]. Using linear gain.")
                    else:
                        # Last layer in sequence - use the previous layer's initialization
                        activation_name, k = last_activation, last_k
                        print(f"  - Initializing {name}[{i}] (last layer) with {last_activation} gain (k={last_k})")

                    gain = calculate_init_gain(activation_name, pass_type=pass_type, k=k)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    std = gain / np.sqrt(fan_in)
                    nn.init.normal_(layer.weight, mean=0.0, std=std)
                    
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)