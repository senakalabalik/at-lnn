# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Written for this work on top of hessian_regularized_lnn (https://github.com/abdullahumuth/hessian_regularized_lnn). Apache-2.0.

import torch
from torch import ones, zeros
import torch.nn as nn
from torch.nn.functional import tanh
from torch.nn.parameter import Parameter
from torch.func import vmap, hessian, jacrev, functional_call
import os
import time
import shutil
from file_exports import save_model, save_history, load_model
import numpy as np


class CombinedNetwork(nn.Module):
    def __init__(self,
                 input_size: int, # we will assume input is always q_qdot, so input_size should be 2*n where n is the number of generalized coordinates. true even for the fixed kinetic version, since the input is still q_qdot, but the network only uses q. this is just to keep the interface consistent and simple.
                 num_layers: int,
                 activation_fn: nn.Module,
                 hidden_dim: int,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float32,
                 lj_fixed: bool = False, # NEW: if True, hardcode analytic LJ and have the MLP learn only the E0-scaled AT residual (V = V_LJ(analytic) + E0*mlp(features)). Default False reproduces plain fully-learned-V behavior.
                 E0: float = 0.1,        # NEW: only used when lj_fixed=True
                 eps_LJ: float = 1.0,    # NEW: only used when lj_fixed=True
                 sigma_LJ: float = 1.0, # NEW: only used when lj_fixed=True
                 features: str = "rcos"): # RELEASE: "rcos" = (r12,r23,r31,cos1,cos2,cos3) as in the paper; "r" = distances only (the 3-input Newtonian control model). The "r" branch is the line that was commented out below for that run.
        super().__init__()
        self.features = features
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.input_size = input_size
        self.n = input_size // 2
        self.lj_fixed = lj_fixed
        self.E0 = E0
        self.eps_LJ = eps_LJ
        self.sigma_LJ = sigma_LJ

        # Build the MLP part
        layers = []
        in_feat=input_size//2 # NOTE and NOTE NOTE NOTE: only q goes into the MLP, so input is halved. this is the main change for the fixed kinetic version.
        if features == "r":
            in_feat = 3
        #in_feat = 3 # NOTE NOTE: this is for the relative coordinates version. this is a test-run really, with the forward() below as well. we'll clean up later based on how the training goes.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_feat, hidden_dim)) 
            layers.append(activation_fn) # maybe change this into a factory (a new instance per layer, later on. rn it's the same one instance for each layer.)
            in_feat = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1, bias=False))
        self.mlp = nn.Sequential(*layers).to(self.device, dtype)
        # Build the quadratic term
        # using the same self.n so we know where q̇ starts
        self.a = 1.0/2.0 # hardcoded kinetic term coefficient (1/2) for T = 1/2 * a * q̇^2

        self.to(self.device, self.dtype)

    '''def forward(self, x: torch.Tensor): # completely fine, but let's try the one below.
        # 1) ensure batch-dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)               # [input_dim] → [1, input_dim]
        # 2) network output
        q = x[:, :self.n]
        qdot = x[:, self.n:]                 # [B, input_dim/2]
        mlp_out = self.mlp(q)                # q-only network  -> V term,
        quad_out = self.a * (qdot**2).sum(dim=1, keepdim=True) 
        out = - mlp_out + quad_out  # [B,1]  -> L = T - V, where mlp_out is the learned potential and quad_out is the fixed kinetic.
        return out.view(-1)'''  # safe flatten to [B] NOTE:
    
    def forward(self, x: torch.Tensor): # NOTE NOTE: NEW! try only feeding relative coordinates into the MLP. Rn it's only for threebody.
        if x.dim() == 1:
            x = x.unsqueeze(0)

        q = x[:, :self.n]
        qdot = x[:, self.n:]

        q = q.view(-1, 3, 2)

        r12 = torch.norm(q[:,0] - q[:,1], dim=1, keepdim=True)
        r23 = torch.norm(q[:,1] - q[:,2], dim=1, keepdim=True)
        r31 = torch.norm(q[:,2] - q[:,0], dim=1, keepdim=True)

        # r = torch.cat([r12, r23, r31], dim=1) NOTE NOTE version
        # V = self.mlp(r) NOTE NOTE version

        v12 = q[:,1] - q[:,0]
        v13 = q[:,2] - q[:,0]
        v23 = q[:,2] - q[:,1]

        cos1 = (v12 * v13).sum(dim=1, keepdim=True) / (r12 * r31 + 1e-6)
        cos2 = (-v12 * v23).sum(dim=1, keepdim=True) / (r12 * r23 + 1e-6)
        cos3 = (-v13 * -v23).sum(dim=1, keepdim=True) / (r31 * r23 + 1e-6)

        #features = torch.cat([torch.log(r12 + 1e-3), torch.log(r23 + 1e-3), torch.log(r31 + 1e-3), cos1, cos2, cos3], dim=1)
        features = torch.cat([r12, r23, r31, cos1, cos2, cos3], dim=1)
        if getattr(self, "features", "rcos") == "r":
            features = torch.cat([r12, r23, r31], dim=1)

        T = self.a * (qdot**2).sum(dim=1, keepdim=True)

        # the LJ FIXED VERSION: toggled via self.lj_fixed (set at construction time, defaults to False).
        # getattr(..., False) fallback keeps this safe for older pickled checkpoints that predate this attribute.
        if getattr(self, "lj_fixed", False):
            sigma, eps, E0 = self.sigma_LJ, self.eps_LJ, self.E0

            V_LJ12 = 4 * eps * ((sigma / r12)**12 - (sigma / r12)**6)
            V_LJ23 = 4 * eps * ((sigma / r23)**12 - (sigma / r23)**6)
            V_LJ31 = 4 * eps * ((sigma / r31)**12 - (sigma / r31)**6)

            V_LJ = V_LJ12 + V_LJ23 + V_LJ31

            V_AT = E0 * self.mlp(features)

            return (T - V_LJ - V_AT).view(-1)

        V = self.mlp(features) # NOTE NOTE NOTE
        return (T - V).view(-1)



class LNN(nn.Module):
    def __init__(self, input_size, num_layers, activation_fn, hidden_dim, device, dtype=torch.float32, enforce_mass_matrix=False, masses=None,
                 lj_fixed: bool = False, E0: float = 0.1, eps_LJ: float = 1.0, sigma_LJ: float = 1.0, # NEW: passthrough to CombinedNetwork's LJ-fixed toggle
                 features: str = "rcos"): # RELEASE: passthrough, see CombinedNetwork
        super(LNN, self).__init__()
        self.dtype = dtype
        if (device != None):
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.n = input_size // 2
#        self.dyt_layer = DyT(input_size).to(self.device, dtype)
        self.enforce_mass_matrix, self.masses = enforce_mass_matrix, masses
        self.network = CombinedNetwork(
            input_size, num_layers, activation_fn, hidden_dim, device, dtype,
            lj_fixed=lj_fixed, E0=E0, eps_LJ=eps_LJ, sigma_LJ=sigma_LJ, features=features
        ).to(self.device, self.dtype)
    
    def compute_loss(self, pred, target, inputs, lambda_penalty=100.0): # may increase lambda_base
        """
        Compute the total loss including both prediction loss and regularization penalty.
        
        Args:
            pred: Predicted qdot_qdotdot from forward pass
            target: Target qdot_qdotdot 
            inputs: Input q_qdot used to compute regularization
            lambda_penalty: Weight for the eigenvalue penalty term
            
        Returns:
            total_loss: Combined prediction loss + penalty
            loss_components: Dict with breakdown of loss components
        """
        # Base prediction loss
        base_loss = torch.mean(torch.abs(pred - target)) # torch.abs(scaled_pred - scaled_target)
        #base_loss = torch.mean(torch.abs(pred[:, self.n:] - target[:, self.n:])) # bc no need to compare q_dots..? doesnt change anything though.

        ##### MODIFIED LOSS FOR SCALE-AWARE ACCELERATION #####
        '''
        acc_t  = torch.sign(target)  * torch.log1p(torch.abs(target))
        pred_t = torch.sign(pred) * torch.log1p(torch.abs(pred))

        base_loss = torch.mean(torch.abs(pred_t - acc_t))  # MAE in log-space'''

        # reasoning:

        # this should act like absolute error for small values (where log1p(x) ~ x) and like relative error for large values (where log1p(x) ~ log(x)). the sign is preserved to allow learning of attractive vs repulsive forces. the 1p is to prevent log(0) issues.

        # we expect the small error contributions to stay ~identical to the plain MAE. 
        # but for the larger errors, now the loss depends on ratios, not the absolue difference. compresses their impact.

        # also gradients now scale like 1/|x| for large values, which should help prevent the huge gradients that can come from large errors in the original MAE. for small values, gradients are more like constant, which is fine.

        # so we're hoping this'll focus more on structure than magnitude extremes.

        ###### END OF MODIFIED LOSS ######


        # Regularization penalty (eigenvalue constraint)
        jac, hess = self.batch_jac_and_hes(self.network, inputs)
        A = hess[:, self.n:, self.n:]  # Mass matrix


        eigenvalues = torch.vmap(torch.linalg.eigvalsh)(A)
        penalty = torch.mean(torch.sum(torch.relu(-eigenvalues), dim=1))
        
        # Total loss
        total_loss = base_loss + lambda_penalty * penalty
        
        # Return components for logging/debugging
        loss_components = {
            'base_loss': base_loss.item(),
            'penalty': penalty.item(),
            'total_loss': total_loss.item()
        }

        #print("min eig A:", torch.linalg.eigvalsh(A).min().item(),
        #    "cond A:", torch.linalg.cond(A).max().item())
   
        return total_loss, loss_components, base_loss, penalty # tempo change for debugging the loss.

    def lagrangian(self, x):
        if (len(x.shape) < 2):
            x = x.unsqueeze(0)

#        dyt_x = self.dyt_layer(x)
#        return self.network(dyt_x)
        return self.network(x)

    def batch_jac_and_hes(self, model, inputs):
        params = dict(model.named_parameters())
    
        def func(params, x):
            return functional_call(model, params, x)
    
        def single_jac(x):
            return jacrev(func, argnums=1)(params, x)
    
        # Compute Jacobian
        jac = vmap(single_jac)(inputs)
        jac = jac.squeeze(1)
    
        # Compute Hessian
        def single_hes(x):
            return jacrev(single_jac)(x)
    
        hes = vmap(single_hes)(inputs)
        hes = hes.squeeze(1)
        
        return jac, hes

    def forward(self, x, t=0):
        if x.dim() < 2:
            x = x.unsqueeze(0)
        B, D = x.shape
        n = self.n
        qdot = x[:, n:]
    
        jac, hess = self.batch_jac_and_hes(self.network, x)
    
        A = hess[:, n:, n:]

        # --- DIAGNOSTIC ---
        # Calculate the condition number. High values mean it's close to singular.
        cond_nums = torch.linalg.cond(A)  # Returns tensor of shape [B]
        max_cond = cond_nums.max().item()
        if max_cond > 1e6:
            problematic_indices = (cond_nums > 1e6).nonzero(as_tuple=True)[0]
            print(f"WARNING: High condition numbers detected. Max: {max_cond:.2e}")
            print(f"Problematic batch indices: {problematic_indices.tolist()}")
        # ------------------
    
        Bmat = hess[:, n:, :n]
        C = jac[:, :n].unsqueeze(-1)
        q = qdot.unsqueeze(-1)
        rhs = C - torch.matmul(Bmat, q)
    
        try:
            Ap = torch.linalg.pinv(A)
        except torch.linalg.LinAlgError as e:
            cond_num = torch.linalg.cond(A).item()
            print(f"FATAL: LinAlgError during pinv! Condition number was {cond_num:.2e}. Error: {e}")
            # Return a zero tensor to prevent crashing the whole script,
            # but the trajectory will be wrong.
            return torch.zeros_like(x)
    
        qdd = torch.matmul(Ap, rhs).squeeze(-1)
        out = torch.cat([qdot, qdd], dim=1)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"FATAL: NaN/Inf in output! Condition number was {cond_num:.2e}. Input was {x.detach().cpu().numpy()}")
    
        if self.training:
            out.retain_grad()

        return out
    
    def fit(self, num_epochs_to_run, train_loader, optimizer, scheduler=None, 
            val_loader=None, gradient_clip_norm=1.0, ckpt_path='/kaggle/working/', lambda_penalty=1000.0):
        
        history = {'train_loss': [], 'val_loss': [], 'lr': [], 
                   'train_base_loss': [], 'train_penalty': [], 'val_base_loss': [], 'val_penalty': []} # 'train_diag_penalty': [], 'train_offdiag_penalty': [],
        if val_loader is None:
            # Remove validation-specific keys if no validation
            del history['val_loss'], history['val_base_loss'], history['val_penalty']
            
        print(f"Starting training for {num_epochs_to_run} epochs...") 
        initial_lr_display = optimizer.param_groups[0]['lr']
        print(f"Initial LR for this run: {initial_lr_display:.2e}, Batch Size: {train_loader.batch_size}")
        if val_loader is not None:
            print(f"Validation enabled with {len(val_loader.dataset)} samples")
        print(f"Regularization penalty weight: {lambda_penalty}")
        
        best_loss = float('inf')
        best_epoch = -1
        
        # Create temporary checkpoint directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_checkpoint_dir = os.path.join(ckpt_path, f"temp_checkpoint_{timestamp}")
        os.makedirs(temp_checkpoint_dir, exist_ok=True)


        
        #lam_eff = float(lambda_penalty)  # running effective lambda
        #frozen = False
   


        for epoch in range(1, num_epochs_to_run + 1):
            self.train() # Set model to training mode
            epoch_train_loss = 0.0
            epoch_train_base_loss = 0.0
            epoch_train_penalty = 0.0
            #epoch_train_penalty_diag = 0.0
            #epoch_train_penalty_offdiag = 0.0


            
            for batch_idx, (xi, yi) in enumerate(train_loader):
                # Move data to the same device as the model
                xi, yi = xi.to(self.device, self.dtype), yi.to(self.device, self.dtype)

                optimizer.zero_grad()
                qdot_qdotdot_pred = self(xi) 
                #torch.cuda.synchronize()

                
                # Use unified loss computation
                total_loss, loss_components, base_loss, penalty = self.compute_loss(qdot_qdotdot_pred, yi, xi, lambda_penalty) # temporary change
                #torch.cuda.synchronize()


                '''
                do_balance = (batch_idx % 100 == 0)  # or True to do every step
                alpha = 0.3                         # EMA update rate (0.05–0.2 are typical)
                lam_max = 100.0                     # safety clamp (pick a sane upper bound)
                if penalty.detach().item() < 1e-3:
                    do_balance= False
                    print('penalty reduced down < 1e-3')
                    if frozen==False: 
                        lam_eff= lam_eff / 10
                        frozen=True

                #do_balance = False # turn it off for diagnosis
                if do_balance:
                    # grad norm of base loss
                    optimizer.zero_grad(set_to_none=True)
                    base_loss.backward(retain_graph=True)
                    g_base = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf")).item()

                    # grad norm of penalty (UNWEIGHTED)
                    optimizer.zero_grad(set_to_none=True)
                    penalty.backward(retain_graph=True)
                    g_pen = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf")).item()

                    # target lambda so gradient magnitudes match
                    lam_target = float(lambda_penalty) * (g_base / (g_pen + 1e-12))

                    # EMA smooth + clamp !!
                    lam_eff = (1 - alpha) * lam_eff + alpha * lam_target
                    lam_eff = max(0.0, min(lam_eff, lam_max))

                    print(f"g_base={g_base:.3e} g_pen={g_pen:.3e} lam_target={lam_target:.3e} lam_eff={lam_eff:.3e}")'''
                # ramped cap (prevents big lambda early)
                #cap = lam_max * frac
                #lam_eff = max(0.0, min(lam_eff, cap))
                optimizer.zero_grad(set_to_none=True)
                #total_loss = base_loss + lam_eff * penalty


                total_loss.backward()
                
                if gradient_clip_norm is not None and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                
                optimizer.step()
                epoch_train_loss += loss_components['total_loss']
                epoch_train_base_loss += loss_components['base_loss']
                epoch_train_penalty += loss_components['penalty']



                baseline = yi.mean(dim=0, keepdim=True)
                model_mae = (qdot_qdotdot_pred  - yi).abs().mean().item()
                print("model_mae:", model_mae, "baseline_mae:",(baseline - yi).abs().mean().item() )





            
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            avg_train_base_loss = epoch_train_base_loss / len(train_loader)
            avg_train_penalty = epoch_train_penalty / len(train_loader)
            #avg_train_penalty_diag = epoch_train_penalty_diag / len(train_loader)
            #avg_train_penalty_offdiag = epoch_train_penalty_offdiag / len(train_loader)

            
            history['train_loss'].append(avg_epoch_train_loss)
            history['train_base_loss'].append(avg_train_base_loss)
            history['train_penalty'].append(avg_train_penalty)

            #history['train_diag_penalty'].append(avg_train_penalty_diag)
            #history['train_offdiag_penalty'].append(avg_train_penalty_offdiag)

            
            # Validation step
            avg_epoch_val_loss = None
            avg_val_base_loss = None  
            avg_val_penalty = None

            if val_loader is not None:
                self.eval()  # Set model to evaluation mode
                epoch_val_loss = 0.0
                epoch_val_base_loss = 0.0
                epoch_val_penalty = 0.0

                
                # Don't use torch.no_grad() because we need gradients for forward pass derivatives
                # But disable gradient accumulation for parameters to prevent memory buildup
                for param in self.parameters():
                    param.grad = None
                
                for val_xi, val_yi in val_loader:
                    val_xi, val_yi = val_xi.to(self.device, self.dtype), val_yi.to(self.device, self.dtype)
                    
                    # Forward pass needs gradients for derivative computation
                    val_pred = self(val_xi)
                    
                    # Use same unified loss computation as training
                    total_loss, loss_components, _,_ = self.compute_loss(val_pred, val_yi, val_xi, lambda_penalty)
                    
                    epoch_val_loss += loss_components['total_loss']
                    epoch_val_base_loss += loss_components['base_loss']
                    epoch_val_penalty += loss_components['penalty']

                    
                    # Clear any gradients that might have accumulated
                    for param in self.parameters():
                        param.grad = None
                
                avg_epoch_val_loss = epoch_val_loss / len(val_loader)
                avg_val_base_loss = epoch_val_base_loss / len(val_loader)
                avg_val_penalty = epoch_val_penalty / len(val_loader)

                
                history['val_loss'].append(avg_epoch_val_loss)
                history['val_base_loss'].append(avg_val_base_loss)
                history['val_penalty'].append(avg_val_penalty)

            
            # Log LR used for the epoch just completed (before scheduler.step())
            lr_for_epoch = optimizer.param_groups[0]['lr']
            history['lr'].append(lr_for_epoch)

            # Use validation loss for best model selection if available, otherwise training loss
            current_loss = avg_epoch_val_loss if avg_epoch_val_loss is not None else avg_epoch_train_loss
            #current_loss = avg_val_base_loss_unw if avg_val_base_loss_unw is not None else avg_epoch_train_loss
            # NOTE: the use of unw_base_loss for the validation choice is not yet completely decided.
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                
                # Save model to temporary checkpoint directory
                save_model(self, temp_checkpoint_dir, overwrite=True)
                
                # Save history up to this point
                save_history(history, temp_checkpoint_dir, overwrite=True)
            
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_epoch_train_loss)
                else:
                    scheduler.step() 

            # Print progress with detailed loss breakdown
            if avg_epoch_val_loss is not None:
                print(f'Epoch {epoch}/{num_epochs_to_run}, Train Loss: {avg_epoch_train_loss:.6f} (Base: {avg_train_base_loss:.6f}, Penalty: {avg_train_penalty:.6f}), Val Loss: {avg_epoch_val_loss:.6f} (Base: {avg_val_base_loss:.6f}, Penalty: {avg_val_penalty:.6f}), LR: {lr_for_epoch:.2e}')
            else:
                print(f'Epoch {epoch}/{num_epochs_to_run}, Train Loss: {avg_epoch_train_loss:.6f} (Base: {avg_train_base_loss:.6f}, Penalty: {avg_train_penalty:.6f}), LR: {lr_for_epoch:.2e}')
            
        print("Training for this configuration finished.")
        
        # Load the best model from temporary checkpoint
        model_path = os.path.join(temp_checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            try:
                # Use load_model function to load the best checkpoint
                best_model = load_model(model_path, self.device, self.dtype, initialized_model=self)
                self.load_state_dict(best_model.state_dict())
                print(f"Successfully loaded best model from checkpoint at epoch {best_epoch}")
                
                # Clean up temporary checkpoint directory
                if os.path.exists(temp_checkpoint_dir):
                    shutil.rmtree(temp_checkpoint_dir)
                    print(f"Cleaned up temporary checkpoint directory: {temp_checkpoint_dir}")
                    
            except Exception as e:
                print(f"Failed to load best model from checkpoint: {e}")
                print("Continuing with current model state")
        else:
            print("No checkpoint found, continuing with current model state")
        
        return history

    def t_forward(self, t, x):
        return self.forward(x)
    
    def plot_lagrangian(self, x):
        return self.lagrangian(torch.Tensor(x)).cpu().detach()
    

def loss(pred, targ):
    """
    Legacy loss function - kept for backward compatibility.
    Consider using LNN.compute_loss() for new code as it includes regularization.
    """
    return torch.mean(torch.abs(pred - targ))

def custom_initialize_weights(model_instance):
    """
    Applies custom weight initialization to the LNN model instance.
    Specifically, initializes the last linear layer of the 'mlp' part
    of the model's 'network' attribute.
    """
    print("Applying custom weight initialization...")
    with torch.no_grad():
        last_linear_layer_mlp = None
        
        # Handle nn.DataParallel if model is wrapped
        network_to_inspect = model_instance.module.network if isinstance(model_instance, torch.nn.DataParallel) else model_instance.network
        
        # Initialize last layer of the main MLP (self.mlp in CombinedNetwork)
        if hasattr(network_to_inspect, 'mlp') and isinstance(network_to_inspect.mlp, torch.nn.Sequential):
            for layer in reversed(network_to_inspect.mlp):
                if isinstance(layer, torch.nn.Linear):
                    last_linear_layer_mlp = layer
                    break
            
            if last_linear_layer_mlp is not None:
                print("Initializing last layer of main MLP.")
                last_linear_layer_mlp.weight.data.uniform_(-0.01, 0.01)
                if last_linear_layer_mlp.bias is not None: # Your mlp's last layer has bias=False
                    last_linear_layer_mlp.bias.data.fill_(0.0)
            else:
                print("Warning: Could not find last linear layer in 'model.network.mlp' for custom initialization.")
        else:
            print("Warning: 'model.network.mlp' not found or not a Sequential module.")

        # Note: Your QuadraticMLP (self.qmlp in CombinedNetwork) uses default PyTorch initializations
        # for its layers, including its own last linear layer. If you also want to
        # initialize it similarly, you'd add another block here to inspect 'model.network.qmlp'.
        # For now, we are only re-applying the initialization you explicitly had for the main 'mlp'.f