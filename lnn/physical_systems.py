# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Modified from hessian_regularized_lnn (A. U. Hamzaogullari, https://github.com/abdullahumuth/hessian_regularized_lnn,
# commit 43442d3, Apache-2.0). Changes relative to that commit: see NOTICE.
# Trimmed to the two systems used in this work (three_body: LJ + Axilrod-Teller; newt_three_body: Newtonian gravity),
# copied verbatim from the authors' working copy. The other systems of the original file are omitted.

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, path
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

class PhysicalSystemBase:
    def validate_implementation(self):
        pass


class three_body(PhysicalSystemBase):
    def __init__(self, m=1.0, E0=0.1, sigma_LJ=1.0, eps_LJ=1.0, g=1.0):
        super().__init__()
        self.dof = 12 #12 # x_i, y_i, x_j, y_j, x_k, y_k and the respective velocities.
        self.angle_indices = []  
        self.scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # these automatically change if normalization=True? depending on dynamic max or the bounds?
        self.param_names = ['E0', 'm', 'sigma_LJ', 'eps_LJ', 'g'] # E0 is the constant of V_AT.

        # Physical constants / fixed params
        self.m = m
        self.E0 = E0
        self.sigma_LJ = sigma_LJ
        self.eps_LJ = eps_LJ
        self.enforce_mass_matrix=True # already hard-coded?
        self.g = g

 
        self.test_params = { 
                # initial demo
                'toy_position': torch.tensor([-1.4, 1.75, 0.03, -0.18, 1.5, 0.25]),  # ([-1.31, -1.34, 0.71,  0.04, -0.94, 0.05]
                'toy_velocity': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # stationary start

                # time domain
                'toy_time_dataset': np.arange(0, 42.55, 0.05), #  44.5,0.05 arranged to stop right before the collapse. (0,100,0.1)
                # okay 45.0 literally kills the process by forcing ode. 44.55 was fine, took 3 seconds to run 44.60 is also fine, just warned the ode, but ran fast and good.
                # analytic solution starts dying painfully by 44.65. then lagrangian is more resistful to errors, nice. but nevertheless.
                # again, this means r~1e-6in training dataset is too bad to resist for the solvers of training, but alpha can help. if not, increase the generation limit
                'time_step': 0.01,
                'time_bounds': (0, 5.0), # 44.5, can i dynamically change this, like an early stopping, in the testing?

                # Physical coordinate limits
                # set realistic spatial limits
                'position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)),  # x_j, x_k, y_k in meters (or nondimensional)
                'velocity_bounds': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),


                'test_seed': 96,
                'num_tests': 1,
                'test_states': None, 

                # for validation 
                'validation_position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)), # was 1.5
                'validation_velocity_bounds': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        }   

        self.train_hyperparams = {
                # Optimization
                'lr': 1e-3,          
                'num_samples': 60000,       
                'test_ratio': 0.1,       
                'total_epochs': 400,       
                'minibatch_size': 128,       
                'train_seed': 100,           

                # Physical bounds for generating training data
                'position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)),
                'velocity_bounds':((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                'eta_min': 1e-6,
            }
        self.model_params = {'num_layers' : 4, # 4, 3 TODO: increase the size further! (+ fine tuning with the trajectory data + different acrhicecture!)
                                'activation_fn' : nn.GELU(),
                                'hidden_dim' : 700, # 500, 128
                            }
        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale

    def kinetic(self, q, q_dot, cos=torch.cos): # TODO: add scaling!
        if len(q_dot.shape) == 1:
            xv_i = q_dot[0] * self.scale[0] # 
            yv_i = q_dot[1] * self.scale[1] #
            xv_j = q_dot[2] * self.scale[2] # 
            yv_j = q_dot[3] * self.scale[3]
            xv_k = q_dot[4] * self.scale[4]
            yv_k = q_dot[5] * self.scale[5]
        else:
            xv_i = q_dot[:, 0] * self.scale[0]
            yv_i = q_dot[:, 1] * self.scale[1]
            xv_j = q_dot[:, 2] * self.scale[2]
            yv_j = q_dot[:, 3] * self.scale[3]
            xv_k = q_dot[:, 4] * self.scale[4]
            yv_k = q_dot[:, 5] * self.scale[5]

        return 0.5 * self.m * (
            yv_i**2  + xv_i**2 +
            yv_j**2  + xv_j**2 +
            yv_k**2 + xv_k**2
        )

    def potential(self, q, q_dot, cos=torch.cos, eps=1e-8, alpha=1e-4): # Axilrod-Teller potential! TODO: add scaling!

        #x_i, y_i, x_j, y_j, x_k, y_k  = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4], q[:, 5]
        if len(q.shape) == 1:
            x_i = q[0] * self.scale[0]
            y_i = q[1] * self.scale[1]
            x_j = q[2] * self.scale[2]
            y_j = q[3] * self.scale[3]
            x_k = q[4] * self.scale[4]
            y_k = q[5] * self.scale[5]
        else:
            x_i = q[:, 0] * self.scale[0]
            y_i = q[:, 1] * self.scale[1]
            x_j = q[:, 2] * self.scale[2]
            y_j = q[:, 3] * self.scale[3]
            x_k = q[:, 4] * self.scale[4]
            y_k = q[:, 5] * self.scale[5]

        #min_dist = eps

        #r_ij = torch.clamp(torch.sqrt((x_i - x_j)**2 + (y_i - y_j)**2) + eps, min=0.01) # this is a good limit, further than this we get issues of collapsing.
        #r_ik = torch.clamp(torch.sqrt((x_i - x_k)**2 + (y_i - y_k)**2) + eps, min=0.01)
        #r_jk = torch.clamp(torch.sqrt((x_j - x_k)**2 + (y_j - y_k)**2) + eps, min=0.01)
        r_ij = torch.sqrt((x_i-x_j)**2 + (y_i-y_j)**2) # + (alpha)
        r_ik = torch.sqrt((x_i - x_k)**2 + (y_i - y_k)**2)
        r_jk = torch.sqrt((x_j - x_k)**2 + (y_j - y_k)**2)
        #print('r_ij is like -physical-:', r_ij, 'and the min,', r_ij.abs().min().item())

        cos_i = torch.clamp((r_ij**2 + r_ik**2 - r_jk**2) / (2*r_ij*r_ik + eps), -1.0, 1.0)
        cos_j = torch.clamp((r_ij**2 + r_jk**2 - r_ik**2) / (2*r_ij*r_jk + eps), -1.0, 1.0)
        cos_k = torch.clamp((r_ik**2 + r_jk**2 - r_ij**2) / (2*r_ik*r_jk + eps), -1.0, 1.0)

        denom = (r_ij * r_jk * r_ik)**3  # even 1e-6 is a thin limit, denominator of ~1e-6 (NO, ~1e-18 !) causes collapse/explosion of potential. 
        # alpha should help. it was 1e-2, now increased bc that was a bit too tough of a limit

        V_AT = self.E0 * (1 + 3*cos_i*cos_j*cos_k) / denom
        #print('DENOM IS', denom)


         # Lennard–Jones two-body term (modeled for shorter-ranges)

        V_LJ_ij =  4 * self.eps_LJ * ((self.sigma_LJ / r_ij)**12 - (self.sigma_LJ / r_ij)**6)
        V_LJ_ik =  4 * self.eps_LJ * ((self.sigma_LJ / r_ik)**12 - (self.sigma_LJ / r_ik)**6)
        V_LJ_jk =  4 * self.eps_LJ * ((self.sigma_LJ / r_jk)**12 - (self.sigma_LJ / r_jk)**6)


        # Total pairwise LJ energy
        V_LJ = V_LJ_ij + V_LJ_ik + V_LJ_jk

        # Total potential energy
        V = V_AT + V_LJ
        #V = ((self.eps_LJ * (self.sigma_LJ)**9) * V_AT) + V_LJ # this mimicks the paper (we dont need this when E0 is explicitly set.)

        ### THE PSEUDO POTENTIAL ######
        '''
        
        A = 1.0
        B = 0.5
        beta = 1.0
        
        rsum = r_ij + r_ik + r_jk
        angle_term = 1 + B * cos_i * cos_j * cos_k
        
        V = A * torch.exp(-beta * rsum) * angle_term  '''
        ###### THE PSEUDO POTENTIAL ######

        return V
    
    def potential_newt(self, q, q_dot=None, cos=torch.cos, softening=1e-6): # newt
        """
        Pure Newtonian gravitational potential energy for 3 bodies in 2D.

        V = -G * sum_{a<b} m_a m_b / r_ab

        Uses a small Plummer softening: r = sqrt(|r|^2 + softening^2)
        to avoid singularities at collision during debugging/training.

        Compatible with q of shape (6,) or (B, 6).
        """
        # gravitational constant in your nondimensional units
        # (choose G=1 by default; you can add it to param_names if you want learnable)
        G = getattr(self, "G", 1.0)

        # unpack (and apply your internal scaling convention)
        if len(q.shape) == 1:
            x_i = q[0] * self.scale[0]
            y_i = q[1] * self.scale[1]
            x_j = q[2] * self.scale[2]
            y_j = q[3] * self.scale[3]
            x_k = q[4] * self.scale[4]
            y_k = q[5] * self.scale[5]
        else:
            x_i = q[:, 0] * self.scale[0]
            y_i = q[:, 1] * self.scale[1]
            x_j = q[:, 2] * self.scale[2]
            y_j = q[:, 3] * self.scale[3]
            x_k = q[:, 4] * self.scale[4]
            y_k = q[:, 5] * self.scale[5]

        # pairwise distances with Plummer softening
        s2 = softening * softening

        dx_ij, dy_ij = x_i - x_j, y_i - y_j
        dx_ik, dy_ik = x_i - x_k, y_i - y_k
        dx_jk, dy_jk = x_j - x_k, y_j - y_k

        r_ij = torch.sqrt(dx_ij*dx_ij + dy_ij*dy_ij + s2)
        r_ik = torch.sqrt(dx_ik*dx_ik + dy_ik*dy_ik + s2)
        r_jk = torch.sqrt(dx_jk*dx_jk + dy_jk*dy_jk + s2)

        # masses: keep it simple with your existing self.m (all equal masses)
        m = self.m

        V = -G * (m*m / r_ij + m*m / r_ik + m*m / r_jk) 
        return V


      
    def potential_constant(self, q, q_dot, cos=torch.cos): # NOTE: a trivial constant force potential, to diagnose the issue of learning

        if len(q.shape) == 1:
            y_i = q[0] * self.scale[0] # x_i
            y_j = q[1] * self.scale[1] # y_i
            y_k = q[2] * self.scale[2] # x_j
         #   y_j = q[3] * self.scale[3]
          #  x_k = q[4] * self.scale[4]
          #  y_k = q[5] * self.scale[5]
        else:
            y_i = q[:, 0] * self.scale[0] # same as above
            y_j = q[:, 1] * self.scale[1] # same
            y_k = q[:, 2] * self.scale[2] # same
            #y_j = q[:, 3] * self.scale[3]
            #x_k = q[:, 4] * self.scale[4]
            #y_k = q[:, 5] * self.scale[5]
        return self.m * self.g * ( y_i + y_j + y_k) # let's see even easier version to catch errors.
    


    def lagrangian(self,x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1) # torch and np has different split() arguments.
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T - V

    def energy(self,x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
            cos = torch.cos
        elif isinstance(x, np.ndarray): 
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.shape)==1:
                #x = np.expand_dims(x, axis=0)
                x = x.unsqueeze(0)
            #q, qt = np.split(x, self.dof//3, axis=-1)
            #cos = np.cos
        q, qt = torch.split(x, self.dof//2, dim=-1)
        cos = torch.cos

        T = self.kinetic(q, qt, cos)
        V = self.potential(q, qt, cos)
        return T + V

    def solve_acceleration(self, q, q_dot, as_numpy=False): # this is safe since m is constant, but we might want it to account for more!
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, dtype=torch.float32, requires_grad=True)
            as_numpy=True
        else:
            q.requires_grad_(True)
            as_numpy=False
        if isinstance(q_dot, np.ndarray):
            q_dot = torch.tensor(q_dot, dtype=torch.float32, requires_grad=True)
            as_numpy=True
        else:
            q_dot.requires_grad_(True)
            as_numpy=False

        # q's are all scaled.        
        V = self.potential(q, q_dot)
        gradV = torch.autograd.grad(V.sum(), q, create_graph=True)[0] # there's a scale factor due to gradient!

        s = torch.tensor(self.scale, dtype=q.dtype, device=q.device)
        if len(q.shape) == 1:
            acc = -gradV / (self.m * (s**2))
        else:
            acc = -gradV / (self.m * (s[None, :]**2)) # returning scaled q_ddot!
        
        # the rescaled, physical acceleration would be -grad(V,q)/s.


        if as_numpy:
            return acc.detach().cpu().numpy()
    
        return acc 

    def plot_solved_dynamics(self, t, path, labelstr="", internal=False, track_motion=True, **kwargs):
        """
        Plot 3-body dynamics with out-of-bound detection visualization.
        """
        # ----------------------------
        # Unpack positions/velocities
        # ----------------------------
        x_i, y_i = path[:, 0], path[:, 1]
        x_j, y_j = path[:, 2], path[:, 3]
        x_k, y_k = path[:, 4], path[:, 5]

        vx_i, vy_i = path[:, 6], path[:, 7]
        vx_j, vy_j = path[:, 8], path[:, 9]
        vx_k, vy_k = path[:, 10], path[:, 11]

        # ----------------------------
        # Threshold parameters
        # ----------------------------
        r_min = kwargs.get("r_min", 0.6) # for pure AT
        r_max = kwargs.get("r_max", 5.0) # for pure AT

        def dist(a1, b1, a2, b2):
            return np.sqrt((a1 - a2)**2 + (b1 - b2)**2)

        r_ij = dist(x_i, y_i, x_j, y_j)
        r_ik = dist(x_i, y_i, x_k, y_k)
        r_jk = dist(x_j, y_j, x_k, y_k)

        bad = (r_ij < r_min) | (r_ij > r_max) | \
            (r_ik < r_min) | (r_ik > r_max) | \
            (r_jk < r_min) | (r_jk > r_max)

        # First violation index (blow-up event)
        blow_idx = np.where(bad)[0]
        blow_t = t[blow_idx[0]] if len(blow_idx) > 0 else None

        # ----------------------------
        # Helper: segments
        # ----------------------------
        def make_segments(x, y):
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            return np.concatenate([pts[:-1], pts[1:]], axis=1)

        segments_i = make_segments(x_i, y_i)
        segments_j = make_segments(x_j, y_j)
        segments_k = make_segments(x_k, y_k)

        good_mask = ~bad[:-1]
        bad_mask = bad[:-1]

        # speeds
        speed_i = np.sqrt(vx_i**2 + vy_i**2)
        speed_j = np.sqrt(vx_j**2 + vy_j**2)
        speed_k = np.sqrt(vx_k**2 + vy_k**2)

        # ----------------------------
        # Figure
        # ----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ============================
        # ENERGY PLOT
        # ============================
        energy = [self.energy(x) for x in path]
        axes[0].plot(t, energy, label=f"Energy - {labelstr}")

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Energy (J)")
        axes[0].set_title("Energy Conservation")
        axes[0].grid(True, alpha=0.3)

        # Blow-up marker
        if blow_t is not None:
            axes[0].axvline(blow_t, color='red', linestyle='--', alpha=0.8,
                            label="Threshold violation")
            axes[0].scatter([blow_t], [energy[blow_idx[0]]],
                            color='red', zorder=5)

        axes[0].legend()

        # ============================
        # TRAJECTORY PLOT
        # ============================
        if track_motion:
            ax = axes[1]

            def plot_segments(segments, speed, cmap, alpha=1.0):
                lc = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=plt.Normalize(speed.min(), speed.max())
                )
                lc.set_array(speed[:len(segments)])
                lc.set_linewidth(2)
                lc.set_alpha(alpha)
                ax.add_collection(lc)

            # GOOD (normal)
            plot_segments(segments_i[good_mask], speed_i, 'viridis', alpha=1.0)
            plot_segments(segments_j[good_mask], speed_j, 'plasma', alpha=1.0)
            plot_segments(segments_k[good_mask], speed_k, 'inferno', alpha=1.0)

            # BAD (out-of-bounds signature)
            plot_segments(segments_i[bad_mask], speed_i, 'viridis', alpha=0.25)
            plot_segments(segments_j[bad_mask], speed_j, 'plasma', alpha=0.25)
            plot_segments(segments_k[bad_mask], speed_k, 'inferno', alpha=0.25)

            # Start/end markers
            ax.plot(x_i[0], y_i[0], 'o', color='green', alpha=0.8, label='i start')
            ax.plot(x_j[0], y_j[0], 'o', color='orange', alpha=0.8, label='j start')
            ax.plot(x_k[0], y_k[0], 'o', color='purple', alpha=0.8, label='k start')

            ax.plot(x_i[-1], y_i[-1], 'o', color='green', alpha=0.4, label='i end')
            ax.plot(x_j[-1], y_j[-1], 'o', color='orange', alpha=0.4, label='j end')
            ax.plot(x_k[-1], y_k[-1], 'o', color='purple', alpha=0.4, label='k end')

            # Blow-up marker in trajectory plot
            if blow_idx is not None and len(blow_idx) > 0:
                bi = blow_idx[0]
                ax.scatter([x_i[bi], x_j[bi], x_k[bi]],
                        [y_i[bi], y_j[bi], y_k[bi]],
                        color='red', s=30, zorder=6, label="Violation point")

            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("3-Body Trajectories (normal vs out-of-bounds)")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        return fig
        
    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)
        plt.xlabel('Time')
        plt.ylabel('Lagrangian')
        plt.title('Lagrangian vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)




class newt_three_body(PhysicalSystemBase):
    def __init__(self, m=1.0, E0=0.1, sigma_LJ=1.0, eps_LJ=1.0, g=1.0):
        super().__init__()
        self.dof = 12 #12 # x_i, y_i, x_j, y_j, x_k, y_k and the respective velocities.
        self.angle_indices = []  
        self.scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # these automatically change if normalization=True? depending on dynamic max or the bounds?
        self.param_names = ['E0', 'm', 'sigma_LJ', 'eps_LJ', 'g'] # E0 is the constant of V_AT.

        # Physical constants / fixed params
        self.m = m
        self.E0 = E0
        self.sigma_LJ = sigma_LJ
        self.eps_LJ = eps_LJ
        self.enforce_mass_matrix=True # already hard-coded?
        self.g = g

 
        self.test_params = { 
                # initial demo
                'toy_position': torch.tensor([-1.4, 1.75, 0.03, -0.18, 1.5, 0.25]),  # ([-1.31, -1.34, 0.71,  0.04, -0.94, 0.05]
                'toy_velocity': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # stationary start

                # time domain
                'toy_time_dataset': np.arange(0, 42.55, 0.05), #  44.5,0.05 arranged to stop right before the collapse. (0,100,0.1)
                # okay 45.0 literally kills the process by forcing ode. 44.55 was fine, took 3 seconds to run 44.60 is also fine, just warned the ode, but ran fast and good.
                # analytic solution starts dying painfully by 44.65. then lagrangian is more resistful to errors, nice. but nevertheless.
                # again, this means r~1e-6in training dataset is too bad to resist for the solvers of training, but alpha can help. if not, increase the generation limit
                'time_step': 0.01,
                'time_bounds': (0, 5.0), # 44.5, can i dynamically change this, like an early stopping, in the testing?

                # Physical coordinate limits
                # set realistic spatial limits
                'position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)),  # x_j, x_k, y_k in meters (or nondimensional)
                'velocity_bounds': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),


                'test_seed': 96,
                'num_tests': 1,
                'test_states': None, 

                # for validation 
                'validation_position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)), # was 1.5
                'validation_velocity_bounds': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        }   

        self.train_hyperparams = {
                # Optimization
                'lr': 1e-3,          
                'num_samples': 60000,       
                'test_ratio': 0.1,       
                'total_epochs': 400,       
                'minibatch_size': 128,       
                'train_seed': 100,           

                # Physical bounds for generating training data
                'position_bounds': ((-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5), (-5.5, 5.5)),
                'velocity_bounds':((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                'eta_min': 1e-6,
            }
        self.model_params = {'num_layers' : 4, # 4, 3 TODO: increase the size further! (+ fine tuning with the trajectory data + different acrhicecture!)
                                'activation_fn' : nn.GELU(),
                                'hidden_dim' : 700, # 500, 128
                            }
        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale

    def kinetic(self, q, q_dot, cos=torch.cos): # TODO: add scaling!
        if len(q_dot.shape) == 1:
            xv_i = q_dot[0] * self.scale[0] # 
            yv_i = q_dot[1] * self.scale[1] #
            xv_j = q_dot[2] * self.scale[2] # 
            yv_j = q_dot[3] * self.scale[3]
            xv_k = q_dot[4] * self.scale[4]
            yv_k = q_dot[5] * self.scale[5]
        else:
            xv_i = q_dot[:, 0] * self.scale[0]
            yv_i = q_dot[:, 1] * self.scale[1]
            xv_j = q_dot[:, 2] * self.scale[2]
            yv_j = q_dot[:, 3] * self.scale[3]
            xv_k = q_dot[:, 4] * self.scale[4]
            yv_k = q_dot[:, 5] * self.scale[5]

        return 0.5 * self.m * (
            yv_i**2  + xv_i**2 +
            yv_j**2  + xv_j**2 +
            yv_k**2 + xv_k**2
        )

    
    def potential(self, q, q_dot=None, cos=torch.cos, softening=1e-6): # newt
        """
        Pure Newtonian gravitational potential energy for 3 bodies in 2D.

        V = -G * sum_{a<b} m_a m_b / r_ab

        Uses a small Plummer softening: r = sqrt(|r|^2 + softening^2)
        to avoid singularities at collision during debugging/training.

        Compatible with q of shape (6,) or (B, 6).
        """
        # gravitational constant in your nondimensional units
        # (choose G=1 by default; you can add it to param_names if you want learnable)
        G = getattr(self, "G", 1.0)

        # unpack (and apply your internal scaling convention)
        if len(q.shape) == 1:
            x_i = q[0] * self.scale[0]
            y_i = q[1] * self.scale[1]
            x_j = q[2] * self.scale[2]
            y_j = q[3] * self.scale[3]
            x_k = q[4] * self.scale[4]
            y_k = q[5] * self.scale[5]
        else:
            x_i = q[:, 0] * self.scale[0]
            y_i = q[:, 1] * self.scale[1]
            x_j = q[:, 2] * self.scale[2]
            y_j = q[:, 3] * self.scale[3]
            x_k = q[:, 4] * self.scale[4]
            y_k = q[:, 5] * self.scale[5]

        # pairwise distances with Plummer softening
        s2 = softening * softening

        dx_ij, dy_ij = x_i - x_j, y_i - y_j
        dx_ik, dy_ik = x_i - x_k, y_i - y_k
        dx_jk, dy_jk = x_j - x_k, y_j - y_k

        r_ij = torch.sqrt(dx_ij*dx_ij + dy_ij*dy_ij + s2)
        r_ik = torch.sqrt(dx_ik*dx_ik + dy_ik*dy_ik + s2)
        r_jk = torch.sqrt(dx_jk*dx_jk + dy_jk*dy_jk + s2)

        # masses: keep it simple with your existing self.m (all equal masses)
        m = self.m

        V = -G * (m*m / r_ij + m*m / r_ik + m*m / r_jk) 
        return V


    def lagrangian(self,x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1) # torch and np has different split() arguments.
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T - V

    def energy(self,x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
            cos = torch.cos
        elif isinstance(x, np.ndarray): 
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.shape)==1:
                #x = np.expand_dims(x, axis=0)
                x = x.unsqueeze(0)
            #q, qt = np.split(x, self.dof//3, axis=-1)
            #cos = np.cos
        q, qt = torch.split(x, self.dof//2, dim=-1)
        cos = torch.cos

        T = self.kinetic(q, qt, cos)
        V = self.potential(q, qt, cos)
        return T + V

    def solve_acceleration(self, q, q_dot, as_numpy=False): # this is safe since m is constant, but we might want it to account for more!
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, dtype=torch.float32, requires_grad=True)
            as_numpy=True
        else:
            q.requires_grad_(True)
            as_numpy=False
        if isinstance(q_dot, np.ndarray):
            q_dot = torch.tensor(q_dot, dtype=torch.float32, requires_grad=True)
            as_numpy=True
        else:
            q_dot.requires_grad_(True)
            as_numpy=False

        # q's are all scaled.        
        V = self.potential(q, q_dot)
        gradV = torch.autograd.grad(V.sum(), q, create_graph=True)[0] # there's a scale factor due to gradient!

        s = torch.tensor(self.scale, dtype=q.dtype, device=q.device)
        if len(q.shape) == 1:
            acc = -gradV / (self.m * (s**2))
        else:
            acc = -gradV / (self.m * (s[None, :]**2)) # returning scaled q_ddot!
        
        # the rescaled, physical acceleration would be -grad(V,q)/s.


        if as_numpy:
            return acc.detach().cpu().numpy()
    
        return acc 

    '''def plot_solved_dynamics(self, t, path, labelstr="", internal=False, track_motion=True, **kwargs): #TODO: add scaling!
        """
        Plot 3-body dynamics in Cartesian coordinates (internal coords removed),
        with energy plot untouched, and a visually enhanced trajectory plot.

        path: array of shape [len(t), 12] (x_i,y_i,x_j,y_j,x_k,y_k,vx_i,vy_i,vx_j,vy_j,vx_k,vy_k)
        """

        # Unpack positions and velocities
        x_i, y_i = path[:, 0], path[:, 1]
        x_j, y_j = path[:, 2], path[:, 3]
        x_k, y_k = path[:, 4], path[:, 5]
        #y_i, y_j, y_k = path[:, 0], path[:, 1], path[:, 2]

        vx_i, vy_i = path[:, 6], path[:, 7]
        vx_j, vy_j = path[:, 8], path[:, 9]
        vx_k, vy_k = path[:, 10], path[:, 11]
        #vy_i, vy_j, vy_k = path[:, 3], path[:, 4], path[:, 5]


        # Energy plot (untouched)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        energy = [self.energy(x) for x in path]
        axes[0].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Energy (J)')
        axes[0].set_title('Energy Conservation')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Enhanced trajectory plot
        if track_motion:
            ax = axes[1]
            # Compute speeds for color mapping
            speed_i = np.sqrt(vx_i**2 + vy_i**2)
            speed_j = np.sqrt(vx_j**2 + vy_j**2)
            speed_k = np.sqrt(vx_k**2 + vy_k**2)

            # Plot smoothed trajectory with velocity colormap
            points_i = np.array([x_i, y_i]).T.reshape(-1,1,2)
            segments_i = np.concatenate([points_i[:-1], points_i[1:]], axis=1)
            points_j = np.array([x_j, y_j]).T.reshape(-1,1,2)
            segments_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
            points_k = np.array([x_k, y_k]).T.reshape(-1,1,2)
            segments_k = np.concatenate([points_k[:-1], points_k[1:]], axis=1)


            lc_i = LineCollection(segments_i, cmap='viridis', norm=plt.Normalize(speed_i.min(), speed_i.max()), label='body i')
            lc_i.set_array(speed_i)
            lc_i.set_linewidth(2)
            ax.add_collection(lc_i)

            lc_j = LineCollection(segments_j, cmap='plasma', norm=plt.Normalize(speed_j.min(), speed_j.max()), label='body j')
            lc_j.set_array(speed_j)
            lc_j.set_linewidth(2)
            ax.add_collection(lc_j)

            lc_k = LineCollection(segments_k, cmap='inferno', norm=plt.Normalize(speed_k.min(), speed_k.max()), label='body k')
            lc_k.set_array(speed_k)
            lc_k.set_linewidth(2)
            ax.add_collection(lc_k)

            # Start/end markers
            #3a.plot([x_i[0], x_j[0], x_k[0]], [y_i[0], y_j[0], y_k[0]], 'ko', alpha=0.6, label='Start')
            #ax.plot([x_i[-1], x_j[-1], x_k[-1]], [y_i[-1], y_j[-1], y_k[-1]], 'o', color='gray', alpha=0.6, label='End')
            ax.plot(x_i[0], y_i[0], 'o', color='green',  alpha=0.8, label='i start')
            ax.plot(x_j[0], y_j[0], 'o', color='orange', alpha=0.8, label='j start')
            ax.plot(x_k[0], y_k[0], 'o', color='purple', alpha=0.8, label='k start')

            # End markers: same colors, lower alpha
            ax.plot(x_i[-1], y_i[-1], 'o', color='green',  alpha=0.4, label='i end')
            ax.plot(x_j[-1], y_j[-1], 'o', color='orange', alpha=0.4, label='j end')
            ax.plot(x_k[-1], y_k[-1], 'o', color='purple', alpha=0.4, label='k end')

            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("3-Body Trajectories (Velocity Colored)")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        return fig'''

    def plot_solved_dynamics(self, t, path, labelstr="", internal=False, track_motion=True, **kwargs):
        """
        Plot 3-body dynamics with out-of-bound detection visualization.
        """
        # ----------------------------
        # Unpack positions/velocities
        # ----------------------------
        x_i, y_i = path[:, 0], path[:, 1]
        x_j, y_j = path[:, 2], path[:, 3]
        x_k, y_k = path[:, 4], path[:, 5]

        vx_i, vy_i = path[:, 6], path[:, 7]
        vx_j, vy_j = path[:, 8], path[:, 9]
        vx_k, vy_k = path[:, 10], path[:, 11]

        # ----------------------------
        # Threshold parameters
        # ----------------------------
        r_min = kwargs.get("r_min", 0.6) # for pure AT
        r_max = kwargs.get("r_max", 5.0) # for pure AT

        def dist(a1, b1, a2, b2):
            return np.sqrt((a1 - a2)**2 + (b1 - b2)**2)

        r_ij = dist(x_i, y_i, x_j, y_j)
        r_ik = dist(x_i, y_i, x_k, y_k)
        r_jk = dist(x_j, y_j, x_k, y_k)

        bad = (r_ij < r_min) | (r_ij > r_max) | \
            (r_ik < r_min) | (r_ik > r_max) | \
            (r_jk < r_min) | (r_jk > r_max)

        # First violation index (blow-up event)
        blow_idx = np.where(bad)[0]
        blow_t = t[blow_idx[0]] if len(blow_idx) > 0 else None

        # ----------------------------
        # Helper: segments
        # ----------------------------
        def make_segments(x, y):
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            return np.concatenate([pts[:-1], pts[1:]], axis=1)

        segments_i = make_segments(x_i, y_i)
        segments_j = make_segments(x_j, y_j)
        segments_k = make_segments(x_k, y_k)

        good_mask = ~bad[:-1]
        bad_mask = bad[:-1]

        # speeds
        speed_i = np.sqrt(vx_i**2 + vy_i**2)
        speed_j = np.sqrt(vx_j**2 + vy_j**2)
        speed_k = np.sqrt(vx_k**2 + vy_k**2)

        # ----------------------------
        # Figure
        # ----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ============================
        # ENERGY PLOT
        # ============================
        energy = [self.energy(x) for x in path]
        axes[0].plot(t, energy, label=f"Energy - {labelstr}")

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Energy (J)")
        axes[0].set_title("Energy Conservation")
        axes[0].grid(True, alpha=0.3)

        # Blow-up marker
        if blow_t is not None:
            axes[0].axvline(blow_t, color='red', linestyle='--', alpha=0.8,
                            label="Threshold violation")
            axes[0].scatter([blow_t], [energy[blow_idx[0]]],
                            color='red', zorder=5)

        axes[0].legend()

        # ============================
        # TRAJECTORY PLOT
        # ============================
        if track_motion:
            ax = axes[1]

            def plot_segments(segments, speed, cmap, alpha=1.0):
                lc = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=plt.Normalize(speed.min(), speed.max())
                )
                lc.set_array(speed[:len(segments)])
                lc.set_linewidth(2)
                lc.set_alpha(alpha)
                ax.add_collection(lc)

            # GOOD (normal)
            plot_segments(segments_i[good_mask], speed_i, 'viridis', alpha=1.0)
            plot_segments(segments_j[good_mask], speed_j, 'plasma', alpha=1.0)
            plot_segments(segments_k[good_mask], speed_k, 'inferno', alpha=1.0)

            # BAD (out-of-bounds signature)
            plot_segments(segments_i[bad_mask], speed_i, 'viridis', alpha=0.25)
            plot_segments(segments_j[bad_mask], speed_j, 'plasma', alpha=0.25)
            plot_segments(segments_k[bad_mask], speed_k, 'inferno', alpha=0.25)

            # Start/end markers
            ax.plot(x_i[0], y_i[0], 'o', color='green', alpha=0.8, label='i start')
            ax.plot(x_j[0], y_j[0], 'o', color='orange', alpha=0.8, label='j start')
            ax.plot(x_k[0], y_k[0], 'o', color='purple', alpha=0.8, label='k start')

            ax.plot(x_i[-1], y_i[-1], 'o', color='green', alpha=0.4, label='i end')
            ax.plot(x_j[-1], y_j[-1], 'o', color='orange', alpha=0.4, label='j end')
            ax.plot(x_k[-1], y_k[-1], 'o', color='purple', alpha=0.4, label='k end')

            # Blow-up marker in trajectory plot
            if blow_idx is not None and len(blow_idx) > 0:
                bi = blow_idx[0]
                ax.scatter([x_i[bi], x_j[bi], x_k[bi]],
                        [y_i[bi], y_j[bi], y_k[bi]],
                        color='red', s=30, zorder=6, label="Violation point")

            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("3-Body Trajectories (normal vs out-of-bounds)")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        return fig
        
    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)
        plt.xlabel('Time')
        plt.ylabel('Lagrangian')
        plt.title('Lagrangian vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)




def initialize_particle_system(physics):
    if physics == "three_body":
        return three_body()
    if physics == "newt_three_body":
        return newt_three_body()
    raise ValueError(f"Unknown physics type: '{physics}' (this release contains 'three_body' and 'newt_three_body').")
