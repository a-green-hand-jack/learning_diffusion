import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
import ipdb

from .ddpm import DDPM


class DDIM(DDPM):
    """
    Denoising Diffusion Implicit Model (DDIM) implementation.
    Inherits from DDPM and modifies the sampling process.

    What should be noted is that the \alpha_t in DDIM paper refers to \bar{\alpha}_t from DDPM !!!
    While to avoid misunderstanding, I just use \bar{\alpha}_t in the DDIM.

    Args:
        eps_model (nn.Module): Neural network that predicts noise ε
        n_step (int): Number of diffusion steps
        ddim_sampling_steps (int): Number of steps for DDIM sampling (can be < n_step)
        ddim_discretize (str): Method to discretize timesteps ('uniform' or 'quad')
        device (torch.device): Device to run the model on
        ddim_eta (float): Coefficient for adding noise in the sampling process
    """

    def __init__(
        self,
        eps_model: nn.Module,
        n_step: int,
        ddim_sampling_steps: int,
        ddim_discretize: str = 'uniform',
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ddim_eta: float = 0.0,
    ):
        super().__init__(eps_model, n_step, device)

        self.ddim_sampling_steps = ddim_sampling_steps
        self.ddim_timesteps = self.get_ddim_timesteps(ddim_discretize)
        self.ddim_eta = ddim_eta
        
    def get_ddim_timesteps(self, ddim_discretize: str = 'uniform') -> np.ndarray:
        """
        Get DDIM timesteps based on different discretization strategies.
        
        Args:
            ddim_discretize (str): Discretization method ('uniform' or 'quad')
            
        Returns:
            np.ndarray: DDIM timesteps
        """
        if ddim_discretize == 'uniform':
            # Uniform spacing
            c = self.n_step // self.ddim_sampling_steps
            steps = np.asarray(list(range(0, self.n_step, c)))
            
        elif ddim_discretize == 'quad':
            # Quadratic spacing
            steps = ((np.linspace(0, np.sqrt(self.n_step * 0.8), self.ddim_sampling_steps)) ** 2).astype(int)
            
        else:
            raise NotImplementedError(f"Discretization method {ddim_discretize} not implemented")
        
        return steps

    def ddim_sample_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Single step of the DDIM sampling process: p_θ(x_{t-1}|x_t)

        The DDIM sampling equation is:
        x_{t-1} = √(α_{t-1}) * x_0_predicted + √(1-α_{t-1}) * ε_θ + σ_t * ε_t

        where:
        - x_0_predicted = (x_t - √(1-α_t) * ε_θ) / √(α_t)
        - σ_t = η * √((1-α_{t-1})/(1-α_t)) * √(1-α_t/α_{t-1})
        - ε_t is random noise
        - η (eta) is a hyperparameter

        Args:
            x_t (torch.Tensor): Input tensor at time t
            t (torch.Tensor): Current timestep
            t_prev (torch.Tensor): Previous timestep

        Returns:
            torch.Tensor: Sample x_{t-1}
        """
        # Predict noise
        eps_theta = self.predict_noise(x_t, t)

        # Get alphas for current and previous timesteps
        alpha_bar_t = self.alpha_bar.index_select(0, t)
        alpha_bar_t_prev = self.alpha_bar.index_select(0, t_prev)

        # Reshape for broadcasting
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (len(x_t.shape) - 1)))
        alpha_bar_t_prev = alpha_bar_t_prev.view(-1, *([1] * (len(x_t.shape) - 1)))

        # Predict x0
        x0_predicted = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
        
        # DDIM deterministic part
        x_t_prev = (
            torch.sqrt(alpha_bar_t_prev) * x0_predicted +
            torch.sqrt(1 - alpha_bar_t_prev) * eps_theta
        )

        # Add noise if eta > 0
        if self.ddim_eta > 0:
            # Calculate sigma_t
            sigma_t = (
                self.ddim_eta *
                torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) *
                torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            )
            
            # Add noise term
            eps_t = torch.randn_like(x_t)
            noise_term = torch.nan_to_num(sigma_t * eps_t, 0.0)
            x_t_prev = x_t_prev + noise_term

        return x_t_prev

    def sample_backforward(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate samples by running the complete reverse diffusion process.
        
        Args:
            shape (Tuple[int, ...]): Shape of the samples to generate [B, C, H, W]
            
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W]
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Iterate through reverse diffusion process
        for idx in range(self.ddim_sampling_steps):
            # Calculate reverse index
            reverse_idx = self.ddim_sampling_steps - 1 - idx
            
            # Get current and previous timesteps
            t = self.ddim_timesteps[reverse_idx]
            t_prev = 0 if reverse_idx == 0 else self.ddim_timesteps[reverse_idx - 1]
            
            # Create tensors for timesteps
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            t_prev_tensor = torch.full((shape[0],), t_prev, device=self.device, dtype=torch.long)

            # Single step of DDIM sampling
            x = self.ddim_sample_step(x_t=x, t=t_tensor, t_prev=t_prev_tensor)
            
        return x