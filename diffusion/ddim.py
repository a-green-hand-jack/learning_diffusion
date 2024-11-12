import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

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
        device (torch.device): Device to run the model on
    """

    def __init__(
        self,
        eps_model: nn.Module,
        n_step: int,
        ddim_sampling_steps: int,
        ddim_discretize: str = 'uniform',
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        ),
        ddim_eta: float = 0.0,
    ):
        super().__init__(eps_model, n_step, device)

        # Create DDIM timestep sequence
        self.ddim_sampling_steps = ddim_sampling_steps
        self.ddim_timesteps = self.get_ddim_timesteps(ddim_discretize)

        self.ddim_eat = ddim_eta
        
    def get_ddim_timesteps(self, ddim_discretize: str = 'uniform') -> torch.Tensor:
        """
        Get DDIM timesteps based on different discretization strategies.
        
        Args:
            ddim_discretize (str): Discretization method ('uniform' or 'quad')
            
        Returns:
            torch.Tensor: DDIM timesteps
        """
        if ddim_discretize == 'uniform':
            # Uniform spacing
            c = self.ddim_sampling_steps // self.n_step
            steps = np.asarray(list(range(0, self.n_step, c))) + 1
            
        elif ddim_discretize == 'quad':
            # Quadratic spacing
            steps = ((np.linspace(0, np.sqrt(self.n_step * 0.8), self.ddim_sampling_steps)) ** 2).astype(int) + 1
            
        else:
            raise NotImplementedError(f"Discretization method {ddim_discretize} not implemented")
        
        return torch.from_numpy(steps).to(self.device)

    def ddim_sample_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Single step of the reverse diffusion process or DDIM sampling process: p_θ(x_{t-1}|x_t)

        x_{t-1} =
            \sqrt{\aplha_{t-1}} * (x_t - \sqrt{1-\alpht_t}\epslion_\theta(x_t)) / (\sqrt{\alpha_t})
            +
            \sqrt{1-\alpha_{t-1} - \alpha_{t}^2} * \epslion_\theta(x_t)
            +
            \sigma_t * \epslion_t

        where \epslion_t is random noise and

        \simga_t = \niu * \sqrt{(1-\alpha_{t-1} / (1-\alpha_t))} * \sqrt{1 - \alpha_t / \alpha_{t-1}}

        where \niu is a hyperparameter

        Here, the **t** is "t" and the **t-1** is t_prev, it's just too trouble to write the later former in \LaTex.

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

        # DDIM sampling equations
        x0_predicted = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(
            alpha_bar_t
        )

        # Clip predicted x0 to [-1, 1] to improve sample quality
        x0_predicted = torch.clamp(x0_predicted, -1.0, 1.0)

        # DDIM deterministic formula
        x_t_prev = (
            torch.sqrt(alpha_bar_t_prev) * x0_predicted
            + torch.sqrt(1 - alpha_bar_t_prev) * eps_theta
        )

        # Add noise with \niu = ddim_eta
        sigma_t = (
            self.ddim_eat
            * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
        )
        eps_t = torch.randn_like(x_t)

        return x_t_prev + eps_t * sigma_t

    def sample_backforward(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate samples using the DDIM sampling process.
        Overrides the DDPM sampling method.

        Args:
            shape (Tuple[int, ...]): Shape of the samples to generate [B, C, H, W]

        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W]
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Iterate through reverse diffusion process using DDIM timesteps
        for i in range(len(self.ddim_timesteps) - 1):
            t = self.ddim_timesteps[i]
            t_prev = self.ddim_timesteps[i + 1]

            # Create batch of current timestep
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            t_prev_tensor = torch.full(
                (shape[0],), t_prev, device=self.device, dtype=torch.long
            )

            # Single step of DDIM sampling
            x = self.ddim_sample_step(x_t=x, t=t_tensor, t_prev=t_prev_tensor)

        return x
