import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation.
    
    The diffusion process gradually adds Gaussian noise to data through T steps, 
    while the denoising process learns to reverse this procedure.
    
    Forward process (diffusion): q(x_t|x_{t-1})
    Reverse process (denoising): p_θ(x_{t-1}|x_t)
    
    Args:
        eps_model (nn.Module): Neural network that predicts noise ε
        n_step (int): Number of diffusion steps
        device (torch.device): Device to run the model on
    """

    def __init__(
        self,
        eps_model: nn.Module,
        n_step: int,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.eps_model = eps_model
        self.n_step = n_step
        self.device = device

        # Define beta schedule
        self.beta = torch.linspace(1e-5, 0.02, n_step).to(self.device)  # β_t
        self.alpha = 1 - self.beta  # α_t = 1 - β_t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # ᾱ_t = ∏_{s=1}^t α_s
        self.sigma2 = self.beta  # σ²_t = β_t for simplified variance

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε_θ(x_t, t) using the noise prediction network.
        
        Args:
            x_t (torch.Tensor): Input tensor of shape [B, C, H, W]
            t (torch.Tensor): Time steps tensor of shape [B]
            
        Returns:
            torch.Tensor: Predicted noise of shape [B, C, H, W]
        """
        return self.eps_model(x_t, t)

    def sample_forward(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t|x_0).
        
        Given x_0 and t, sample x_t using the formula:
        x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε, where ε ~ N(0, I)
        
        Args:
            x0 (torch.Tensor): Initial data tensor of shape [B, C, H, W]
            t (torch.Tensor): Time steps tensor of shape [B]
            eps (torch.Tensor, optional): Optional noise tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Noisy sample x_t of shape [B, C, H, W]
        """
        if eps is None:
            eps = torch.randn_like(x0)  # Sample from N(0, I)
        
        # Ensure correct shapes for broadcasting
        alpha_bar_t = self.alpha_bar.index_select(0, t)
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (len(x0.shape) - 1)))
        
        # q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
        mean = torch.sqrt(alpha_bar_t) * x0
        var = 1 - alpha_bar_t
        
        return mean + torch.sqrt(var) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the simplified loss L = E[||ε - ε_θ(x_t, t)||²].
        
        Args:
            x0 (torch.Tensor): Initial data tensor of shape [B, C, H, W]
            noise (torch.Tensor, optional): Optional noise tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_step, (batch_size,), device=self.device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get noisy sample x_t
        x_t = self.sample_forward(x0, t, eps=noise)
        # Predict noise
        eps_theta = self.predict_noise(x_t, t)
        
        return F.mse_loss(noise, eps_theta)

    def ddpm_sample_step(
        self, x_t: torch.Tensor, t: torch.Tensor, simple_var: bool = True
    ) -> torch.Tensor:
        """
        Single step of the reverse diffusion process: p_θ(x_{t-1}|x_t).
        
        The reverse process is parameterized as:
        p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t)
        
        where μ_θ = 1/√(α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
        
        Args:
            x_t (torch.Tensor): Input tensor of shape [B, C, H, W]
            t (torch.Tensor): Time steps tensor of shape [B]
            simple_var (bool): Whether to use simplified variance β_t
            
        Returns:
            torch.Tensor: Sample x_{t-1} of shape [B, C, H, W]
        """
        # Predict noise
        eps_theta = self.predict_noise(x_t, t)
        
        # Get required variables and adjust dimensions
        alpha_bar_t = self.alpha_bar.index_select(0, t)
        alpha_t = self.alpha.index_select(0, t)
        beta_t = self.beta.index_select(0, t)
        
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (len(x_t.shape) - 1)))
        alpha_t = alpha_t.view(-1, *([1] * (len(x_t.shape) - 1)))
        beta_t = beta_t.view(-1, *([1] * (len(x_t.shape) - 1)))
        
        # Calculate mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_theta
        )
        
        # Calculate variance
        if simple_var:
            var = beta_t  # Simplified σ²_t = β_t
        else:
            var = (1 - alpha_bar_t) / (1 - alpha_t) * beta_t  # Full variance
        
        # Sample x_{t-1}
        eps = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * eps

    def sample_backforward(
        self, shape: Tuple[int, ...], simple_var: bool = True
    ) -> torch.Tensor:
        """
        Generate samples by running the complete reverse diffusion process.
        
        Args:
            shape (Tuple[int, ...]): Shape of the samples to generate [B, C, H, W]
            simple_var (bool): Whether to use simplified variance
            
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W]
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Iterate through reverse diffusion process
        for t in reversed(range(self.n_step)):
            # Create batch of current timestep
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            # Single step of reverse diffusion
            x = self.ddpm_sample_step(x_t=x, t=t_tensor, simple_var=simple_var)
        
        return x


if __name__ == "__main__":
    # Example usage of torch.gather
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor shape:", x.shape)
    print("Original tensor:\n", x)
    
    indices = torch.tensor([[0, 1], [1, 2]])
    print("\nIndices shape:", indices.shape)
    print("Indices:\n", indices)
    
    out = torch.gather(x, 0, indices)
    print("\nGathered tensor shape:", out.shape)
    print("Gathered tensor:\n", out)