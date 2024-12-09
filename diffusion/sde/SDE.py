"""抽象 SDE 类，以及几种具体的 SDE 实现"""

import abc
import torch
import numpy as np

def get_sde(name: str) -> type["BaseSDE"]:
    if name == "VPSDE":
        return VPSDE
    elif name == "VESDE":
        return VESDE
    elif name == "SubVPSDE":
        return SubVPSDE
    else:
        raise ValueError(f"未知的SDE类型: {name}")


class BaseSDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """在前向过程中计算 drift 和 diffusion"""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """确定 SDE 的边缘分布的参数，$p_t(x)$， 也就是从 x_0 一步得到 x_t 的分布的参数
        其实是描述了 扰动核 perturbation kernel
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """生成从 $p_T(x)$ 中采样的样本"""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """计算 $p_T(x)$ 的 log 概率密度；不过我还没发现它应该在哪里使用

        在概率流 ODE 中，这个函数用于计算对数似然。
        参数：
          z: 隐变量
        返回：
            log 概率密度
        """
        pass

    def discretize(self, x, t):
        """离散形式的 SDE：x_{i+1}=x_i+f_i（x_i）+G_i z_i。

        用于正向扩散采样和概率流采样。

        参数：
          x：一个 torch 张量
          t：一个代表时间步长的 torch 浮点数（从0到'self. T'）

        返回：
          f、G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return (
            f,
            G,
        )  # f(x, t) * dt 和 g(t) * sqrt(dt)； 值得注意的是这里本来应该是 g(t) * dw 但是因为 dw 是标准正态分布，所以可以简化为 g(t) * sqrt(dt)

    def reverse(self, score_fn, probability_flow=False):
        """
        创立反向的 SDE 或 ODE
        放在这里就是为了保证前向过程和反向过程是一直的
        因为在建立反向过程的时候仅仅是补充了 score_fn = s_\theta(x, t) 而已

        参数：
          score_fn: 一个依赖于时间的时间依赖的 score-based 模型，它接受 x 和 t 并返回分数。
          probability_flow: 如果为 True，则创建用于概率流采样的反向时间 ODE。
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """为反向 SDE 或 ODE 创建漂移和扩散函数"""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                diffusion = (
                    0.0 if self.probability_flow else diffusion
                )  # 如果概率流为真，则将扩散函数设置为零，也就是消除了噪音
                return drift, diffusion

            def discretize(self, x, t):
                """为了反向 SDE 或 ODE 的离散化的采样"""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = (
                    torch.zeros_like(G) if self.probability_flow else G
                )  # 如果概率流为真，则将扩散函数设置为零
                return rev_f, rev_G

        return RSDE()

    def score(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算score function ∇_x(t)log p(x(t)|x(0))
        
        Args:
            x0: 原始数据
            xt: 扰动后的数据
            t: 时间点
        
        Returns:
            score值
        """
        raise NotImplementedError("每个具体的SDE子类都需要实现这个方法")
    
    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """计算损失函数的权重系数 λ(t)
        
        Args:
            t: 时间点
            
        Returns:
            权重系数
        """
        raise NotImplementedError("每个具体的SDE子类都需要实现这个方法")

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """获取边际分布的标准差
        
        Args:
            t: 时间点
            
        Returns:
            对应时间点的标准差
        """
        # 使用一个空的张量来获取标准差
        # 因为我们只关心标准差，而标准差与输入x_0无关
        _, std = self.marginal_prob(torch.zeros((1,), device=t.device), t)
        return std


class VPSDE(BaseSDE):
    """
    SDE 的特殊情况，其中漂移和扩散函数是线性的
    Variance Preserving SDE

    dx = f(x, t) dt + g(t) dw
      = - 1 / 2 * \beta(t) * x dt + \sqrt(\beta(t)) * dw_t
      = - 1 / 2 * \beta(t) * x dt + \sqrt(\beta(t)) * \sqrt(dt)
    beta(t) = b_min + (b_max - b_min) * t for t in [\epsilon, 1], \epsilon = 1e-5
    """

    def __init__(self, N, b_min, b_max):
        super().__init__(N)

        self.b_min = b_min
        self.b_max = b_max
        self.N = N

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x, t):
        """计算漂移和扩散系数"""
        beta = self.b_min + (self.b_max - self.b_min) * t
        drift = -0.5 * beta * x
        diffusion = beta**0.5
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """计算边缘分布的参数, 这样就可以从 x_0 一步得到 x_t 的分布的参数， 其实就是 x_t 的均值和方差.
        这是对应的数学表达：
                        mu = exp(log_mean_coeff) * x_0
                        sigma = sqrt(1 - exp(2 * log_mean_coeff))

        其实是描述了 扰动核 perturbation kernel"""
        log_mean_coeff = -0.25 * t**2 * (self.b_max - self.b_min) - 0.5 * t * self.b_min
        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """从 $p_T(x)$ 中采样, 其实就是得到了对应形状的标准正态分布的样本"""
        return torch.randn(*shape, device=self.device)

    def prior_logp(self, z):
        """
        这个函数是在计算标准正态分布(standard normal distribution)的对数概率密度函数(log probability density function)。

        让我解析一下这个函数：

        1) 输入z是一个张量，形状可能是(batch_size, channels, height, width)

        2) N = np.prod(shape[1:]) 计算的是除了batch维度外所有维度的乘积，即总特征数

        3) 标准正态分布的概率密度函数是：
        p(x) = (1/√(2π)) * exp(-x²/2)

        4) 取对数后变成：
        log p(x) = -1/2 * log(2π) - x²/2

        5) 对于多维的情况，因为假设各维度独立，所以：
        - 第一项变成 -N/2 * log(2π)
        - 第二项变成所有维度的平方和的负半值 -∑(x²)/2

        这个函数主要用于：
        - 评估模型生成的样本的似然度
        - 在训练过程中作为正则化项
        - 在扩散模型中，它表示噪声分布的对数概率密度

        你说得对，这个函数与λ(t)没有直接关系。它是用来计算最终时刻T的先验分布(prior distribution)的对数概率密度，而λ(t)是描述扩散过程中噪声添加速率的函数。
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def score(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # VP-SDE的特定实现
        sigma = self.marginal_std(t)
        sigma = sigma.view(-1, 1, 1, 1)
        return (x0 - xt) / (sigma ** 2)
    
    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        # VP-SDE的权重实现
        return self.marginal_std(t) ** 2


class VESDE(BaseSDE):
    """
    SDE 的特殊情况，其中漂移和扩散函数是线性的
    Variance Exploding SDE

    dx = \sqrt(d[\sigma(t)^2] / dt) * dw
           = \sqrt(d[\sigma(t)^2] / dt) * \sqrt(dt)

    sigma(t) = sigma_min * (sigma_max / sigma_min) ^ t
    """

    def __init__(self, N, sigma_min, sigma_max):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x, t):
        """计算漂移和扩散"""

        drift = torch.zeros_like(x)

        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * torch.sqrt(
            2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))
        )
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """计算边缘分布的参数, 这样就可以从 x_0 一步得到 x_t 的分布的参数， 其实就是 x_t 的均值和方差。
        对应是数学公式是：
        mu = x_0
        std = sigma_min * (sigma_max / sigma_min) ^ t
        """
        mean = x_0
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return mean, std

    def prior_sampling(self, shape):
        """从 $p_T(x)$ 中采样, 其实就是得到了对应形状的标准正态分布的样本
        这是因为在VESDE (Variance Exploding SDE) 中，`sigma_max`代表了扩散过程最终状态的噪声水平。让我解释一下：

                1. 在VESDE中，扩散过程是从原始数据逐渐增加噪声，噪声水平从`sigma_min`增加到`sigma_max`

                2. `prior_sampling`函数的目的是从扩散过程的最终分布p_T(x)中采样，这个分布应该对应于扩散过程t=T时的状态

                3. 根据VESDE的定义，在t=T=1时：
                - 边缘分布是高斯分布
                - 均值为0
                - 标准差为sigma_max

                4. 因此，要从这个分布中采样，我们需要：
                ```python
                torch.randn(*shape) * self.sigma_max
                ```
                - `torch.randn(*shape)`生成标准正态分布样本（均值0，标准差1）
                - 乘以`self.sigma_max`将标准差缩放到正确的水平

                这与`marginal_prob`方法是一致的，当t=1时，标准差正好是`sigma_max`。这确保了采样得到的样本确实来自扩散过程的最终分布。
        """
        return torch.randn(*shape, device=self.device) * self.sigma_max

    def prior_logp(self, z):
        """计算标准正态分布的对数概率密度函数"""
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2.0 * self.sigma_max**2)
        return logps

    def score(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # VE-SDE的特定实现
        sigma = self.marginal_std(t)
        sigma = sigma.view(-1, 1, 1, 1)
        return (x0 - xt) / (sigma ** 2)
    
    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        # VE-SDE的权重实现
        return torch.ones_like(t)


class SubVPSDE(VPSDE):
    """
    SDE 的特殊情况，其中漂移和扩散函数是线性的
    Sub-Variance Preserving SDE
    """

    def __init__(
        self,
        N,
        b_min,
        b_max,
    ):
        super().__init__(N, b_min, b_max)

    def sde(self, x, t):
        beta_t = self.b_min + (self.b_max - self.b_min) * t
        drift = -0.5 * beta_t * x
        tmp = 1 - torch.exp(
            -2 * (self.b_min * t + (self.b_max - self.b_min) * t**2 / 2)
        )
        diffusion = torch.sqrt(tmp * beta_t)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """计算边缘分布的参数, 这样就可以从 x_0 一步得到 x_t 的分布的参数， 其实就是 x_t 的均值和方差。
        对应是数学公式是：
        log_mean_coeff = -0.25 * t**2 * (self.b_max - self.b_min) - 0.5 * t * self.b_min
        mean = x_0 * torch.exp(log_mean_coeff)
        std = 1 - torch.exp(2 * log_mean_coeff)
        """
        log_mean_coeff = -0.25 * t**2 * (self.b_max - self.b_min) - 0.5 * t * self.b_min
        mean = x_0 * torch.exp(log_mean_coeff)
        std = 1 - torch.exp(2 * log_mean_coeff)

        return mean, std


