# pyright: reportInvalidTypeForm=false
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUnusedImport=false
# ruff: noqa: F401, E741

from typing import Dict, List, Optional, Callable, NewType, TypeVar, Tuple
from sympy import random_poly
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb

# torch sets default dtype to double
torch.set_default_dtype(torch.float64)


class SO3Config:
    """SO(3)上各种物理量的类型定义和约束"""

    # 定义类型别名

    basis: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    R = NewType("R", torch.Tensor)  # 旋转矩阵, 属于SO(3)流形, shape=(..., 3, 3)
    tangent = NewType(
        "tangent", torch.Tensor
    )  # so(3)李代数中的元素(反对称矩阵), shape=(..., 3, 3)
    v = NewType("v", torch.Tensor)  # 旋转向量, 属于R^3, shape=(..., 3)
    omega = NewType("omega", torch.Tensor)  # 旋转角度标量, 属于R^+, shape=(...)


class SO3Algebra:
    """SO(3)的李代数相关操作"""

    @staticmethod
    def hat(v: SO3Config.v) -> SO3Config.tangent:
        """hat 映射: R^3 -> so(3)"""
        x1, x2, x3 = v[..., 0], v[..., 1], v[..., 2]
        zeros = torch.zeros_like(x1)
        X = torch.stack(
            [
                torch.stack([zeros, -x3, x2], dim=-1),
                torch.stack([x3, zeros, -x1], dim=-1),
                torch.stack([-x2, x1, zeros], dim=-1),
            ],
            dim=-2,
        )
        return SO3Config.tangent(X)

    @staticmethod
    def vee(X: SO3Config.tangent) -> SO3Config.v:
        """vee 映射: so(3) -> R^3"""
        v = torch.stack(
            [
                X[..., 2, 1],  # x1
                X[..., 0, 2],  # x2
                X[..., 1, 0],  # x3
            ],
            dim=-1,
        )
        return SO3Config.v(v)

    @staticmethod
    def Log(R: SO3Config.R) -> SO3Config.v:
        """
        Log 映射: SO(3) -> R^3
        支持批量处理: R的shape可以是(..., 3, 3)
        返回的向量shape为(..., 3)
        """
        theta = SO3Algebra.Omega(R)  # shape: (...)
        eps = 1e-10
        mask = theta < eps

        # 添加维度以便正确广播
        theta = theta[..., None]  # shape: (..., 1, 1)
        mask = mask[..., None]  # shape: (..., 1, 1)

        sin_theta = torch.where(mask, 1.0, torch.sin(theta))
        factor = torch.where(mask, 0.5, 0.5 * theta / sin_theta)

        R_minus_RT = R - R.transpose(-2, -1)

        # 提取向量分量并确保维度匹配
        v = factor * torch.stack(
            [
                R_minus_RT[..., 2, 1],  # shape: (...)
                R_minus_RT[..., 0, 2],
                R_minus_RT[..., 1, 0],
            ],
            dim=-1,
        )  # 最终shape: (..., 3)

        return SO3Config.v(v)

    @staticmethod
    def log(R: SO3Config.R) -> SO3Config.tangent:
        """Log 映射: SO(3) ->so(3)"""
        return SO3Config.tangent(SO3Algebra.hat(SO3Algebra.Log(R)))

    @staticmethod
    def exp(tangent: SO3Config.tangent) -> SO3Config.R:
        """exp 映射: so(3) -> SO(3)"""
        result = torch.linalg.matrix_exp(tangent)
        return SO3Config.R(result)

    @staticmethod
    def Exp(v: SO3Config.v) -> SO3Config.R:
        """Exp 映射: R^3 -> SO(3)"""
        return SO3Algebra.exp(SO3Algebra.hat(v))

    @staticmethod
    def Omega(R: SO3Config.R) -> SO3Config.omega:
        """Omega 映射: SO(3) -> R^+

        支持批量处理: R的shape可以是(..., 3, 3)
        返回的角度shape为(...)
        """
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # shape: (...)
        theta = torch.arccos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

        return SO3Config.omega(theta)

    @staticmethod
    def expmap(R0: SO3Config.R, tangent: SO3Config.tangent) -> SO3Config.R:
        """expmap 映射: SO(3)xso(3) -> SO(3)"""
        skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)

        return torch.einsum("...ij,...jk->...ik", R0, torch.linalg.matrix_exp(skew_sym))

    @staticmethod
    def angle_density_unif(omega: SO3Config.omega) -> torch.Tensor:
        """计算SO(3)上均匀分布的角度密度"""
        return (1 - torch.cos(omega)) / np.pi

    @staticmethod
    def tangent_gaussian(R0: SO3Config.R) -> SO3Config.tangent:
        """生成R0处切空间中的高斯随机向量"""
        # 生成R^3上的标准高斯噪声并映射到so(3)
        v = torch.randn(R0.shape[:-2] + (3,), dtype=torch.float64)
        X = SO3Algebra.hat(v)  # 从R^3映射到so(3)
        X = torch.einsum("...ij,...jk->...ik", R0, X)  # 左传输到R0处的切空间
        return SO3Config.tangent(X)

    @staticmethod
    def riemannian_gradient(
        f: Callable[[SO3Config.R], torch.Tensor], R: SO3Config.R
    ) -> SO3Config.tangent:
        """计算SO(3)上函数f在点R处的黎曼梯度

        Args:
            f: SO(3)上的标量函数
            R: SO(3)中的点
        Returns:
            grad_R: f在R处的黎曼梯度，属于R处的切空间
        """
        coefficients = torch.zeros(
            list(R.shape[:-2]) + [3], requires_grad=True, dtype=R.dtype, device=R.device
        )

        # 计算扰动后的旋转矩阵
        R_delta = SO3Algebra.expmap(
            R0=R,
            tangent=torch.einsum("...ij,...jk->...ik", R, SO3Algebra.hat(coefficients)),
        )

        # 计算梯度
        grad_coefficients = torch.autograd.grad(f(R_delta).sum(), coefficients)[0]

        # 将梯度投影到切空间（确保反对称性）
        X = SO3Algebra.hat(grad_coefficients)  # 转换为反对称矩阵
        grad_R = torch.einsum("...ij,...jk->...ik", R, X)

        # 确保结果是反对称的
        grad_R = 0.5 * (grad_R - grad_R.transpose(-2, -1))

        return SO3Config.tangent(grad_R)


class IGSO3Diffusion:
    def __init__(
        self,
        L: int,
        N_atoms: int,
        M: int = 1000,
        N: int = 10000,
        p_initial: Optional[Callable[[], SO3Config.R]] = None,
        drift: Optional[Callable[[SO3Config.R, float], SO3Config.tangent]] = None,
        T: float = 5.0,
    ):
        self.L = L
        self.N_atoms = N_atoms
        self.M = M
        self.N = N
        self.p_initial = p_initial
        self.drift = drift
        self.T = T

        self.mu_ks = self.p_inv(self.N_atoms)
        self.ts = np.linspace(0, self.T, 200)

    def p_inv(self, N) -> SO3Config.R:
        """从 U(SO(3)) 采样"""
        omega_grid = torch.linspace(0, np.pi, self.M)
        cdf = np.cumsum(
            SO3Algebra.angle_density_unif(SO3Config.omega(omega_grid)).numpy()
        ) / (self.M / np.pi)

        omegas = np.interp(np.random.rand(N), cdf, omega_grid)
        
        axes = np.random.randn(N, 3)
        
        axes = omegas[:, None] * axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        
        # # ipdb.set_trace()
        return SO3Config.R(SO3Algebra.Exp(SO3Config.v(torch.tensor(axes))))

    def p_0(self) -> SO3Config.R:
        """采样初始分布"""
        return SO3Config.R(self.mu_ks[torch.randint(self.mu_ks.shape[0], size=[self.N])])
        


    def p_t(self, Rt: SO3Config.R, t: float) -> torch.Tensor:
        """计算时间t时的概率密度"""
        # ipdb.set_trace()
        return (
            sum(
                [
                    self.igso3_density(
                        SO3Config.R(torch.einsum("ij,...jk->...ik", mu_k, Rt)), t
                    )
                    for mu_k in self.mu_ks
                ]
            )
            / self.N_atoms
        )

    def igso3_density(self, Rt: SO3Config.R, t: float) -> torch.Tensor:
        return self.f_igso3(SO3Algebra.Omega(Rt), t, self.L)

    def f_igso3(self, omega: SO3Config.omega, t: float, L: int = 500) -> torch.Tensor:
        ls: torch.Tensor = torch.arange(L)[None]
        return (
            (2 * ls + 1)
            * torch.exp(-ls * (ls + 1) * t / 2)
            * torch.sin(omega[:, None] * (ls + 1 / 2))
            / torch.sin(omega[:, None] / 2)
        ).sum(dim=-1)

    def geodesic_random_walk(
        self,
        p_initial: Optional[Callable[[], SO3Config.R]] = None,
        drift: Optional[Callable[[SO3Config.R, float], SO3Config.tangent]] = None,
        ts: Optional[np.ndarray] = None,
    ) -> Dict[float, SO3Config.R]:
        """测地线随机游走"""
        p_initial_fn = p_initial if p_initial is not None else self.p_initial
        drift_fn = drift if drift is not None else self.drift
        ts_fn = ts if ts is not None else self.ts

        Rts: Dict[float, SO3Config.R] = {ts_fn[0]: p_initial_fn()}
        # ipdb.set_trace()
        for i in range(1, len(ts_fn)):
            dt = ts_fn[i] - ts_fn[i - 1]
            Rt_prev = Rts[ts_fn[i - 1]]
            # ipdb.set_trace()
            drift_term = (
                drift_fn(Rt_prev, ts_fn[i - 1]) * dt
                if drift_fn
                else SO3Config.tangent(torch.zeros_like(Rt_prev))
            )
            noise_term = SO3Algebra.tangent_gaussian(Rt_prev) * np.sqrt(abs(dt))

            Rts[ts_fn[i]] = SO3Config.R(
                SO3Algebra.expmap(Rt_prev, drift_term + noise_term)
            )
        # ipdb.set_trace()
        return Rts

    def score_t(self, Rt: SO3Config.R, t: float) -> SO3Config.tangent:
        """计算score function"""
        # ipdb.set_trace()
        return SO3Config.tangent(
            SO3Algebra.riemannian_gradient(lambda R: torch.log(self.p_t(R, t)), Rt)
        )


def visualize_random_walk(igso3: IGSO3Diffusion, save_folder: str):
    """可视化测地线随机游走的结果"""

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    random_walk = igso3.geodesic_random_walk(
        p_initial=lambda: SO3Algebra.exp(torch.zeros(igso3.N, 3, 3)),
        drift=lambda Rt, t: 0.0,
    )


    t_idcs_plot = [5, 25, 100, -1]
    fig, axs = plt.subplots(
        1, len(t_idcs_plot), dpi=100, figsize=(3 * len(t_idcs_plot), 3)
    )
    # # ipdb.set_trace()

    for i, t_idx in enumerate(t_idcs_plot):
        t = igso3.ts[t_idx]
        # 从测地线随机游走中绘制旋转角度的经验分布
        bins = np.linspace(0, np.pi, 20)
        axs[i].hist(
            SO3Algebra.Omega(random_walk[t]),
            bins=bins,
            density=True,
            histtype="step",
            label="Empirical",
        )
        axs[i].set_title(f"t={t:0.01f}")

        # 计算旋转角度的密度，以及均匀分布的密度
        omega_grid = torch.linspace(0, np.pi, 1000)
        
        pdf_angle = igso3.f_igso3(omega_grid, t) * SO3Algebra.angle_density_unif(
            omega_grid
        )
        # # ipdb.set_trace()
        axs[i].plot(omega_grid, pdf_angle.numpy(), label="IGSO3(*;I, t)")
        axs[i].plot(
            omega_grid,
            SO3Algebra.angle_density_unif(omega_grid).numpy(),
            label="Uniform",
        )

        axs[i].set_xlabel("Angle of rotation (radians)")
    axs[0].legend()
    axs[0].set_ylabel("p(angle)")

    plt.suptitle("Agreement of IGSO3 density and geodesic random walk", y=1.05)
    plt.tight_layout()
    plt.savefig(save_folder / "igso3_density_and_geodesic_random_walk.png")
    plt.show()


def visualize_forward_reverse(igso3: IGSO3Diffusion, save_folder: str):
    """可视化前向和反向过程"""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    def p_inv(N: int, M: int = 1000) -> torch.Tensor:
        omega_grid: torch.Tensor = torch.linspace(0, np.pi, M)
        cdf: np.ndarray = np.cumsum(SO3Algebra.angle_density_unif(omega_grid).numpy()) / (M / np.pi)
        omegas: np.ndarray = np.interp(np.random.rand(N), cdf, omega_grid)
        axes: np.ndarray = np.random.randn(N, 3)
        axes = omegas[:, None] * axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        return SO3Config.R(SO3Algebra.Exp(SO3Config.v(torch.tensor(axes))))
    mu_ks = p_inv(igso3.N_atoms)
    def p_0(N:int):
        return mu_ks[torch.randint(mu_ks.shape[0], size=[N])]
    
    def p_t(Rt: torch.Tensor, t: float) -> torch.Tensor:
        
        return (
            sum(
                [
                    igso3.igso3_density(torch.einsum("ji,...jk->...ik", mu_k, Rt), t)
                    for mu_k in mu_ks
                ]
            )
            / igso3.N_atoms
        )
    
    def score_t(Rt: torch.Tensor, t: float,) -> torch.Tensor:
        return SO3Algebra.riemannian_gradient(lambda R: torch.log(p_t(R, t)), Rt)
    
    T = 5.0
    N = 5000
    ts: np.ndarray = np.linspace(0, T, 200) 
    
    # 模拟前向过程
    forward_samples = igso3.geodesic_random_walk(
        p_initial=lambda: p_0(N), drift=lambda Rt, t: 0.0,ts=ts
    )

    # 模拟反向过程
    reverse_samples = igso3.geodesic_random_walk(
        p_initial=lambda: p_inv(N),
        drift=lambda Rt, t: -score_t(Rt, t),
        ts=ts[::-1],
    )  

    # 可视化前向和反向过程在几个时间步的边际分布
    for i in [3, 5, 10, 25, 50, 100]:
        t = igso3.ts[i]
        Rt_forward = SO3Algebra.Log(forward_samples[t])
        Rt_reverse = SO3Algebra.Log(reverse_samples[t])

        # ipdb.set_trace()
        
        fig, axs = plt.subplots(1, 3, dpi=100, figsize=(9, 3))
        fig.suptitle(f"t={t:.1f}")
        bins = np.linspace(-np.pi, np.pi, 15)

        for Rt, label in [(Rt_forward, "forward"), (Rt_reverse, "reverse")]:
            for j in range(3):
                axs[j].hist(
                    Rt[:, j].numpy(), bins, density=True, histtype="step", label=label
                )
                axs[j].set_xlabel(f"e{j}")
                axs[j].set_ylabel("density")
        plt.legend()
        plt.savefig(save_folder / f"forward_reverse_distribution_t={t:.1f}.png")
        plt.close()


if __name__ == "__main__":
    # 创建IGSO3Diffusion实例
    igso3_diffusion = IGSO3Diffusion(L=500, N_atoms=3, M=1000, N=5000, T=5.0)
    # ipdb.set_trace()    

    # 可视化测地线随机游走
    visualize_random_walk(igso3_diffusion, save_folder="results_v3")

    # 可视化前向和反向过程
    visualize_forward_reverse(igso3_diffusion, save_folder="results_v3")
