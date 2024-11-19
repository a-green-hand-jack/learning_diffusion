# pyright: reportInvalidTypeForm=false
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUnusedImport=false
# ruff: noqa: F401, E741

from typing import TypeVar, NewType, Dict, List, Optional, Callable
from sqlalchemy import Result
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

# 设置默认数据类型为double
torch.set_default_dtype(torch.float64)

# 定义正交基底 (确保使用float64)
basis = torch.tensor([
    [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
], dtype=torch.float64)

class SO3Config:
    """SO(3)上各种物理量的类型定义和约束"""
    
    # 定义类型别名
    R = NewType('R', torch.Tensor)  # 旋转矩阵: shape (..., 3, 3)
    tangent = NewType('tangent', torch.Tensor)  # 切空间元素: shape (..., 3, 3)
    omega = NewType('omega', torch.Tensor)  # 旋转角度: shape (...)
    axis = NewType('axis', torch.Tensor)  # 旋转轴: shape (..., 3)
    
    @staticmethod
    def check_R(R: R) -> bool:
        """检查是否为有效的旋转矩阵"""
        if not isinstance(R, torch.Tensor):
            return False
        if R.shape[-2:] != (3, 3):
            return False
        # 检查正交性
        I = torch.eye(3, device=R.device)
        is_orthogonal = torch.allclose(
            torch.matmul(R, R.transpose(-2, -1)), 
            I.expand(R.shape[:-2] + (3, 3)),
            atol=1e-6
        )
        # 检查行列式是否为1
        has_det_one = torch.allclose(
            torch.det(R),
            torch.ones(R.shape[:-2], device=R.device),
            atol=1e-6
        )
        return is_orthogonal and has_det_one
    
    @staticmethod
    def check_tangent(X: tangent) -> bool:
        """检查是否为有效的切空间元素(反对称矩阵)"""
        if not isinstance(X, torch.Tensor):
            return False
        if X.shape[-2:] != (3, 3):
            return False
        # 检查反对称性
        return torch.allclose(
            X + X.transpose(-2, -1),
            torch.zeros_like(X),
            atol=1e-6
        )
    
    @staticmethod
    def check_omega(omega: omega) -> bool:
        """检查是否为有效的旋转角度"""
        if not isinstance(omega, torch.Tensor):
            return False
        # 角度应该在[0, π]范围内
        return ((omega >= 0) & (omega <= np.pi)).all()
    
    @staticmethod
    def check_axis(v: axis) -> bool:
        """检查是否为有效的旋转轴(单位向量)"""
        if not isinstance(v, torch.Tensor):
            return False
        if v.shape[-1] != 3:
            return False
        # 检查是否为单位向量
        return torch.allclose(
            torch.norm(v, dim=-1),
            torch.ones(v.shape[:-1], device=v.device),
            atol=1e-6
        )

    @staticmethod
    def random_R(batch_shape=()):
        """生成随机旋转矩阵"""
        # 使用QR分解生成随机正交矩阵
        A = torch.randn(batch_shape + (3, 3))
        Q, R = torch.linalg.qr(A)
        # 确保行列式为1
        det = torch.det(Q)
        Q[det < 0] *= -1
        return SO3Config.R(Q)

    @staticmethod
    def random_tangent(batch_shape=()):
        """生成随机切空间元素"""
        v = torch.randn(batch_shape + (3,))
        return SO3Config.tangent(SO3Algebra.hat(v))

class SO3Algebra:
    """SO(3)的李代数相关操作"""

    @staticmethod
    def hat(v: torch.Tensor) -> SO3Config.tangent:
        """hat 映射: R^3 -> so(3)"""
        result = torch.einsum("...i,ijk->...jk", v, basis)
        return SO3Config.tangent(result)

    @staticmethod
    def vee(X: SO3Config.tangent) -> torch.Tensor:
        """vee 映射: so(3) -> R^3"""
        return torch.stack([-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]], dim=-1)

    @staticmethod
    def Log(R: SO3Config.R) -> torch.Tensor:
        """对数映射: SO(3) -> R^3"""
        # assert SO3Config.check_R(R), "Input must be a valid rotation matrix"
        return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())

    @staticmethod
    def exp(X: SO3Config.tangent) -> SO3Config.R:
        """指数映射: so(3) -> SO(3)"""
        # assert SO3Config.check_tangent(X), "Input must be a skew-symmetric matrix"
        return SO3Config.R(torch.linalg.matrix_exp(X))

    @staticmethod
    def expmap(R0: SO3Config.R, tangent: SO3Config.tangent) -> SO3Config.R:
        """指数映射: T_{R0}SO(3) -> SO(3)"""
        # assert SO3Config.check_R(R0), "R0 must be a valid rotation matrix"
        # assert SO3Config.check_tangent(tangent), "tangent must be skew-symmetric"
        skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
        # skew_sym = (skew_sym - skew_sym.transpose(-2, -1)) / 2
        return SO3Config.R(torch.einsum("...ij,...jk->...ik", R0, SO3Algebra.exp(skew_sym)))

    @staticmethod
    def Omega(R: SO3Config.R) -> SO3Config.omega:
        """计算旋转矩阵的旋转角度"""
        return SO3Config.omega(
            torch.arccos(
                torch.clamp(
                    (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2,
                    -1, 1
                )
            )
        )

    @staticmethod
    def angle_density_unif(omega: SO3Config.omega) -> torch.Tensor:
        """计算SO(3)上均匀分布的角度密度"""
        return 2 * (1 - torch.cos(omega)) / np.pi

    @staticmethod
    def tangent_gaussian(R0: SO3Config.R) -> SO3Config.tangent:
        """
        生成R0处切空间中的高斯随机向量
        
        参数:
            R0: SO(3)中的参考点, shape (..., 3, 3)
            
        返回:
            R0处切空间中的高斯随机向量
        """
        # 生成R^3上的标准高斯噪声并映射到so(3)
        v = torch.randn(R0.shape[:-2] + (3,), dtype=torch.float64)
        X = SO3Algebra.hat(v)
        # 把X转换为反对称矩阵， 也就是从R^3映射到so(3)
        X = torch.einsum("...ij,...jk->...ik", R0, X)
        # 强制反对称性
        result = (X - X.transpose(-2, -1)) / 2
        
        # 左传输到R0处的切空间
        return SO3Config.tangent(result)
        
    def riemannian_gradient(f, R):
        """
        输入:
            f - 在SO(3)上的标量函数
            R - SO(3)中的点
        输出: f在R处的黎曼梯度
        数学公式: grad_R f = P_R(∂f/∂R)，其中P_R是切空间投影
        示例:
            grad = riemannian_gradient(lambda R: torch.sum(R), R)

        说明:
        1. 使用自动微分计算梯度
        2. 通过hat映射确保梯度在切空间中
        3. 将梯度转换到R处的切空间
        """
        coefficients = torch.zeros(list(R.shape[:-2]) + [3], requires_grad=True)
        R_delta = SO3Algebra.expmap(
            R0=R, tangent=torch.einsum("...ij,...jk->...ik", R, SO3Algebra.hat(coefficients))
        )
        grad_coefficients = torch.autograd.grad(f(R_delta).sum(), coefficients)[0]
        return torch.einsum("...ij,...jk->...ik", R, SO3Algebra.hat(grad_coefficients))
    
    
class IGSO3Diffusion(BaseModel):
    """IGSO3扩散过程的实现"""

    class Config:
        arbitrary_types_allowed = True

    L: int = Field(500, description="展开项数")
    N_atoms: int = Field(10, description="目标测度中的原子数量")
    M: int = Field(1000, description="采样数量")
    mu_ks: torch.Tensor = Field(None, description="目标测度的原子")

    def __init__(self, **data):
        super().__init__(**data)
        # 初始化目标测度原子
        self.mu_ks = self.p_inv(self.N_atoms, M=self.M)

    def f_igso3(self, omega: SO3Config.omega, t: float, L: int = 500) -> torch.Tensor:
        """计算IGSO3核函数"""
        k = torch.arange(L, device=omega.device, dtype=torch.float64)
        coef = (2 * k + 1) * torch.exp(-k * (k + 1) * t / 2)
        return torch.sum(
            coef[:, None] * torch.sin((k[:, None] + 0.5) * omega) / torch.sin(omega / 2),
            dim=0
        )

    def igso3_density(self, R: SO3Config.R, t: float) -> torch.Tensor:
        """计算IGSO3密度"""
        omega = SO3Algebra.Omega(R)
        return self.f_igso3(omega, t, self.L)

    def compute_score(self, R: SO3Config.R, t: float) -> SO3Config.tangent:
        """计算t时刻的评分函数"""
        return SO3Algebra.riemannian_gradient(
            lambda R_: torch.log(self.sample_pt(R_, t)), R
        )

    def geodesic_random_walk(
        self,
        p_initial: Callable[[], SO3Config.R],
        drift: Callable[[SO3Config.R, float], SO3Config.tangent],
        ts: np.ndarray,
    ) -> Dict[float, SO3Config.R]:
        """实现测地线随机游走"""
        Rts = {ts[0]: p_initial()}
        
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            # 分别计算漂移项和扩散项
            drift_term = drift(Rts[ts[i - 1]], ts[i - 1])
            diffusion_term = SO3Algebra.tangent_gaussian(Rts[ts[i - 1]])
            
            # 将标量乘法应用到向量形式
            drift_vec = SO3Algebra.vee(drift_term)
            diffusion_vec = SO3Algebra.vee(diffusion_term)
            
            # 合并并重新转换为切空间元素
            combined_vec = drift_vec * dt + diffusion_vec * np.sqrt(abs(dt))
            tangent = SO3Algebra.hat(combined_vec)
            
            # 使用指数映射更新位置
            Rts[ts[i]] = SO3Algebra.expmap(
                R0=Rts[ts[i - 1]],
                tangent=tangent
            )

        return Rts

    def sample_p0(self, N: int) -> SO3Config.R:
        """从离散目标测度中采样"""
        idx = torch.randint(len(self.mu_ks), size=(N,))
        return self.mu_ks[idx]

    def sample_pt(self, Rt: SO3Config.R, t: float) -> torch.Tensor:
        """计算t时刻的采样概率"""
        return sum(
            self.igso3_density(torch.einsum("ji,...jk->...ik", mu_k, Rt), t)
            for mu_k in self.mu_ks
        ) / self.N_atoms

    @staticmethod
    def p_inv(N: int, M: int = 1000) -> SO3Config.R:
        """生成SO(3)上均匀分布的旋转矩阵样本"""
        omega_grid = torch.linspace(0, np.pi, M, dtype=torch.float64)
        cdf = np.cumsum(SO3Algebra.angle_density_unif(omega_grid).numpy()) / (M / np.pi)
        
        # 使用逆变换采样生成角度
        omegas = np.interp(np.random.rand(N), cdf, omega_grid.numpy())
        
        # 生成随机旋转轴
        axes = np.random.randn(N, 3)
        axes = omegas[:, None] * axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        
        # 确保转换为float64
        return SO3Config.R(SO3Algebra.exp(SO3Algebra.hat(torch.tensor(axes, dtype=torch.float64))))

def plot_angle_distributions(
    forward_samples: Dict[float, SO3Config.R],
    reverse_samples: Dict[float, SO3Config.R],
    ts: np.ndarray,
    t_idcs_plot: List[int],
    save_path: Optional[str] = None,
) -> None:
    """绘制不同时间点的角度分布"""
    _, axs = plt.subplots(1, len(t_idcs_plot), dpi=100, figsize=(3 * len(t_idcs_plot), 3))
    if len(t_idcs_plot) == 1:
        axs = [axs]

    for i, t_idx in enumerate(t_idcs_plot):
        bins = np.linspace(0, np.pi, 20)
        axs[i].hist(
            SO3Algebra.Omega(forward_samples[ts[t_idx]]),
            bins=bins,
            density=True,
            histtype="step",
            label="Empirical",
        )
        axs[i].set_title(f"t={ts[t_idx]:0.01f}")

        omega_grid = torch.linspace(0, np.pi, 1000)
        pdf_angle = diffusion.f_igso3(omega_grid, ts[t_idx]) * SO3Algebra.angle_density_unif(omega_grid)
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

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.joinpath("angle_distributions.png"))
        plt.close()
    else:
        plt.show()

def plot_process_comparison(
    forward_samples: Dict[float, SO3Config.R],
    reverse_samples: Dict[float, SO3Config.R],
    ts: np.ndarray,
    time_points: List[int],
    save_path: Optional[str] = None,
) -> None:
    """比较前向和反向过程在不同时间点的分布"""
    for i in time_points:
        t = ts[i]
        Rt_forward = SO3Algebra.Log(forward_samples[ts[i]])
        Rt_reverse = SO3Algebra.Log(reverse_samples[ts[i]])

        fig, axs = plt.subplots(1, 3, dpi=100, figsize=(9, 3))
        fig.suptitle(f"t={t:.1f}")
        bins = np.linspace(-np.pi, np.pi, 15)

        for Rt, label in [(Rt_forward, "forward"), (Rt_reverse, "reverse")]:
            axs[0].set_ylabel("density")
            for j in range(3):
                axs[j].hist(
                    Rt[:, j].numpy(),
                    bins,
                    density=True,
                    histtype="step",
                    label=label,
                )
                axs[j].set_xlabel(f"e_{j}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path.joinpath(f"process_comparison_t={t:.1f}.png"))
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # 初始化
    diffusion = IGSO3Diffusion(L=500, N_atoms=5, M=5000)

    # 设置时间参数
    T = 5.0
    ts = np.linspace(0, T, 200)
    N = 5000  # 采样数量

    # 前向过程
    forward_samples = diffusion.geodesic_random_walk(
        p_initial=lambda: diffusion.sample_p0(N),
        drift=lambda R, t: torch.zeros_like(R),  # 无漂移项
        ts=ts,
    )

    # 反向过程
    reverse_samples = diffusion.geodesic_random_walk(
        p_initial=lambda: SO3Config.random_R((N,)),
        drift=lambda R, t: -diffusion.compute_score(R, t),
        ts=ts[::-1],
    )

    # 可视化结果
    t_idcs_plot = [5, 25, 100, -1]
    plot_angle_distributions(
        forward_samples,
        reverse_samples,
        ts,
        t_idcs_plot,
        save_path="./results/",
    )

    time_points = [3, 5, 10, 25, 50, 100]
    plot_process_comparison(
        forward_samples,
        reverse_samples,
        ts,
        time_points,
        save_path="./results/",
    )
