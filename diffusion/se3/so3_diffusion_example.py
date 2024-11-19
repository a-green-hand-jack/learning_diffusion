# -*- coding: utf-8 -*-
# pyright: reportInvalidTypeForm=false
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUnusedImport=false
# ruff: noqa: F401, E741, E402

"""
SO3_diffusion_example.ipynb

自动生成于 Colab。

原始文件位于
    https://colab.research.google.com/drive/1J3Yq2AbqjFYPksdXxo8BGAbWBdBxuD2o

# SO(3) 扩散建模和 IGSO(3) 分布
该笔记本的目的是提供一个关于 SO(3) 流形上的基于分数的生成建模方法的最小示例。

关键特性包括：
* 展示了 IGSO(3) 密度与测地线随机游走之间的一致性，以及它们对 SO(3) 上均匀分布的收敛。
* 展示了针对 SO(3) 上离散测度的前向和反向过程之间的一致性，对于这些测度，Stein 分数可以通过自动微分（而不是通过分数匹配）直接计算。
* 实现了在 SO(3) 上的各向同性高斯（IGSO3），其方差参数被重新调整以符合 Lie 代数上的布朗运动，该运动是针对 SO(3) 的规范内积定义的。

此笔记本由 Brian L. Trippe, Valentin de Bortoli, Jason Yim 和 Emile Mathieu 编写。
"""

import torch

torch.set_default_dtype(torch.float64)  # 设置默认的数据类型为 double
import numpy as np

np.random.seed(42)  # 设置随机种子
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import typing
from typing import Callable, Dict, List, Tuple
# 定义 SO(3) 流形上的原始操作

# SO(3) 的正交基，形状为 [3, 3, 3]
basis: torch.Tensor = torch.tensor(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


# 从向量空间 R^3 到 Lie 代数 so(3) 的 hat 映射
def hat(v: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,ijk->...jk", v, basis)


# 从 SO(3) 到 R^3 的对数映射（即旋转向量）
def Log(R: torch.Tensor) -> torch.Tensor:
    return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())


# 从 SO(3) 到 so(3) 的对数映射，这是矩阵对数
def log(R: torch.Tensor) -> torch.Tensor:
    return hat(Log(R))


# 从 so(3) 到 SO(3) 的指数映射，这是矩阵指数
def exp(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(A)


# 从 R0 处的切空间到 SO(3) 的指数映射
def expmap(R0: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))


# 返回旋转角度。SO(3) 到 R^+
def Omega(R: torch.Tensor) -> torch.Tensor:
    return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1) - 1) / 2)


# 旋转角度的边缘密度，对于 SO(3) 上的均匀密度
def angle_density_unif(omega: torch.Tensor) -> torch.Tensor:
    return (1 - torch.cos(omega)) / np.pi


# 在 R0 处的切空间中进行正态采样
def tangent_gaussian(R0: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...jk->...ik", R0, hat(torch.randn(R0.shape[0], 3)))


def riemannian_gradient(
    f: Callable[[torch.Tensor], torch.Tensor], R: torch.Tensor
) -> torch.Tensor:
    coefficients: torch.Tensor = torch.zeros(
        list(R.shape[:-2]) + [3], requires_grad=True
    )
    R_delta: torch.Tensor = expmap(
        R, torch.einsum("...ij,...jk->...ik", R, hat(coefficients))
    )
    grad_coefficients: torch.Tensor = torch.autograd.grad(
        f(R_delta).sum(), coefficients
    )[0]
    return torch.einsum("...ij,...jk->...ik", R, hat(grad_coefficients))





## 定义 IGSO3 密度，测地线随机游走，并检查它们的一致性。


# IGSO3 密度的幂级数展开。
def f_igso3(omega: torch.Tensor, t: float, L: int = 500) -> torch.Tensor:
    ls: torch.Tensor = torch.arange(L)[None]  # 形状为 [1, L]
    return (
        (2 * ls + 1)
        * torch.exp(-ls * (ls + 1) * t / 2)
        * torch.sin(omega[:, None] * (ls + 1 / 2))
        / torch.sin(omega[:, None] / 2)
    ).sum(dim=-1)


# IGSO3(Rt; I_3, t)，相对于 SO(3) 上的体积形式的密度
def igso3_density(Rt: torch.Tensor, t: float, L: int = 500) -> torch.Tensor:
    return f_igso3(Omega(Rt), t, L)


# 模拟前向和反向过程的程序
def geodesic_random_walk(
    p_initial: typing.Callable[[], torch.Tensor],
    drift: typing.Callable[[torch.Tensor, float], torch.Tensor],
    ts: np.ndarray,
) -> typing.Dict[float, torch.Tensor]:
    Rts: typing.Dict[float, torch.Tensor] = {ts[0]: p_initial()}
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]  # 反向过程为负
        Rts[ts[i]] = expmap(
            Rts[ts[i - 1]],
            drift(Rts[ts[i - 1]], ts[i - 1]) * dt
            + tangent_gaussian(Rts[ts[i - 1]]) * np.sqrt(abs(dt)),
        )
    return Rts


def score_t(Rt: torch.Tensor, t: float) -> torch.Tensor:
    return riemannian_gradient(lambda R: torch.log(p_t(R, t)), Rt)


# 从 U(SO(3)) 采样 N 次，通过插值均匀分布的累积分布函数
def p_inv(N: int, M: int = 1000) -> torch.Tensor:
    omega_grid: torch.Tensor = torch.linspace(0, np.pi, M)
    cdf: np.ndarray = np.cumsum(angle_density_unif(omega_grid).numpy()) / (M / np.pi)
    omegas: np.ndarray = np.interp(np.random.rand(N), cdf, omega_grid)
    axes: np.ndarray = np.random.randn(N, 3)
    axes = omegas[:, None] * axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    return exp(hat(torch.tensor(axes)))


# 定义 SO(3) 上的离散目标测度，以及其在 t>0 时的分数
N_atoms: int = 3
mu_ks: torch.Tensor = p_inv(N_atoms)  # 定义目标测度的原子


# 采样 p_0 ~ (1/N_atoms)\sum_k Dirac_{mu_k}
def p_0(N: int) -> torch.Tensor:
    return mu_ks[torch.randint(mu_ks.shape[0], size=[N])]


# 密度的离散目标噪声在时间 t
def p_t(Rt: torch.Tensor, t: float) -> torch.Tensor:
    return (
        sum(
            [
                igso3_density(torch.einsum("ji,...jk->...ik", mu_k, Rt), t)
                for mu_k in mu_ks
            ]
        )
        / N_atoms
    )


### 模拟测地线随机游走
N: int = 5000  # 样本数量
T: float = 5.0  # 最终时间
ts: np.ndarray = np.linspace(0, T, 200)  # [0, T] 的离散化

random_walk: typing.Dict[float, torch.Tensor] = geodesic_random_walk(
    p_initial=lambda: exp(torch.zeros(N, 3, 3)), drift=lambda Rt, t: 0.0, ts=ts
)

t_idcs_plot: typing.List[int] = [5, 25, 100, -1]
fig, axs = plt.subplots(1, len(t_idcs_plot), dpi=100, figsize=(3 * len(t_idcs_plot), 3))
for i, t_idx in enumerate(t_idcs_plot):
    # 从测地线随机游走中绘制旋转角度的经验分布
    bins: np.ndarray = np.linspace(0, np.pi, 20)
    axs[i].hist(
        Omega(random_walk[ts[t_idx]]),
        bins=bins,
        density=True,
        histtype="step",
        label="Empirical",
    )
    axs[i].set_title(f"t={ts[t_idx]:0.01f}")

    # 计算旋转角度的密度，以及均匀分布的密度
    omega_grid: torch.Tensor = torch.linspace(0, np.pi, 1000)
    pdf_angle: torch.Tensor = f_igso3(omega_grid, ts[t_idx]) * angle_density_unif(
        omega_grid
    )
    axs[i].plot(omega_grid, pdf_angle.numpy(), label="IGSO3(*;I, t)")
    axs[i].plot(omega_grid, angle_density_unif(omega_grid).numpy(), label="Uniform")

    axs[i].set_xlabel("Angle of rotation (radians)")
axs[0].legend()
axs[0].set_ylabel("p(angle)")

plt.suptitle("Agreement of IGSO3 density and geodesic random walk", y=1.05)
plt.tight_layout()
plt.savefig("./results_gt/igso3_density_and_geodesic_random_walk.png")
plt.show()



# 模拟前向和反向过程
forward_samples: Dict[float, torch.Tensor] = geodesic_random_walk(
    p_initial=lambda: p_0(N), drift=lambda Rt, t: 0.0, ts=ts
)
reverse_samples: Dict[float, torch.Tensor] = geodesic_random_walk(
    p_initial=lambda: p_inv(N), drift=lambda Rt, t: -score_t(Rt, t), ts=ts[::-1]
)


# 可视化前向和反向过程在几个时间步的边际分布的一致性
for i in [3, 5, 10, 25, 50, 100]:
    t: float = ts[i]
    Rt_forward: torch.Tensor = Log(forward_samples[ts[i]])
    Rt_reverse: torch.Tensor = Log(reverse_samples[ts[i]])
    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(9, 3))
    fig.suptitle(f"t={t:.1f}")
    bins: np.ndarray = np.linspace(-np.pi, np.pi, 15)
    for Rt, label in [(Rt_forward, "forward"), (Rt_reverse, "reverse")]:
        for j in range(3):
            axs[j].hist(
                Rt[:, j].numpy(), bins, density=True, histtype="step", label=label
            )
            axs[j].set_xlabel(f"e{j}")
            axs[j].set_ylabel("density")
    plt.legend()
    plt.savefig(f"./results_gt/forward_reverse_distribution_t={t:.1f}.png")
    # plt.show()
