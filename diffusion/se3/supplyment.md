# torch的一些补充

## torch.einsum


`torch.einsum` 是一个强大的函数，它实现了爱因斯坦求和约定（Einstein Summation Convention）。让我详细解释一下：

### 基本语法
`torch.einsum` 可以用一个简洁的字符串表达式来表示复杂的张量运算。其基本语法是：
```python
torch.einsum(equation, *operands)
```

### 常见模式举例

```python
# 1. 向量点积
a = torch.randn(3)
b = torch.randn(3)
c = torch.einsum('i,i->', a, b)
# 等价于: c = (a * b).sum()

# 2. 矩阵乘法
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.einsum('ij,jk->ik', a, b)
# 等价于: c = a @ b

# 3. 批量矩阵乘法
a = torch.randn(10, 2, 3)  # 批量大小为10
b = torch.randn(10, 3, 4)
c = torch.einsum('bij,bjk->bik', a, b)
# 等价于: c = torch.bmm(a, b)

# 4. 外积
a = torch.randn(3)
b = torch.randn(4)
c = torch.einsum('i,j->ij', a, b)
# 等价于: c = torch.outer(a, b)
```

### 在SO(3)代码中的应用

让我们看看代码中的几个实际例子：

```python
# 1. 在hat函数中：
def hat(v):
    """
    v: (..., 3)
    输出: (..., 3, 3)
    """
    return torch.einsum("...i,ijk->...jk", v, basis)
    # 这里的操作是：
    # - ...i 表示批量维度和3D向量的分量
    # - ijk 是预定义基底的三个维度
    # - ...jk 是输出的3x3矩阵
    # 实际上在计算：sum_i(v[...i] * basis[i,j,k])

# 2. 在expmap函数中：
def expmap(R0, tangent):
    """
    R0: (..., 3, 3)
    tangent: (..., 3, 3)
    """
    # 计算 R0^T * tangent
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    # 等价于：
    # skew_sym = torch.matmul(R0.transpose(-2, -1), tangent)
    
    # 计算 R0 * exp(skew_sym)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))
    # 等价于：
    # return torch.matmul(R0, exp(skew_sym))
```


### 命名规则
在einsum表达式中：
- 小写字母表示维度
- ... 表示任意数量的批量维度
- -> 后面的部分表示输出维度
- 重复的字母表示要在该维度上求和
- 不重复的字母表示要保留的维度
### 优势
- 简洁性：用一行代码表达复杂的张量运算
- 灵活性：可以处理任意维度的张量运算
- 可读性：通过表达式直观地表示运算过程
- 效率：PyTorch会自动优化运算

### 注意事项
- 虽然强大，但需要仔细检查维度匹配
- 对于简单操作（如普通矩阵乘法），使用标准操作符（如 @）可能更清晰
- 在处理大型张量时，要注意内存使用


## torch.linalg.matrix_exp

`torch.linalg.matrix_exp` 计算矩阵指数（matrix exponential），这是一个将矩阵映射到矩阵的函数。在SO(3)中，它的作用是将李代数so(3)中的元素（反对称矩阵）映射到李群SO(3)中的元素（旋转矩阵）。

### 数学定义

矩阵指数的定义是一个无穷级数：

```python
"""
exp(A) = I + A + (A²/2!) + (A³/3!) + ...

对于3x3反对称矩阵A，如果θ = ||A||（Frobenius范数），则有Rodrigues公式：
exp(A) = I + (sin θ/θ)A + ((1-cos θ)/θ²)A²
"""
```

### 在SO(3)中的具体应用

```python
import torch
import numpy as np

def hat(v):
    """将3D向量转换为反对称矩阵"""
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float)

def manual_matrix_exp(A):
    """手动实现矩阵指数（仅用于演示）"""
    # 计算θ（反对称矩阵的范数）
    theta = torch.sqrt(torch.sum(A**2) / 2)
    
    if theta < 1e-8:
        return torch.eye(3)
    
    # Rodrigues公式
    return (torch.eye(3) + 
            (torch.sin(theta)/theta) * A + 
            ((1-torch.cos(theta))/(theta**2)) * (A @ A))

# 示例：绕z轴旋转90度
angle = np.pi/2
v = torch.tensor([0., 0., angle])  # 旋转向量
A = hat(v)  # 转换为反对称矩阵

# 使用PyTorch的矩阵指数
R1 = torch.linalg.matrix_exp(A)
print("PyTorch matrix_exp结果:")
print(R1)

# 使用手动实现的矩阵指数
R2 = manual_matrix_exp(A)
print("\n手动实现结果:")
print(R2)
```

### 为什么需要矩阵指数？

1. **李群-李代数对应关系**：
```python
def exp_map(v):
    """从旋转向量到旋转矩阵的映射"""
    return torch.linalg.matrix_exp(hat(v))

def log_map(R):
    """从旋转矩阵到旋转向量的映射"""
    # 这是exp_map的逆操作
    return Rotation.from_matrix(R.numpy()).as_rotvec()

# 示例
v = torch.tensor([0., 0., np.pi/2])  # 旋转向量
R = exp_map(v)  # 转换为旋转矩阵
v_back = log_map(R)  # 转换回旋转向量
```

2. **插值和优化**：
```python
def interpolate_rotations(R1, R2, t):
    """在两个旋转之间插值"""
    # 转换到李代数
    v1 = log_map(R1)
    v2 = log_map(R2)
    
    # 在李代数中线性插值
    v_t = (1-t) * v1 + t * v2
    
    # 转换回李群
    return exp_map(torch.tensor(v_t))
```

3. **速度和加速度表示**：
```python
"""
在机器人学和计算机视觉中，角速度自然地表示为李代数中的元素
通过矩阵指数，我们可以将速度积分得到位置
"""
def integrate_velocity(omega, dt):
    """
    积分角速度得到旋转
    omega: 角速度向量
    dt: 时间步长
    """
    v = omega * dt  # 速度乘以时间得到增量旋转
    return exp_map(v)
```

### 实际应用示例

```python
import torch
import numpy as np

# 1. 基本旋转示例
def create_rotation(axis, angle):
    """创建绕指定轴的旋转"""
    v = axis * angle    # 旋转向量， vector = axis (unit vector) * angle
    A = hat(v)          # 反对称矩阵
    return torch.linalg.matrix_exp(A)

# 创建绕x、y、z轴的基本旋转
angle = np.pi/4  # 45度
Rx = create_rotation(torch.tensor([1., 0., 0.]), angle)
Ry = create_rotation(torch.tensor([0., 1., 0.]), angle)
Rz = create_rotation(torch.tensor([0., 0., 1.]), angle)

# 2. 连续旋转的插值
def slerp(R1, R2, t):
    """球面线性插值"""
    # 计算相对旋转
    R_rel = R2 @ R1.T   # 使用旋转矩阵计算相对旋转比较便利
    # 转换为旋转向量
    v_rel = torch.tensor(log_map(R_rel))    # 使用旋转向量在插值和优化中更方便
    # 缩放并应用
    return R1 @ exp_map(t * v_rel)

# 创建插值序列
t_values = torch.linspace(0, 1, 10)
interpolated_rotations = [slerp(Rx, Rz, t) for t in t_values]
```

矩阵指数在旋转表示中扮演着关键角色，它提供了：
1. 在李群和李代数之间转换的方法
2. 平滑插值的基础
3. 处理角速度和姿态变化的工具

这使得它在机器人学、计算机视觉和图形学等领域都非常重要。



# 旋转的表示方法

```python
from scipy.spatial.transform import Rotation
import numpy as np

# 创建一个示例旋转矩阵（绕z轴旋转90度）
R = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

rot = Rotation.from_matrix(R)

# 1. 旋转矩阵表示 (3x3)
matrix = rot.as_matrix()
print("旋转矩阵:\n", matrix)

# 2. 旋转向量表示 (3,)
rotvec = rot.as_rotvec()
print("旋转向量:", rotvec)  # 约等于 [0, 0, π/2]

# 3. 四元数表示 (4,)
quat = rot.as_quat()
print("四元数:", quat)

# 4. 欧拉角表示 (3,)
euler = rot.as_euler('xyz')
print("欧拉角:", euler)
```

## 关于Rotation.from_matrix()


- 从3x3旋转矩阵创建Rotation对象
- 输入：旋转矩阵，形状为(..., 3, 3)
- 输出：Rotation对象，形状为(..., ) 

## 关于Rotation.as_rotvec()

- 将Rotation对象转换为旋转向量
- 输入：Rotation对象，形状为(..., )
- 输出：旋转向量，形状为(..., 3)

### 为什么要使用旋转向量？

1. 紧凑性：旋转向量比旋转矩阵更紧凑，占用内存更少
2. 计算效率：在某些情况下，旋转向量计算更高效
3. 可解释性：旋转向量更直观，容易理解
4. 奇异性：旋转向量不存在奇异性问题，而旋转矩阵在某些情况下（如万向节锁）会出现奇异性

### 旋转向量的组成


旋转向量 v = θ * u
其中：
- u 是单位向量，表示旋转轴的方向
- θ 是标量，表示旋转角度（弧度）
- v 的方向就是旋转轴，v 的长度就是旋转角度

```python
import numpy as np
import torch

# 示例：创建一个旋转向量
theta = np.pi/2  # 90度旋转
axis = np.array([0, 0, 1])  # 绕z轴旋转
rotation_vector = theta * axis  # 旋转向量

print(f"旋转向量: {rotation_vector}")  # 输出: [0, 0, π/2]
```
## Omega

在SO(3)中，`Omega(R)` 函数计算的是旋转矩阵R对应的旋转角度。让我解释一下这个概念：

### Omega函数的定义

```python
def Omega(R):
    """
    计算旋转矩阵R对应的旋转角度
    
    输入: R - SO(3)中的旋转矩阵
    输出: θ ∈ [0, π] - 旋转角度（以弧度为单位）
    
    数学公式: θ = arccos((tr(R) - 1)/2)
    """
    return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1) - 1) / 2)
```

### 为什么是R^+（非负实数）

1. **数学原因**：
```python
"""
对于任意旋转矩阵R：
1. tr(R) ∈ [-1, 3]  # 旋转矩阵的迹的范围
2. (tr(R) - 1)/2 ∈ [-1, 1]  # arccos的输入必须在[-1,1]范围内
3. arccos: [-1, 1] → [0, π]  # arccos的值域是[0,π]
"""

# 示例
def check_angle_range():
    # 创建一些旋转矩阵
    angles = torch.linspace(0, 2*np.pi, 100)
    for angle in angles:
        R = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        omega = Omega(R)
        assert 0 <= omega <= np.pi  # 角度总是在[0,π]范围内
```

2. **几何解释**：
```python
"""
旋转角度θ的范围是[0, π]的原因：
1. 任何旋转都可以用不超过180度的旋转表示
2. 顺时针转200度 = 逆时针转160度
3. 这避免了表示的歧义性
"""

# 示例：两种等价的旋转
def equivalent_rotations():
    axis = torch.tensor([0., 0., 1.])  # z轴
    
    # 200度顺时针旋转
    theta1 = 200 * np.pi / 180
    R1 = exp_map(theta1 * axis)
    
    # 160度逆时针旋转
    theta2 = -160 * np.pi / 180
    R2 = exp_map(theta2 * axis)
    
    # 它们的Omega值相同
    print(f"R1的角度: {Omega(R1)}")  # 约等于2.793 (160°)
    print(f"R2的角度: {Omega(R2)}")  # 约等于2.793 (160°)
```

3. **实际应用**：
```python
def analyze_rotation(R):
    """分析旋转矩阵的属性"""
    theta = Omega(R)
    
    # 从旋转矩阵恢复旋转轴和角度
    rotvec = Rotation.from_matrix(R.numpy()).as_rotvec()
    axis = rotvec / np.linalg.norm(rotvec) if np.linalg.norm(rotvec) > 0 else np.zeros(3)
    
    print(f"旋转角度: {theta.item() * 180/np.pi}度")
    print(f"旋转轴: {axis}")
    
    if theta < 1e-6:
        print("这是一个恒等旋转")
    elif abs(theta - np.pi) < 1e-6:
        print("这是一个180度旋转")
    else:
        print("这是一个一般的旋转")

# 示例使用
R1 = torch.eye(3)  # 恒等旋转
R2 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # 180度旋转
R3 = exp_map(torch.tensor([0., 0., np.pi/4]))  # 45度旋转

for R in [R1, R2, R3]:
    analyze_rotation(R)
    print()
```

4. **与其他表示的关系**：
```python
def rotation_representations(theta):
    """展示不同旋转表示之间的关系"""
    # 假设绕z轴旋转
    axis = np.array([0, 0, 1])
    
    # 旋转向量
    rotvec = theta * axis
    
    # 旋转矩阵
    R = Rotation.from_rotvec(rotvec).as_matrix()
    
    # 计算Omega
    omega = Omega(torch.tensor(R))
    
    print(f"输入角度: {theta * 180/np.pi}度")
    print(f"Omega(R): {omega.item() * 180/np.pi}度")
    print(f"旋转向量范数: {np.linalg.norm(rotvec) * 180/np.pi}度")

# 测试不同角度
for theta in [np.pi/4, np.pi, 3*np.pi/2]:
    rotation_representations(theta)
    print()
```

总结：
1. Omega(R)总是返回[0, π]范围内的值
2. 这种表示避免了旋转表示的歧义性
3. 任何超过180度的旋转都可以用更小的角度表示
4. 这种表示在旋转插值和优化中特别有用

这就是为什么Omega函数的值域是R^+（更准确地说是[0, π]）的原因。这种设计使得旋转表示更加规范和无歧义。