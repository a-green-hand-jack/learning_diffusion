import torch
import numpy as np
from so3_diffusion import SO3Config, SO3Algebra
import pytest

class TestSO3:
    # @pytest.fixture
    def setup(self):
        """设置基本测试数据"""
        torch.manual_seed(42)
        # 生成随机旋转矩阵
        v = torch.randn(3)
        R = SO3Algebra.Exp(v)
        return {'v': v, 'R': R}

    def test_hat_vee_inverse(self, setup):
        """测试hat和vee映射是否互逆"""
        v = setup['v']
        X = SO3Algebra.hat(v)
        v_recovered = SO3Algebra.vee(X)
        assert torch.allclose(v, v_recovered, atol=1e-7)

    def test_exp_log_inverse(self, setup):
        """测试exp和log映射是否互逆"""
        R = setup['R']
        X = SO3Algebra.log(R)
        R_recovered = SO3Algebra.exp(X)
        assert torch.allclose(R, R_recovered, atol=1e-7)

    def test_Exp_Log_inverse(self, setup):
        """测试Exp和Log映射是否互逆"""
        v = setup['v']
        R = SO3Algebra.Exp(v)
        v_recovered = SO3Algebra.Log(R)
        assert torch.allclose(v, v_recovered, atol=1e-7)

    def test_rotation_properties(self, setup):
        """测试旋转矩阵的基本性质"""
        R = setup['R']
        
        # 检查行列式是否为1
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-7)
        
        # 检查正交性
        RTR = torch.matmul(R.T, R)
        I = torch.eye(3, dtype=R.dtype)
        assert torch.allclose(RTR, I, atol=1e-7)

    def test_small_angle_stability(self):
        """测试小角度情况下的数值稳定性"""
        # 生成接近零的旋转向量
        v_small = torch.tensor([1e-8, 1e-8, 1e-8])
        R_small = SO3Algebra.Exp(v_small)
        v_recovered = SO3Algebra.Log(R_small)
        assert torch.allclose(v_small, v_recovered, atol=1e-7)

    def test_riemannian_gradient(self, setup):
        """测试黎曼梯度计算"""
        R = setup['R']
        
        # 定义一个简单的测试函数：f(R) = ||R - I||^2
        def f(R):
            I = torch.eye(3, dtype=R.dtype, device=R.device)
            return torch.sum((R - I) ** 2)
        
        grad_R = SO3Algebra.riemannian_gradient(f, R)
        
        # 验证梯度是否属于切空间（反对称性）
        assert torch.allclose(grad_R + grad_R.transpose(-2, -1), 
                            torch.zeros_like(grad_R), atol=1e-7)

    def test_tangent_gaussian(self):
        """测试切空间高斯采样"""
        # 生成基准点
        R0 = torch.eye(3)
        
        # 采样多个点检验统计特性
        n_samples = 1000
        samples = torch.stack([SO3Algebra.tangent_gaussian(R0) 
                             for _ in range(n_samples)])
        
        # 验证样本均值是否接近0
        mean = torch.mean(samples, dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)
        
        # 验证是否都是反对称矩阵
        for X in samples:
            assert torch.allclose(X + X.T, torch.zeros_like(X), atol=1e-7)

    def test_batch_operations(self):
        """测试批量操作"""
        # 生成批量数据
        batch_size = 10
        v_batch = torch.randn(batch_size, 3)
        
        # 测试批量Exp映射
        R_batch = SO3Algebra.Exp(v_batch)
        print(R_batch.shape)
        assert R_batch.shape == (batch_size, 3, 3)
        
        # 测试批量Log映射
        v_recovered = SO3Algebra.Log(R_batch)
        print(v_recovered.shape)
        assert v_recovered.shape == (batch_size, 3)
        assert torch.allclose(v_batch, v_recovered, atol=1e-7)
        
        
    def test_angle_density(self, setup):
        """测试角度密度计算"""
        omega = SO3Algebra.Omega(setup['R'])
        density = SO3Algebra.angle_density_unif(omega)
        
        # 验证密度是否为非负值
        assert torch.all(density >= 0)
        
        # 对于0到π的角度，验证密度的单调性
        test_angles = torch.linspace(0, np.pi, 100)
        densities = SO3Algebra.angle_density_unif(test_angles)
        assert torch.all(torch.diff(densities) >= -1e-7)  # 应该是单调递增的
        
        


def run_manual_tests():
    print("开始运行SO3测试...")
    test_instance = TestSO3()
    
    # 设置测试数据
    setup_data = test_instance.setup()
    
    # 运行各个测试
    print("\n1. 测试hat和vee映射的互逆性")
    test_instance.test_hat_vee_inverse(setup_data)
    print("✓ 通过")
    
    print("\n2. 测试exp和log映射的互逆性")
    test_instance.test_exp_log_inverse(setup_data)
    print("✓ 通过")
    
    print("\n3. 测试Exp和Log映射的互逆性")
    test_instance.test_Exp_Log_inverse(setup_data)
    print("✓ 通过")
    
    print("\n4. 测试旋转矩阵的基本性质")
    test_instance.test_rotation_properties(setup_data)
    print("✓ 通过")
    
    print("\n5. 测试小角度情况下的数值稳定性")
    test_instance.test_small_angle_stability()
    print("✓ 通过")
    
    print("\n6. 测试黎曼梯度计算")
    test_instance.test_riemannian_gradient(setup_data)
    print("✓ 通过")
    
    print("\n7. 测试切空间高斯采样")
    test_instance.test_tangent_gaussian()
    print("✓ 通过")
    
    print("\n8. 测试批量操作")
    test_instance.test_batch_operations()
    print("✓ 通过")
    
    print("\n9. 测试角度密度计算")
    test_instance.test_angle_density(setup_data)
    print("✓ 通过")
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    run_manual_tests()