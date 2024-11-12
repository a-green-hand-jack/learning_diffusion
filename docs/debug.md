# IPDB

在quest上使用`debugpy`很麻烦，所以就不用了。

使用ipdb的设置很简单。首先确保安装了ipdb：

```bash
pip install ipdb
```

然后在您的model文件中添加断点：

```python
# 在文件开头导入
import ipdb

class YourModel(nn.Module):
    def forward(self, x):
        ipdb.set_trace()  # 这里会触发断点
        # 您的模型代码
```

ipdb的常用命令：
- `n`: next，执行下一行
- `s`: step，步入函数
- `c`: continue，继续执行
- `p 变量名`: 打印变量值
- `p dir(对象)`: 查看对象的所有属性和方法
- `ll`: 显示更多代码上下文
- `w`: where，显示当前的调用栈
- `q`: quit，退出调试器
- `h`: help，显示帮助信息

比如在模型中的具体应用：
```python
class MyNet(nn.Module):
    def forward(self, x):
        # 在关键处添加断点
        print(f"Shape before debug: {x.shape}")
        ipdb.set_trace()  # 程序会在这里暂停
        
        # 或者添加条件断点
        if torch.isnan(x).any():
            ipdb.set_trace()
            
        return x
```

ipdb相比pdb的优势：
- 语法高亮
- 更好的代码补全
- 更清晰的变量展示
- 支持post-mortem调试

需要我提供更具体的调试示例吗？