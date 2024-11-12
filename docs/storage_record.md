# 数据迁移和软链接操作记录

## 背景
由于项目空间限制，将大容量数据文件夹迁移至scratch目录下，并通过软链接保持原有代码结构。

## 目录信息
- 原始目录：`/projects/p32013/WJK/learning_diffusion/`
- Scratch目录：`/scratch/hlv8980/WJK/learning_diffusion/`

## 迁移的文件夹
以下文件夹因体积较大被迁移到scratch：
1. data
2. wandb
3. trained_models
4. results 

## 具体操作步骤

1. 创建scratch目标目录（如果不存在）：
```bash
mkdir -p /scratch/hlv8980/WJK/learning_diffusion/
mkdir -p /projects/p32013/WJK/learning_diffusion/data 
mkdir -p /projects/p32013/WJK/learning_diffusion/wandb 
mkdir -p /projects/p32013/WJK/learning_diffusion/trained_models 
mkdir -p /projects/p32013/WJK/learning_diffusion/results
```

2. 移动文件夹到scratch：
```bash
mv /projects/p32013/WJK/learning_diffusion/data /scratch/hlv8980/WJK/learning_diffusion/
mv /projects/p32013/WJK/learning_diffusion/wandb /scratch/hlv8980/WJK/learning_diffusion/
mv /projects/p32013/WJK/learning_diffusion/trained_models /scratch/hlv8980/WJK/learning_diffusion/
mv /projects/p32013/WJK/learning_diffusion/results /scratch/hlv8980/WJK/learning_diffusion/
```

3. 在原目录创建软链接：
```bash
cd /projects/p32013/WJK/learning_diffusion/
ln -s /scratch/hlv8980/WJK/learning_diffusion/data ./
ln -s /scratch/hlv8980/WJK/learning_diffusion/wandb ./
ln -s /scratch/hlv8980/WJK/learning_diffusion/trained_models ./
ln -s /scratch/hlv8980/WJK/learning_diffusion/results ./
```

## 验证方法
使用以下命令检查软链接是否正确创建：
```bash
ls -l
```

软链接正确创建后会显示类似如下输出：
```
lrwxrwxrwx 1 hlv8980 p32013 xx xx xx data -> /scratch/hlv8980/WJK/learning_diffusion/data
lrwxrwxrwx 1 hlv8980 p32013 xx xx xx wandb -> /scratch/hlv8980/WJK/learning_diffusion/wandb
lrwxrwxrwx 1 hlv8980 p32013 xx xx xx trained_models -> /scratch/hlv8980/WJK/learning_diffusion/trained_models
lrwxrwxrwx 1 hlv8980 p32013 xx xx xx results -> /scratch/hlv8980/WJK/learning_diffusion/results
```

## 注意事项
- 确保scratch目录有足够的存储空间
- 软链接创建后，原有代码无需修改路径
- 定期检查scratch空间使用情况
  - checkscratch X
