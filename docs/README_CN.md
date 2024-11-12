# 启动

包括建立本地的个人文件夹，加载个人 git 设置，连接到自己的 GitHub 账号。

## 建立与 GitHub 的 ssh 连接

### 密钥

首先生成密钥对，值得注意的是，生成的密钥应该位于 `WJK` 文件夹中的 `.ssh_jieke`子文件夹中，使用下面的命令：

```cmd
# 保证再 WJK 这个文件夹下
ssh-keygen -t rsa -C "a-green-hand-jack@users.noreply.github.com" -f ".ssh_jieke/id_rsa_jieke"
```

这样就在`.ssh_jieke`文件夹下得到了`id_rsa_jieke  id_rsa_jieke.pub`，随后查看公钥，使用`cat id_rsa_jieke.pub`。随后复制公钥到自己的 [GitHub 设置](https://github.com/settings/keys) 中，然后添加公钥。

### 验证 agent 

```cmd
eval $(ssh-agent -s)
```

### 添加密钥

```cmd
ssh-add .ssh_jieke/id_rsa_jieke
# Identity added: /home/hlv8980/.ssh_jieke/id_rsa_jieke (/home/hlv8980/.ssh_jieke/id_rsa_jieke)
```

> 这个很关键，如果发现不能和自己的 repo 连接了就先**验证 agent**然后再**添加密钥**

### 克隆

```cmd
eval $(ssh-agent -s)
ssh-add .ssh_jieke/id_rsa_jieke 
git clone ssh://git@ssh.github.com:443/a-green-hand-jack/geneLLM.git
```

### 上传

```cmd
eval $(ssh-agent -s)
ssh-add .ssh_jieke/id_rsa_jieke 
git push
```

# 环境

## pip

```cmd
conda env create -f ./env/SE3nv.yml
pip install --force-reinstall -r ./env/requirements.txt
```

```cmd
conda env export > ./env/SE3nv.yml
pip freeze > ./env/requirements.txt
```

### 拉取

```cmd
git fetch origin
git branch -r
git branch --set-upstream-to=origin/jack
git pull
```

综合上面的过程，我们就实现了与 GitHub 的 ssh 连接的建立；同时，也不会影响到别人。


# 作业
我的工作路径：`../../projects/p32013/WJK/Point_cloud_head_GFM/`

## 确定资源

## 申请机器

```cmd
salloc --account=p32013 -p normal --cpus-per-task 4 --mem 10G -t 02:00:00 
salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 24 --mem 100G --constraint=sxm -t 48:00:00 
salloc --account=p32013 -p gengpu-long --gres gpu:a100:1 --cpus-per-task 24 --mem 100G -t 240:00:00 
salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 24 --mem 100G -t 240:00:00 
```

### 日志

> 免得自己找不到设备

```cmd
# 2024-10-19-19:19
(CFD) [hlv8980@quser33 Point_cloud_head_GFM]$ salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 24 --mem 100G --constraint=sxm -t 24:00:00
salloc: Pending job allocation 4191479
salloc: job 4191479 queued and waiting for resources
salloc: job 4191479 has been allocated resources
salloc: Granted job allocation 4191479
salloc: Waiting for resource configuration
salloc: Nodes qgpu2015 are ready for job
```


## 查看资源

```cmd
sinfo
```
## 运行程序

```cmd
squeue -u hlv8980
```

squeue -u hlv8980
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           3845773    gengpu interact  hlv8980  R    1:10:46      1 qgpu2007
## 结束资源

scancel 3845773

# Google Drive

尝试连接 Google Drive 和服务器，以便快速处理大文件。