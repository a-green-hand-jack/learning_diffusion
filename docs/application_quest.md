# Git & GitHub

## check
```cmd
git branch
```
## push

```cmd
eval $(ssh-agent -s)
ssh-add .ssh_jieke/id_rsa_jieke 
git push
```
## pull

```cmd
# 确保你在jack分支上
git checkout jack

# 推送到远程仓库
git push origin jack
```

## clone

```cmd
eval $(ssh-agent -s)
ssh-add .ssh_jieke/id_rsa_jieke 
git clone git@github.com:a-green-hand-jack/learning_diffusion.git
```

## check

### check jobs

```cmd
squeue -u $user_name  # default: hlv8980
```

### check resource

```cmd
sinfo
```

### kill jobs

```cmd
scancel $job_name   # sample: 3845773
```
# record

## 2024-11-12
```cmd
salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 12 --mem 50G --constraint=sxm -t 24:00:00
salloc: Pending job allocation 6510374
salloc: job 6510374 queued and waiting for resources
salloc: job 6510374 has been allocated resources
salloc: Granted job allocation 6510374
salloc: Nodes qgpu2013 are ready for job
```

## 2024-11-12
```cmd
salloc --account=p32013 -p normal --cpus-per-task 4 --mem 10G -t 24:00:00
salloc: Pending job allocation 6511792                                                                                   salloc: job 6511792 queued and waiting for resources                                                                                salloc: job 6511792 has been allocated resources                                                                                 salloc: Granted job allocation 6511792                                                                                   salloc: Nodes qnode9027 are ready for job  
```

## 2024-11-12-15：29
```cmd
salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 12 --mem 50G --constraint=sxm -t 24:00:00
salloc: Pending job allocation 6801856
salloc: job 6801856 queued and waiting for resources
salloc: job 6801856 has been allocated resources
salloc: Granted job allocation 6801856
salloc: Nodes qgpu2012 are ready for job
```

## 2024-11-15 17：00

```cmd
salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 12 --mem 50G --constraint=sxm -t 24:00:00
```