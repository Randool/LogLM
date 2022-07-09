# LogLM

Bert-like model with variable length of `[CLS]`

## SLURM提交

支持多机多卡训练，默认使用slurm提交任务。

```bash
sbatch ./dist_launch.sbatch
```

可调整脚本中的部分参数。

## 手动单机多卡

```bash
./local_launch.sh
```

可调整脚本中的部分参数。

## 手动多机多卡

假设要在机器A和机器B上运行多机多卡训练，假设A为主节点其IP为IPa，A和B分别有4块GPU。

1. 首先保证A和B上拥有相同的代码和数据。
2. 在A上执行`python tools/find_unused_port.py`获取未被使用的端口号`P`。
3. 修改A上的`local_launch.sh`：
  ```bash
   export MASTER_HOST=IPa
   export MASTER_PORT=P
   export SLURM_NNODES=2
   export SLURM_NODEID=0
   export nproc_per_node=4
   ```
4. 修改B上的`local_launch.sh`：
  ```bash
   export MASTER_HOST=IPa
   export MASTER_PORT=P
   export SLURM_NNODES=2
   export SLURM_NODEID=1
   export nproc_per_node=4
   ```
5. 在A和B上分别执行`./local_launch.sh`
