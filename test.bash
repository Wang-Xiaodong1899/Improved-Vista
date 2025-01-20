#!/bin/bash
#SBATCH --job-name=acnet      # 作业名称
#SBATCH --output=acnet_5z_output_test.log # 输出文件
#SBATCH --error=acnet_5z_error_test.log   # 错误文件
#SBATCH --partition=GPUA40               # 分区名称（根据实际情况修改）
#SBATCH --gres=gpu:4                 # 申请4张GPU
#SBATCH --ntasks=1                    # 任务数量
#SBATCH --cpus-per-task=32            # 每个任务分配32个CPU核心
#SBATCH --time=24:00:00               # 作业最大运行时间

source activate XXX
