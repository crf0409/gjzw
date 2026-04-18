#!/bin/bash

# --- 配置 ---
# 定义你想要使用的GPU ID列表。
# 如果你有4张卡，就是 0 1 2 3
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# --- 主逻辑 ---
echo "Starting parallel benchmark tests on $NUM_GPUS GPUs..."

# 查找所有以 _benchmark.py 结尾的文件
scripts=(*_benchmark.py)
num_scripts=${#scripts[@]}
echo "Found $num_scripts scripts to run."

# 用于跟踪当前正在运行的后台任务数量
job_count=0

# 遍历所有脚本
for script in "${scripts[@]}"
do
  # 从脚本文件名中提取基础名称
  base_name=$(basename -s _benchmark.py "$script")
  
  # 使用模运算来轮流选择GPU ID
  # 这样可以确保任务均匀分配到所有GPU上
  gpu_id=${GPUS[$((job_count % NUM_GPUS))]}
  
  # 定义日志文件名 (建议加入GPU ID以作区分)
  log_file="${base_name}_gpu${gpu_id}.log"
  
  echo "--------------------------------------------------"
  echo "Dispatching: $script to run on GPU $gpu_id"
  echo "Saving output to: $log_file"
  
  # 在指定的GPU上以后台模式(&)运行Python脚本
  CUDA_VISIBLE_DEVICES=$gpu_id python "$script" > "$log_file" 2>&1 &
  
  # 任务计数器加一
  job_count=$((job_count + 1))
  
  # 如果正在运行的任务数达到了GPU的数量，就等待任意一个任务完成
  # 这样可以腾出一个“槽位”给下一个任务
  if [ $job_count -ge $NUM_GPUS ]; then
    wait -n
  fi
done

# 等待所有剩余的后台任务全部完成
echo "--------------------------------------------------"
echo "All scripts have been dispatched. Waiting for the last running jobs to complete..."
wait

echo "--------------------------------------------------"
echo "All benchmark tests are complete."