#!/bin/sh
#####################################################
## PlugMJ Slurm 模板 
##                                  2024/10/23 @Dream
#####################################################
## 本模板用于提交 Slurm 的 PlugMJ 任务
#####################################################
## 在用户根目录下创建 result 目录，并在 result 目录下
## 创建任务名的目录，用于存放 JSON 文件、结果和日志
## 
## 例如名为 2DP2601 的项目，其目录结构:
## result
## └── 2DP2601
##     ├── Task_2DP2601.json  -> PlugMJ JSON 文件
##     ├── Result_2DP2601.csv -> 求解结果
##     ├── 2DP2601_slurm.sh   -> 本脚本
##     ├── 2DP2601_log.txt    -> 日志
##     ├── 2DP2601_error.txt  -> 错误日志
##     └── 2DP2601_mem.log    -> 内存监控日志
#####################################################
## 根据需求修改带 * 号的参数:
## =====================
##     Slurm Setting      
## =====================
# *Slurm_Task_Name 任务名
#SBATCH --job-name=SDP_TASK
# *申请的队列名
#SBATCH --partition=d2_hpc
# *每个核申请的内存
#SBATCH --mem-per-cpu=32gb
# *任务数
#SBATCH --ntasks=1
# *每个任务申请的核数
#SBATCH --cpus-per-task=5
# *节点数量
#SBATCH --nodes=1
# log
# 日志保存为 CWD/{Slurm_Task_Name}_[log, error].txt
#SBATCH --output=%x_log.txt
#SBATCH --error=%x_error.txt
## =====================
##        Setting     
## =====================
# *工作目录: 将执行 ~/result/{Slurm_Task_Name}.json 的任务
CWD="$HOME/result/${SLURM_JOB_NAME}"
# *内存监控的更新时间 (秒)
MONITOR_INTERVAL=60
# 内存监控日志文件名
MONITOR_LOG="${SLURM_JOB_NAME}_mem.log"

# ================ #
#      Monitor     #
# ================ #
cd $CWD
# 如果已经存在了则删除
if [ -f "$MONITOR_LOG" ]; then
  rm -f "$MONITOR_LOG"
fi
touch "$MONITOR_LOG"
  
nameflag=`scontrol show hostname $SLURM_JOB_NODELIST`
jobflag=`squeue | grep $SLURM_JOB_ID | wc -l`
Monitor() {
  while true; do
    if [ -n "$pid" ]; then
      usage=$(ps -p $pid -o rss=)
      usage=$(echo "scale=3; $usage / (1024^2)" | bc)
      timestamp=$(date +%Y%m%d-%T)
      echo "$timestamp, $usage" >> $MONITOR_LOG
    else
      break
    fi
    sleep $MONITOR_INTERVAL
  done
}

# =============== # 
#       RUN       # 
# =============== # 
# *执行求解
PlugMJ -t "Task_${SLURM_JOB_NAME}.json" -o "Result_${SLURM_JOB_NAME}.csv" &

# 获取 pid
pid=$!

# 启动 Monitor
Monitor &
monitorPID=$!

wait $pid
kill $monitorPID
