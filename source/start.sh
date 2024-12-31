#!/bin/sh
# ================ #
#      Setting     #
# ================ #
# Task 文件名
PLUGMJ_TASK_NAME="Task.json"
# 输出文件名
PLUGMJ_EXPORT_NAME="Result.csv"
# 调用线程数
PLUGMJ_THREAD_NUM=0
# 任务名
PLUGMJ_JOB_NAME="name"

# *内存监控的更新时间 (秒)
MONITOR_INTERVAL=10
# 内存监控日志文件名
MONITOR_LOG="${PLUGMJ_JOB_NAME}_mem.log"

# ================ #
#      Monitor     #
# ================ #
# 如果已经存在了则删除
if [ -f "$MONITOR_LOG" ]; then
  rm -f "$MONITOR_LOG"
fi
touch "$MONITOR_LOG"
  
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
# ! 修改此行以运行命令
PlugMJ -t "$PLUGMJ_TASK_NAME" -o "$PLUGMJ_EXPORT_NAME" -T "$PLUGMJ_THREAD_NUM" -n "$PLUGMJ_JOB_NAME" &

# 获取 pid
pid=$!

# 启动 Monitor
Monitor &
monitorPID=$!

wait $pid
kill $monitorPID
