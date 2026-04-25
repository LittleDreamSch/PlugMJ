"""
数据储存器
"""

import json
import os

import pandas as pd
from plugmj.utils.log import Log


class DataSaver:
    def __init__(self, path, logger: Log) -> None:
        """
        初始化

        Args:
            path(str): 储存路径
            logger(Log): 日志器
        """
        self.path = path
        self.data = []
        self.logger = logger

    def save(self):
        """
        保存结果到 csv
        """
        df = pd.DataFrame(self.data)
        df.to_csv(self.path, index=False, header=False)
        self.logger.success(f"Data saved to {self.path}")

    def append(self, new_data):
        """
        添加结果

        Args:
            new_data(tuple): 新结果，结果格式为 ((g_val, prim_sol, dual_sol), (x1, x2, x3, ...))
        """
        self.data.append(new_data)

    def save_stats(self, total_time, mem_samples):
        """
        保存运行统计到 JSON 文件（与 CSV 同目录）。

        Args:
            total_time (float): 总优化时间（秒）
            mem_samples (list[float]): 每步采样的内存使用（MB）
        """
        stats_path = os.path.splitext(self.path)[0] + "_stats.json"
        stats = {
            "optimization_time": round(total_time, 4),
            "peak_memory_mb": round(max(mem_samples), 1) if mem_samples else 0,
            "avg_memory_mb": round(sum(mem_samples) / len(mem_samples), 1) if mem_samples else 0,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        self.logger.success(f"Stats saved to {stats_path}")
