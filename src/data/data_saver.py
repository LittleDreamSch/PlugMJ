"""
数据储存器
"""

import pandas as pd
from utils.log import Log


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
