"""
cvxpy 接口
"""

import cvxpy as cp
import numpy as np
from interface.interface import Interface
from data.task_loader import TaskLoader
from data.data_saver import DataSaver
from utils.log import Log


class CvxpyInterface(Interface):
    def __init__(
        self, task_loader: TaskLoader, log: Log, data_saver: DataSaver, **MOSEK_OPTIONS
    ):
        """
        初始化 cvxpy 接口

        Args:
            task_loader (TaskLoader): 任务加载器
            logger (Log): 日志器
            data_saver (DataSaver): 数据保存器
            **MOSEK_OPTIONS: MOSEK 参数
        """
        super().__init__(task_loader, log, data_saver)

    # TODO: 多线程
    # 参考 https://www.cvxpy.org/tutorial/intro/index.html#changing-the-problem 末尾

    # TODO: get_problem_data 获取 mosek 标准问题
    # 参考 https://www.cvxpy.org/tutorial/advanced/index.html
