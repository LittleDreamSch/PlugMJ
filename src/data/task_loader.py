"""
加载并格式化给定的 Json 文件
"""

import numpy as np
import json
import time
import os
from scipy.sparse import coo_matrix as coo


class TaskLoader:
    """
    加载并格式化给定的 Json 文件
    """

    def __init__(self, task_json_file, logger, name="SDP_TASK"):
        """
        初始化

        Args:
            task_json_file: 给定的 Json 文件路径
            logger: 日志记录器
            name: 任务名
        """
        # 判断 task_json_file 是否存在
        if not os.path.exists(task_json_file):
            logger.critical(f"Task file {task_json_file} not found.")
            exit(1)
        else:
            logger.info(f"Load task from {task_json_file}")
            with open(task_json_file, "r") as f:
                self.task_json = json.load(f)

            self.logger = logger
            self.name = name
            self.load_task()
            self.log_info()

    def load_task(self):
        """
        解析加载的 Json 数据
        """
        start_time = time.time()
        # 任务名
        # self.name = self.task_json["taskname"]
        # 变量个数
        self.n_var = self.task_json["variable_length"]
        # 参数取值
        self.g_vals = (
            [0] if "para_value" not in self.task_json else self.task_json["para_value"]
        )
        # 精度
        self.eps = 1e-6 if "eps" not in self.task_json else self.task_json["eps"]
        # PSD 约束个数
        self.n_psd = self.task_json["constrains_length"]
        # PSD 约束维度
        self.d_psd = self.task_json["constrains_dim"]
        # 目标函数
        self.target = self.task_json["target"]
        # PSD 约束
        self.psd = self.load_psd(self.d_psd, self.task_json["constrains"])
        # 线性约束
        self.lc = self.task_json["eqConstrains"]
        # 线性约束数量
        self.n_lc = len(self.task_json["eqConstrains"])

        # 打印加载时间
        end_time = time.time()
        self.logger.success(f"Loading task time: {end_time - start_time:.4f}s")

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, raw_target):
        """
        加载目标函数

        Args:
            raw_target: 原始目标函数数据，[[c_index_1, c_value_1], [c_index_2, c_value_2],...]

        Returns:
            c_index: 目标函数变量索引
            c_value: 目标函数系数
        """
        self.logger.info("Loading target function...")
        c_index = [c[0] for c in raw_target]
        c_value = [c[1] for c in raw_target]
        self._target = c_index, c_value

    def load_psd(self, dims, raw_psd):
        """
        加载 PSD 矩阵

        Args:
            dims: 矩阵维度
            raw_psd: 原始 PSD 矩阵数据，每个矩阵的格式为
                     {
                        "Index1": [[ROW1], [COL1], [VAL1]],
                        "Index2": [[ROW2], [COL2], [VAL2]],
                        ...
                        "Cons"  : [[ROWC], [COLC], [VALC]]
                     }
                     每一项描述了一个 PSD 矩阵的上三角部分，其中 Index 为矩阵的索引，ROW/COL 为非零元素的行/列索引，VAL 为非零元素的值，Cons 为常数项

        Returns:
            psd: [
                     [cons_coo_1, [F_coo_1], [Index_1]],
                     [cons_coo_2, [F_coo_2], [Index_2]],
                     ...
                     [cons_coo_m, [F_coo_m], [Index_m]]
                 ]
                 每一项表示一个 PSD 矩阵的上三角部分，cons_coo 为常数项的 coo 格式表示，F_coo 为矩阵的 coo 格式表示，Index 为矩阵的索引，
                 psd 矩阵可以被还原为 cons_coo_1 + F_coo_1 @ Index_1
        """
        self.logger.info("Loading PSD ..")

        psds = []
        for id_psd, each_psd in enumerate(raw_psd):
            F_coo = []
            index = []
            dim = dims[id_psd]
            cons_coo = coo((dim, dim))
            for key in each_psd:
                if key == "Cons":
                    cons_coo = coo(
                        (each_psd[key][2], (each_psd[key][0], each_psd[key][1])),
                        shape=(dim, dim),
                    )
                else:
                    index.append(int(key))
                    F_coo.append(
                        coo(
                            (each_psd[key][2], (each_psd[key][0], each_psd[key][1])),
                            shape=(dim, dim),
                        )
                    )
            psds.append([cons_coo, F_coo.copy(), index.copy()])

        return psds

    @property
    def lc(self):
        """获得线性约束"""
        return self._lc

    @lc.setter
    def lc(self, raw_lc):
        """
        加载线性约束

        等式约束的格式为 [cons, [i1, j1], ..., [in, jn]]，表示

        (i1[0] + i1[1] * g) * x[j1] + ... == -(cons[0] + cons[1] * g)

        self.lc(tuple):
            A (coo, n x n_var): 线性约束矩阵常数项
            Ag(coo, n x n_var): 线性约束矩阵变量项
            b (np.array, n x 1) : 线性约束常数项
            bg(np.array, n x 1) : 线性约束变量项

        线性约束可以被表示为 (A + g * Ag) @ x == b + g * bg

        Args:
            raw_lc: 原始线性约束数据

        """
        self.logger.info("Loading linear constraints...")
        A_row, A_col, A_val, Ag_val = [], [], [], []
        A_shape = (len(raw_lc), self.n_var)
        b, bg = [], []
        for idx, each_lc in enumerate(raw_lc):
            b.append([-each_lc[0][0]])
            bg.append([-each_lc[0][1]])
            A_row.extend([idx] * len(each_lc[1:]))
            A_col.extend([_[1] for _ in each_lc[1:]])
            A_val.extend([_[0][0] for _ in each_lc[1:]])
            Ag_val.extend([_[0][1] for _ in each_lc[1:]])

        self._lc = (
            coo((A_val, (A_row, A_col)), shape=A_shape),
            coo((Ag_val, (A_row, A_col)), shape=A_shape),
            np.array(b),
            np.array(bg),
        )

    def log_info(self):
        """
        log 输出任务信息
        """
        g_vals = [round(_, 3) for _ in self.g_vals]
        self.logger.info("*** Task Info ***")
        self.logger.info(f"Name      : {self.name}")
        self.logger.info(f"Variables : {self.n_var}")
        self.logger.info(f"Parameters: {g_vals}")
        self.logger.info(f"EPS       : {self.eps}")
        self.logger.info(f"Target    : {self.target}")
        self.logger.info(f"PSD       : {self.n_psd}")
        self.logger.info(f"PSD dim   : {self.d_psd}")
        self.logger.info(f"Linear    : {self.n_lc}")
        self.logger.info("*****************")

    def __str__(self):
        """
        打印任务信息

        *** Task Info ***
        Name      : 任务名
        Variables : 变量个数
        Parameters: 参数取值
        EPS       : 精度
        Target    : 目标函数
        PSD       : PSD 约束个数
        PSD dim   : PSD 约束维度
        Linear    : 线性约束数量
        """
        g_vals = [round(_, 3) for _ in self.g_vals]
        s = f"""*** Task Info ***\n- Name      : {self.name}\n- Variables : {self.n_var}\n- Parameters: {g_vals}\n- EPS       : {self.eps}\n- Target    : {self.target}\n- PSD       : {self.n_psd}\n- PSD dim   : {self.d_psd}\n- Linear    : {self.n_lc}\n*****************"""
        return s
