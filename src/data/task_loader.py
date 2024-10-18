"""
加载并格式化给定的 Json 文件
"""

import numpy as np
import json
import time
import os


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
        self.target = self.load_target(self.task_json["target"])
        # PSD 约束
        self.psd = self.load_psd(self.d_psd, self.task_json["constrains"])
        # 线性约束
        self.lc = self.load_lc(self.task_json["eqConstrains"])
        # 线性约束数量
        self.n_lc = len(self.task_json["eqConstrains"])

        # 打印加载时间
        end_time = time.time()
        self.logger.success(f"Loading task time: {end_time - start_time:.4f}s")

    def load_target(self, raw_target):
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
        return c_index, c_value

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
            vec_psd: sMat(psd) 格式表示的 PSD 矩阵上三角部分
                     [[f_row], [f_col], [f_val], [g_row], [g_val]]

                     表示 Mosek 的 conic 约束中 F[f_row[i], f_col[i]] = f_vals[i], g[g_row[i]] = g_vals[i]
        """
        self.logger.info("Loading PSD ..")

        def sMat(psd, dim):
            """
            将 PSD 矩阵转换为 sMat 格式

            sMat([m11, m12, m13;
                  m12, m22, m23;
                  m13, m23, m33]) = [m11, sqrt(2) * m12, sqrt(2) * m13, m22, sqrt(2) * m23, m33]

            Args:
                psd: 原始 PSD 矩阵数据，[[row], [col], [val]]
                dim: 矩阵维度

            Returns:
                row: 非零元素行索引
                val: 非零元素的值
            """
            row = []
            val = []

            # 转置 psd
            psd_t = [[psd[0][i], psd[1][i], psd[2][i]] for i in range(len(psd[0]))]
            # 排序
            psd_t = sorted(psd_t, key=lambda x: (x[0], x[1]))

            for each_psd in psd_t:
                id = int(
                    1 / 2 * (2 * dim - each_psd[0] + 1) * each_psd[0]
                    + each_psd[1]
                    - each_psd[0]
                )
                row.append(id)  # 计算列索引
                val.append(
                    each_psd[2]
                    if each_psd[0] == each_psd[1]
                    else np.sqrt(2) * each_psd[2]
                )

            return row, val

        # 转化所有的 PSD 为 sMat 格式
        start_row = 0
        psd_cols, psd_rows, psd_vals, g_rows, g_vals = [], [], [], [], []
        for id_psd, each_psd in enumerate(raw_psd):
            for key in each_psd:
                if key == "Cons":
                    g_row, g_val = sMat(each_psd[key], dims[id_psd])
                    g_rows.extend([_ + start_row for _ in g_row])
                    g_vals.extend(g_val)
                else:
                    row, val = sMat(each_psd[key], dims[id_psd])
                    psd_cols.extend([int(key)] * len(row))
                    psd_rows.extend([_ + start_row for _ in row])
                    psd_vals.extend(val)

            start_row += int(dims[id_psd] * (dims[id_psd] + 1) / 2)
        return psd_rows, psd_cols, psd_vals, g_rows, g_vals

    def load_lc(self, raw_lc):
        """
        加载线性约束

        等式约束的格式为 [cons, [i1, j1], ..., [in, jn]]，表示

        (i1[0] + i1[1] * g) * x[j1] + ... == -(cons[0] + cons[1] * g)

        Args:
            raw_lc: 原始线性约束数据

        Returns:
            np.array: [
                        [[i1_1], [i2_1], ..., [in_1]],
                        [[i1_2], [i2_2], ..., [in_2]],
                        ...,
                        [[i1_n], [i2_n], ..., [in_n]]
                      ]
            列索引:   [
                        [j1_1, j2_1, ..., jn_1],
                        [j1_2, j2_2, ..., jn_2],
                        ...,
                        [j1_n, j2_n, ..., jn_n]
                      ]
            np.array: 常数项 [[cons_1], [cons_2], ..., [cons_n]]
        """
        self.logger.info("Loading linear constraints...")
        lc, col, b = [], [], []
        for each_lc in raw_lc:
            # cons
            b.append([-each_lc[0][0], -each_lc[0][1]])
            # A
            lc.append(np.array([_[0] for _ in each_lc[1:]]))  # [[i1], [i2], ..., [in]]
            col.append([_[1] for _ in each_lc[1:]])  # [j1, j2, ... , jn]，列索引
        return lc, col, np.array(b)

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
