"""
提供了与MOSEK的接口
"""

import mosek
import numpy as np
from loguru import logger
from core.task_loader import TaskLoader
from core.data_saver import DataSaver
from core.log import Log


class MosekInterface:
    def __init__(
        self, task_loader: TaskLoader, logger: Log, data_saver: DataSaver, g=0
    ):
        """
        初始化MOSEK接口

        Args:
            task_loader (TaskLoader): 任务加载器
            logger (Log): 日志器
            data_saver (DataSaver): 数据保存器
            g (float): 参数值
        """
        self.task_loader = task_loader
        self.logger = logger
        self.data_saver = data_saver

        # 创建MOSEK环境和任务
        self.env = mosek.Env()
        self.task = self.env.Task()

        # 初始化 MOSEK
        self.init_mosek()

        ## 定义问题
        self.init_problem()

        ## Logger
        self.task.set_Stream(mosek.streamtype.log, self.mosek_log)
        self.task.putintparam(mosek.iparam.log_cut_second_opt, 0)  # 关闭 mosek 日志衰减

    def mosek_log(self, msg):
        """
        MOSEK 日志接口

        Args:
            msg (str): 日志信息
        """
        self.logger.mosek(msg.rstrip())  # 删去末尾的换行符

    def init_mosek(self):
        """
        初始化 MOSEK 参数
        """
        ## 初始化 MOSEK 参数
        # 设置任务名
        self.task.puttaskname(self.task_loader.name)
        # 设置变量数量
        self.task.appendvars(self.task_loader.n_var)
        # 设置线性约束数量
        self.task.appendcons(self.task_loader.n_lc)
        # 设置 PSD 约束的维数，每个 d 维的 psd 提供了 d(d+1)/2 个约束
        dims = np.array(self.task_loader.d_psd)
        self.d_vec = dims * (dims + 1) // 2
        self.task.appendafes(np.sum(self.d_vec))
        # 设置 PSD 变量的维数
        self.task.appendbarvars(dims)
        # 设置精度
        self.eps = self.task_loader.eps

    def init_problem(self):
        """
        初始化问题
        """
        # 设置目标函数
        self.target = self.task_loader.target[0], self.task_loader.target[1]
        # 设置 psd
        self.psd = self.task_loader.psd
        # 设置变量，线性约束于此引入
        self.g = 0
        # 不等式约束，-1 <= x[i] <= 1
        self.parse_ineqs()

    def optimize(self):
        """
        优化问题
        """
        self.logger.info("Start optimizing")
        for i, g_val in enumerate(self.task_loader.g_vals):
            self.logger.info("======================")
            self.logger.info(
                f"[{i + 1} / {len(self.task_loader.g_vals)}] Set g = {g_val}"
            )
            # 更新 g 值
            self.g = g_val
            # 求解
            self.task.optimize()
            self.task.solutionsummary(mosek.streamtype.msg)

            # 获取问题状态
            solsta = self.task.getsolsta(mosek.soltype.itr)
            if solsta == mosek.solsta.optimal:
                # TODO: 最优解保存
                self.logger.success(f"Find optimal solution for g = {g_val}")
                sol = np.array(self.task.getxx(mosek.soltype.itr))
                # 插入最优解
                data = np.array(
                    [
                        g_val,  # 变量值
                        self.task.getprimalobj(mosek.soltype.itr),  # 原始问题解
                        self.task.getdualobj(mosek.soltype.itr),  # 对偶解
                    ]
                )
                # 保存
                self.data_saver.append(np.concatenate((data, sol)))
            elif (
                solsta == mosek.solsta.dual_infeas_cer
                or solsta == mosek.solsta.prim_infeas_cer
            ):
                self.logger.error(
                    f"Infeasible solution for g = {g_val}. Result won't be saved. "
                )
            else:
                self.logger.error(
                    f"Unknown solution status for g = {g_val}. Result won't be saved. "
                )
        # 保存结果
        self.data_saver.save()

    @property
    def target(self):
        if self._target is None:
            self._target = ([], [])
        return self._target

    @target.setter
    def target(self, c):
        """
        设置目标函数

        Args:
            c: (c_idx, c_val)
                c_idx (list): 目标函数变量的索引
                c_val (list): 目标函数变量的系数
        """
        c_idx, c_val = c
        self._target = (c_idx, c_val)
        self.task.putclist(c_idx, c_val)

        # TODO: 添加优化的方向的选项
        self.task.putobjsense(mosek.objsense.minimize)

    @property
    def psd(self):
        """
        获取 PSD 约束
        """
        if self._psd is None:
            return []
        return self._psd

    @psd.setter
    def psd(self, psd):
        """
        设置 PSD 约束

        Args:
            psd = f_row, f_col, f_val, g_row, g_val
                f_row (list): PSD 变量的行索引
                f_col (list): PSD 变量的列索引
                f_val (list): PSD 变量的系数
                g_row (list): PSD 常数项索引
                g_val (list): PSD 常数项系数
        """
        f_row, f_col, f_val, g_row, g_val = psd
        self._psd = (f_row, f_col, f_val, g_row, g_val)
        # F
        self.task.putafefentrylist(f_row, f_col, f_val)
        # g
        self.task.putafeglist(g_row, g_val)
        # Domain
        cu = np.insert(np.cumsum(self.d_vec), 0, 0)
        for i in range(len(cu) - 1):
            self.task.appendacc(
                self.task.appendsvecpsdconedomain(self.d_vec[i]),
                range(cu[i], cu[i + 1]),
                None,
            )

    @property
    def g(self):
        """
        获取参数值
        """
        if self._g is None:
            self._g = 0
        return self._g

    @g.setter
    def g(self, g):
        """
        设置参数值

        Args:
            g (float): 参数值
        """
        self._g = g

        # 使用新的 g 去更新线性约束
        a, col, cons = self.task_loader.lc
        # BUG: 计算得到的 A 可能为零，MOSEK 会有 WARNING
        self._A = [_[:, 0] + _[:, 1] * g for _ in a]
        self._cons = cons[:, 0] + cons[:, 1] * g
        self._col = col

        # 设置新的线性约束
        for row in range(len(self._A)):
            self.task.putarow(row, col[row], self._A[row])
            self.task.putconbound(
                row, mosek.boundkey.fx, self._cons[row], self._cons[row]
            )

    def parse_ineqs(self):
        """
        解析非线性约束

        会将所有的参数约定 -1 <= x <= 1
        """
        for j in range(self.task_loader.n_var):
            self.task.putvarbound(j, mosek.boundkey.fr, -1, 1)

    @property
    def eps(self):
        """
        获取精度
        """
        if self._eps is None:
            self._eps = 1e-6
        return self._eps

    @eps.setter
    def eps(self, eps):
        """
        设置精度

        Args:
            eps (float): 精度
        """
        self._eps = eps
        tol_params = self.tolerance_params()
        self.logger.debug("SET TOLERANCE PARAMS")
        for param in tol_params:
            self.task.putparam(param, str(self._eps))

    @property
    def start_point(self):
        # TODO: 实现 start_point 的设置，参考 dparam.intpnt_qo_tol_pfeas 和 dparam.intpnt_tol_dfeas 等参数
        pass

    @staticmethod
    def tolerance_params():
        # TODO: 确认一下具体应该设置哪些参数
        # tolerance parameters from
        # https://docs.mosek.com/latest/pythonapi/param-groups.html
        return (
            # Conic interior-point tolerances
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            # "MSK_DPAR_INTPNT_CO_TOL_INFEAS",
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED",
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            # Interior-point tolerances
            "MSK_DPAR_INTPNT_TOL_DFEAS",
            # "MSK_DPAR_INTPNT_TOL_INFEAS",
            "MSK_DPAR_INTPNT_TOL_MU_RED",
            "MSK_DPAR_INTPNT_TOL_PFEAS",
            "MSK_DPAR_INTPNT_TOL_REL_GAP",
            # Simplex tolerances
            "MSK_DPAR_BASIS_REL_TOL_S",
            "MSK_DPAR_BASIS_TOL_S",
            "MSK_DPAR_BASIS_TOL_X",
            # MIO tolerances
            "MSK_DPAR_MIO_TOL_ABS_GAP",
            "MSK_DPAR_MIO_TOL_ABS_RELAX_INT",
            "MSK_DPAR_MIO_TOL_FEAS",
            "MSK_DPAR_MIO_TOL_REL_GAP",
        )

    def save_model(self, name=""):
        # TODO:
        """
        保存模型
        """
        pass
