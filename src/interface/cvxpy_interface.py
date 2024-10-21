"""
cvxpy 接口
"""

# TODO: 多线程
# 参考 https://www.cvxpy.org/tutorial/intro/index.html#changing-the-problem 末尾

# TODO: get_problem_data 获取 mosek 标准问题
# 参考 https://www.cvxpy.org/tutorial/advanced/index.html

import cvxpy as cp
import numpy as np
from interface.interface import Interface
from data.task_loader import TaskLoader
from data.data_saver import DataSaver
from utils.log import Log


class CvxpyInterface(Interface):
    def __init__(
        self,
        task_loader: TaskLoader,
        log: Log,
        data_saver: DataSaver,
        direction: str,
        **MOSEK_OPTIONS,
    ):
        """
        初始化 cvxpy 接口

        Args:
            task_loader (TaskLoader): 任务加载器
            logger (Log): 日志器
            data_saver (DataSaver): 数据保存器
            direction (str): 优化方向
            **MOSEK_OPTIONS: MOSEK 参数
        """
        super().__init__(task_loader, log, data_saver, direction, **MOSEK_OPTIONS)
        log.info("use cvxpy interface")

        self.init_cvxpy()
        self.init_problem()

    def init_cvxpy(self):
        """
        初始化 cvxpy 环境
        """
        # 变量和常数
        self.x = cp.Variable(shape=(self.task_loader.n_var, 1), name="x")
        self.para = cp.Parameter(name="g", value=0)
        # 目标函数
        self.target = self.task_loader.target[0], self.task_loader.target[1]
        # 线性约束
        self.lc = self.task_loader.lc
        # psd
        self.psd = self.task_loader.psd
        # 线性不等式
        self.ineqs = -1, 1

    def init_problem(self):
        """
        初始化问题
        """
        constraints = self.lc + self.psd + self.ineqs
        self.problem = cp.Problem(self._target, constraints)

    @property
    def target(self):
        return super().target

    @target.setter
    def target(self, c):
        t = cp.sum([self.x[c[0][_]][0] * c[1][_] for _ in range(len(c[0]))])
        if self.direction == "min":
            self._target = cp.Minimize(t)
        else:
            self._target = cp.Maximize(t)

    @property
    def lc(self):
        return super().lc

    @lc.setter
    def lc(self, lcs):
        A, Ag, b, bg = lcs
        self._lc = [A @ self.x + self.para * (Ag @ self.x) == b + bg * self.para]

    @property
    def psd(self):
        return super().psd

    @psd.setter
    def psd(self, psd):
        # TODO: vectorize LMI，没找到 cvxpy 的例子
        self._psd = []
        for each_psd in psd:
            cons, f, var_idx = each_psd
            # NOTE: CVXPY 不要求 X 是对称的以要求其为 PSD，X >> 0 会要求 X + X.T >> 0，对角线元素需要 / 2
            dig = cons.row == cons.col
            cons.data[dig] *= 0.5
            for _ in range(len(f)):
                f[_].data[f[_].row == f[_].col] *= 0.5

            # psd = cons + f[xi] * xi
            self._psd.append(
                cons
                + cp.sum([f[_] * self.x[var_idx[_]][0] for _ in range(len(var_idx))])
                >> 0
            )

    @property
    def ineqs(self):
        """
        线性不等式
        """
        if self._ineqs is None:
            self._ineqs = []
        return self._ineqs

    @ineqs.setter
    def ineqs(self, bound):
        """
        设置线性不等式

        Args:
            bound(tuple) = lb, ub
                lb (float): 下界
                ub (float): 上界
        """
        self._ineqs = [bound[0] <= self.x, self.x <= bound[1]]

    def eps_handler(self, eps):
        """
        生成 mosek 的 eps 参数

        Args:
            eps (float): 精度

        Returns:
            dict: mosek 参数
        """
        params = Interface.tolerance_params()
        return {key: eps for key in params}

    def optimize(self):
        # NOTE: cvxpy 的 eps 参数不可信，其会设置 infeasible 的阈值设置为 eps，导致 mosek 过早结束
        # 精度讨论：https://github.com/cvxpy/cvxpy/issues/434
        # 尽管 cvxpy 回汇报 OPTIMAL 的状态和解，但是可能会有大条件破坏

        # 构造 Mosek 参数
        msk_params = {
            **self.MOSEK_OPTIONS,
            **self.eps_handler(1e-6),
        }

        self.logger.info("Start optimizing")
        for i, g_val in enumerate(self.task_loader.g_vals):
            self.logger.info("======================")
            self.logger.info(
                f"[{i + 1} / {len(self.task_loader.g_vals)}] Set g = {g_val}"
            )
            # 更新 g 值
            self.para.value = g_val
            # 优化
            self.problem.solve(
                solver=cp.MOSEK,
                verbose=True,
                canon_backend=cp.SCIPY_CANON_BACKEND,
                mosek_params=msk_params,
            )

            # 打印时间
            self.logger.info(f"Compilation  Time: {self.problem.compilation_time}")
            self.logger.info(f"Optimization Time: {self.problem._solve_time}")

            # 获得结果
            status = self.problem.status
            if status == cp.OPTIMAL or status == cp.OPTIMAL_INACCURATE:
                if status == cp.OPTIMAL:
                    self.logger.success(f"[g = {g_val}] STATUS: OPTIMAL")
                else:
                    self.logger.warning(f"[g = {g_val}] STATUS: OPTIMAL_INACCURATE")
                # 获取结果
                obj = self.problem.value
                sol = [_[0] for _ in self.x.value]
                self.logger.info(f"Objective value: {obj}")
                # 储存结果
                # g_val, obj, sol
                self.data_saver.append(
                    np.concatenate((np.array([g_val, obj]), np.array(sol)))
                )
            elif status == cp.INFEASIBLE:
                self.logger.error(
                    f"[g = {g_val}] STATUS: INFEASIBLE. Result won't be stored."
                )
        self.data_saver.save()
