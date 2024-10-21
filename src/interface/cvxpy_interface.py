"""
cvxpy 接口
"""

# TODO: 多线程
# 参考 https://www.cvxpy.org/tutorial/intro/index.html#changing-the-problem 末尾

# TODO: get_problem_data 获取 mosek 标准问题
# 参考 https://www.cvxpy.org/tutorial/advanced/index.html

import cvxpy as cp
from cvxpy.reductions.solvers import solver
import numpy as np
from scipy.sparse import coo_matrix
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

        # 总时间
        self.total_time = 0.0
        self.total_complie_time = 0.0

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
        if A.shape[0] != 0 and Ag.shape[0] != 0:
            self._lc = [A @ self.x + self.para * (Ag @ self.x) == b + bg * self.para]
        elif A.shape[0] == 0 and Ag.shape[0] != 0:
            self._lc = [self.para * (Ag @ self.x) == b + bg * self.para]
        elif A.shape[0] != 0 and Ag.shape[0] == 0:
            self._lc = [A @ self.x == b + bg * self.para]

    @property
    def psd(self):
        return super().psd

    @psd.setter
    def psd(self, psd):
        self._psd = []
        for each_psd in psd:
            cons, f, var_idx = each_psd
            f, cons = self.vectorize_psd(cons, f, var_idx)
            afe = cp.vec_to_upper_tri(f @ self.x + cons)
            self._psd.append(afe >> 0)

    def vectorize_psd(self, cons, f, var_idx):
        """
        psd 约束向量化

        Args:
            cons (scipy.sparse.coo_matrix): 标量矩阵
            f (list): 系数矩阵
            var_idx (list): 变量索引

        Returns:
            F (np.array, d*(d+1)/2 x n_var): 向量化后的系数矩阵
            cons (np.array, d*(d+1)/2 x 1): 向量化后的标量矩阵
        """

        def vec(mat):
            """
            向量化矩阵

            Args:
                mat (scipy.sparse.coo_matrix): 上三角稀疏矩阵

            Returns:
                row (list): 行索引
                val (list): 值

            对角线上的元素会 /2，以满足 cvxpy 的格式
            """
            t_row = mat.row
            t_col = mat.col
            t_val = mat.data
            dim = mat.shape[0]

            row, val = [], []
            for i in range(len(t_row)):
                # 值
                if t_row[i] == t_col[i]:
                    val.append(
                        t_val[i] * 0.5
                    )  # NOTE: CVXPY 不要求 X 是对称的以要求其为 PSD，X >> 0 会要求 X + X.T >> 0，对角线元素需要 / 2
                else:
                    val.append(t_val[i])
                # 索引
                row.append(
                    int(0.5 * (2 * dim - t_row[i] + 1) * t_row[i] + t_col[i] - t_row[i])
                )
            return row, val

        dim = cons.shape[0]
        # 系数矩阵
        f_row, f_val, f_col = [], [], []
        for i in range(len(f)):
            row, val = vec(f[i])
            f_row += row
            f_col += [var_idx[i]] * len(row)
            f_val += val
        f = coo_matrix(
            (f_val, (f_row, f_col)),
            shape=(dim * (dim + 1) // 2, self.task_loader.n_var),
        )
        # 标量矩阵
        cons_row, cons_val = vec(cons)
        cons = coo_matrix(
            (cons_val, (cons_row, np.zeros(len(cons_row)))),
            shape=(dim * (dim + 1) // 2, 1),
        )
        return f, cons

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
            **self.eps_handler(self.eps),
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
            solve_time = self.problem._solve_time
            compile_time = self.problem.compilation_time
            self.logger.info(f"Compilation  Time: {compile_time}")
            self.logger.info(f"Optimization Time: {solve_time}")

            if solve_time is not None:
                self.total_time += solve_time
            if compile_time is not None:
                self.total_complie_time += compile_time

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

        # 输出总时间
        self.logger.info("*************************")
        self.logger.info(f"- Total Optimization Time : {self.total_time:.2f} s")
        self.logger.info(f"- Total Compilation  Time : {self.total_complie_time:.2f} s")
        self.logger.info(
            f"- Total Time              : {self.total_complie_time + self.total_time:.2f} s"
        )

        # 输出结果到文件
        self.data_saver.save()
