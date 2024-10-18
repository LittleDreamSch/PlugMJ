"""
提供了与MOSEK的接口
"""

import mosek
import numpy as np
from interface.interface import Interface
from data.task_loader import TaskLoader
from data.data_saver import DataSaver
from utils.log import Log


class MosekInterface(Interface):
    def __init__(
        self,
        task_loader: TaskLoader,
        logger: Log,
        data_saver: DataSaver,
        **MOSEK_OPTIONS,
    ):
        """
        初始化MOSEK接口

        Args:
            task_loader (TaskLoader): 任务加载器
            logger (Log): 日志器
            data_saver (DataSaver): 数据保存器
            **MOSEK_OPTIONS: MOSEK 参数
        """
        super().__init__(task_loader, logger, data_saver, **MOSEK_OPTIONS)

        # 初始化 MOSEK
        self.init_mosek()

        ## 定义问题
        self.init_problem()

        ## Logger
        self.task.set_Stream(mosek.streamtype.log, self.mosek_log)

        # 设置参数
        self.mosek_options_handdler(**MOSEK_OPTIONS)

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
        # 创建MOSEK环境和任务
        self.env = mosek.Env()
        self.task = self.env.Task()

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
        # 设置初始点
        self.set_start_point()

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
        return super().target

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
        return super().psd

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
        return super().g

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

    @property
    def lc(self):
        return super().lc

    @lc.setter
    def lc(self, lcs):
        pass

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
        return super().eps

    @eps.setter
    def eps(self, eps):
        """
        设置精度

        Args:
            eps (float): 精度
        """
        self._eps = eps
        tol_params = self.tolerance_params()
        for param in tol_params:
            self.task.putparam(param, str(self._eps))

    def set_start_point(self):
        # TODO: 设置初始点为满足变量边界条件的点，这里的 constant 怎么指定?
        self.task.putintparam(
            mosek.iparam.intpnt_starting_point, mosek.startpointtype.constant
        )
        pass

    def mosek_options_handdler(self, **MOSEK_OPTIONS):
        """
        设置 MOSEK 参数

        Args:
            **MOSEK_OPTIONS: MOSEK 参数
        """
        for key, value in MOSEK_OPTIONS.items():
            self.task.putparam(key, str(value))

    @property
    def thread(self):
        """
        获取线程数
        """
        if self._thread is None:
            self._thread = 0
        return self._thread

    @thread.setter
    def thread(self, thread):
        """
        设置线程数
        Args:
            thread (int): 线程数
        """
        self._thread = thread
        self.task.putintparam(mosek.iparam.num_threads, thread)

    def save_model(self, name="task.ptf"):
        """
        保存模型
        """
        self.task.writedata(name)
