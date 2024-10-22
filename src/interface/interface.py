"""
接口基类
"""

from abc import abstractmethod
from data.task_loader import TaskLoader
from data.data_saver import DataSaver
from utils.log import Log


class Interface:
    def __init__(
        self,
        task_loader: TaskLoader,
        logger: Log,
        data_saver: DataSaver,
        direction: str,
        **solver_options,
    ):
        """
        初始化MOSEK接口

        Args:
            task_loader (TaskLoader): 任务加载器
            logger (Log): 日志器
            data_saver (DataSaver): 数据保存器
            direction (str): 优化方向
            **solver_options: 求解器参数
        """
        self.task_loader = task_loader
        self.logger = logger
        self.data_saver = data_saver
        self.solver_options = solver_options
        if direction not in ["min", "max"]:
            logger.error(f"Invalid direction: {direction}. Must be either min or max.")
            exit(1)
        else:
            self.direction = direction

        self._psd = []

    def solver_options_handler(self, solver_options: dict):
        """
        处理 solver_options

        Args:
            solver_options (dict): 优化器参数
        """
        # MOSEK 参数
        msk_options = solver_options.pop("MOSEK_OPTIONS", None)
        if msk_options is not None:
            self.MOSEK_OPTIONS = msk_options

        # 精度
        eps_opt = solver_options.pop("eps", None)
        if eps_opt is not None:
            self.logger.info(f"Reset eps to {eps_opt}")
            self.eps = eps_opt

    @abstractmethod
    def optimize(self):
        """
        求解
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        目标函数
        """
        if not hasattr(self, "_target"):
            self._target = ([], [])
        return self._target

    @target.setter
    @abstractmethod
    def target(self, c):
        """
        设置目标函数

        Args:
            c: (c_idx, c_val)
                c_idx (list): 目标函数变量的索引
                c_val (list): 目标函数变量的系数
        """
        pass

    @property
    @abstractmethod
    def psd(self):
        """
        获取 PSD 约束
        """
        if not hasattr(self, "_psd"):
            return []
        return self._psd

    @psd.setter
    @abstractmethod
    def psd(self, psd):
        """
        设置 PSD 约束

        Args:
            psd: [
                     [cons_coo_1, [F_coo_1], [Index_1]],
                     [cons_coo_2, [F_coo_2], [Index_2]],
                     ...
                     [cons_coo_m, [F_coo_m], [Index_m]]
                 ]
                 每一项表示一个 PSD 矩阵的上三角部分，cons_coo 为常数项的 coo 格式表示，F_coo 为矩阵的 coo 格式表示，Index 为矩阵的索引，
                 psd 矩阵可以被还原为 cons_coo_1 + F_coo_1 @ Index_1
        """
        pass

    @property
    @abstractmethod
    def eps(self):
        """
        获取精度
        """
        if not hasattr(self, "_eps"):
            self._eps = 1e-6
        return self._eps

    @eps.setter
    @abstractmethod
    def eps(self, eps):
        """
        设置精度

        Args:
            eps (float): 精度
        """
        self._eps = eps

    @property
    @abstractmethod
    def lc(self):
        """
        获取线性约束
        """
        if not hasattr(self, "_lc"):
            self._lc = []
        return self._lc

    @lc.setter
    @abstractmethod
    def lc(self, lcs):
        """
        设置线性约束

        Args:
            lcs (list): 线性约束
                Ag, A, bg, b : (A + g * Ag) @ x == b + g * bg
        """
        pass

    @staticmethod
    def tolerance_params():
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
