"""
文件夹格式的任务加载器。
"""

import json
import os
import re
import time

import numpy as np
from scipy.sparse import coo_matrix as coo

from plugmj.data.mtx_reader import read_mtx


def _natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


class FolderTaskLoader:
    """
    从文件夹结构加载 SDP 任务。

    产出与 TaskLoader 完全相同的属性接口。
    """

    def __init__(self, task_dir: str, logger, name: str = "SDP_TASK"):
        if not os.path.isdir(task_dir):
            logger.critical(f"Task directory {task_dir} not found.")
            exit(1)

        self.logger = logger
        self.logger.info(f"Load task from folder {task_dir}")

        task_json_path = os.path.join(task_dir, "task.json")
        if not os.path.isfile(task_json_path):
            logger.critical(f"task.json not found in {task_dir}")
            exit(1)

        with open(task_json_path, "r") as f:
            meta = json.load(f)

        self.name = meta.get("taskname", name)
        self._load(meta, task_dir)
        self._log_info()

    def _load(self, meta: dict, task_dir: str):
        start_time = time.time()

        # 精度
        self.eps = meta.get("eps", 1e-6)

        # 参数扫描值
        para_value = meta.get("para_value", [])
        self.g_vals = para_value if len(para_value) > 0 else [0]

        # 目标函数
        self._target = self._parse_target(meta["target"])

        # PSD 约束
        psd_dir = os.path.join(task_dir, "PSD")
        self._psd, self.n_psd, self.d_psd = self._load_psd(psd_dir)

        # n_var：优先使用 task.json 中的声明，否则从 MTX 文件推断
        inferred_n_var = self._infer_n_var()
        self.n_var = meta.get("variable_length", inferred_n_var)
        if self.n_var < inferred_n_var:
            self.logger.warning(
                f"variable_length={self.n_var} < inferred {inferred_n_var}, using inferred value"
            )
            self.n_var = inferred_n_var

        # 线性约束
        lr_dir = os.path.join(task_dir, "LR")
        if os.path.isdir(lr_dir):
            self._lc, self.n_lc = self._load_lc(lr_dir)
        else:
            self._lc = []
            self.n_lc = 0

        self.logger.success(f"Loading task time: {time.time() - start_time:.4f}s")

    @property
    def target(self):
        return self._target

    @property
    def psd(self):
        return self._psd

    @property
    def lc(self):
        return self._lc

    @staticmethod
    def _parse_target(raw_target):
        c_index = [c[0] for c in raw_target]
        c_value = [c[1] for c in raw_target]
        return c_index, c_value

    def _load_psd(self, psd_dir: str):
        if not os.path.isdir(psd_dir):
            self.logger.critical(f"PSD directory not found: {psd_dir}")
            exit(1)

        self.logger.info("Loading PSD from MTX files...")
        psds = []
        dims = []

        subdirs = sorted(
            [d for d in os.listdir(psd_dir) if os.path.isdir(os.path.join(psd_dir, d))],
            key=_natural_sort_key,
        )

        for subdir in subdirs:
            subdir_path = os.path.join(psd_dir, subdir)
            cons_path = os.path.join(subdir_path, "cons.mtx")

            if not os.path.isfile(cons_path):
                self.logger.critical(f"cons.mtx not found in {subdir_path}")
                exit(1)

            cons_coo = read_mtx(cons_path)
            dim = cons_coo.shape[0]
            dims.append(dim)

            f_coos = []
            var_indices = []

            for fname in sorted(os.listdir(subdir_path), key=_natural_sort_key):
                if fname == "cons.mtx" or not fname.endswith(".mtx"):
                    continue
                var_idx = int(fname[:-4])  # "0.mtx" → 0
                f_coo = read_mtx(os.path.join(subdir_path, fname))
                f_coos.append(f_coo)
                var_indices.append(var_idx)

            psds.append([cons_coo, f_coos, var_indices])

        return psds, len(psds), dims

    def _infer_n_var(self) -> int:
        max_var = 0
        for _, _, var_indices in self._psd:
            if var_indices:
                max_var = max(max_var, max(var_indices) + 1)
        return max_var

    def _load_lc(self, lr_dir: str):
        self.logger.info("Loading linear constraints from MTX files...")

        # 扫描 C_k.mtx 和 D_k.mtx，k 为 λ 的幂次
        c_files, d_files = {}, {}
        for f in os.listdir(lr_dir):
            m = re.match(r"C_(\d+)\.mtx$", f)
            if m:
                c_files[int(m.group(1))] = os.path.join(lr_dir, f)
            m = re.match(r"D_(\d+)\.mtx$", f)
            if m:
                d_files[int(m.group(1))] = os.path.join(lr_dir, f)

        if not c_files and not d_files:
            return [], 0

        max_deg = max(max(c_files.keys(), default=0), max(d_files.keys(), default=0))

        # 从任意一个文件推断维度
        first_file = list(c_files.values())[0] if c_files else list(d_files.values())[0]
        first_mat = read_mtx(first_file)
        n_lc = first_mat.shape[0]
        n_var = first_mat.shape[1] if first_mat.shape[1] > 1 else self.n_var

        lc = []
        for k in range(max_deg + 1):
            C_k = read_mtx(c_files[k]) if k in c_files else coo((n_lc, n_var))
            if k in d_files:
                D_k = read_mtx(d_files[k]).toarray().reshape(n_lc, 1)
            else:
                D_k = np.zeros((n_lc, 1))
            lc.append((C_k, D_k))

        return lc, n_lc

    def _log_info(self):
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
        g_vals = [round(_, 3) for _ in self.g_vals]
        return (
            f"*** Task Info ***\n"
            f"- Name      : {self.name}\n"
            f"- Variables : {self.n_var}\n"
            f"- Parameters: {g_vals}\n"
            f"- EPS       : {self.eps}\n"
            f"- Target    : {self.target}\n"
            f"- PSD       : {self.n_psd}\n"
            f"- PSD dim   : {self.d_psd}\n"
            f"- Linear    : {self.n_lc}\n"
            f"*****************"
        )
