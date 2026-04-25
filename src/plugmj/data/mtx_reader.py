"""
自定义 Matrix Market (MTX) 文件读取器，支持有理数值。
"""

from fractions import Fraction

import numpy as np
from scipy.sparse import coo_matrix


def _parse_value(s: str) -> float:
    if "/" in s:
        return float(Fraction(s))
    return float(s)


def read_mtx(path: str) -> coo_matrix:
    """
    读取 MTX 文件，返回 scipy.sparse.coo_matrix。
    支持 value 字段中的有理数（如 3/4）。

    Args:
        path: MTX 文件路径
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # 解析头部
    header = lines[0].strip()
    if not header.startswith("%%MatrixMarket"):
        raise ValueError(f"Invalid MTX header in {path}: {header}")

    parts = header.split()
    if len(parts) < 5 or parts[2] != "coordinate":
        raise ValueError(f"Expected coordinate matrix, got: {header}")

    # 跳过注释行
    idx = 1
    while idx < len(lines) and lines[idx].startswith("%"):
        idx += 1

    # 尺寸行
    nrows, ncols, nnz = map(int, lines[idx].split())
    idx += 1

    if nnz == 0:
        return coo_matrix((nrows, ncols))

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)

    for i in range(nnz):
        parts = lines[idx + i].split()
        rows[i] = int(parts[0]) - 1  # 1-based → 0-based
        cols[i] = int(parts[1]) - 1
        vals[i] = _parse_value(parts[2])

    return coo_matrix((vals, (rows, cols)), shape=(nrows, ncols))
