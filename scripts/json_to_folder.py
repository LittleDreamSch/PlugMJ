"""
将旧格式 JSON 任务文件转换为新的文件夹格式。

用法: python scripts/json_to_folder.py <input.json> <output_dir>
"""

import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix


def write_mtx(path: str, mat: coo_matrix):
    """将 scipy 稀疏矩阵写入 MTX 文件。"""
    mat = coo_matrix(mat)
    nrows, ncols = mat.shape
    nnz = mat.nnz
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{nrows} {ncols} {nnz}\n")
        for r, c, v in zip(mat.row, mat.col, mat.data):
            # 避免写入 -0
            if v == 0.0:
                v = 0.0
            f.write(f"{r + 1} {c + 1} {v}\n")


def convert(input_json: str, output_dir: str):
    with open(input_json, "r") as f:
        task = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # task.json
    meta = {"target": task["target"]}
    if "taskname" in task:
        meta["taskname"] = task["taskname"]
    if "eps" in task:
        meta["eps"] = task["eps"]
    if "para_value" in task:
        meta["para_value"] = task["para_value"]
    if "variable_length" in task:
        meta["variable_length"] = task["variable_length"]

    with open(os.path.join(output_dir, "task.json"), "w") as f:
        json.dump(meta, f, indent=4)

    n_var = task["variable_length"]
    dims = task["constrains_dim"]
    raw_psd = task["constrains"]

    # PSD/
    for i, each_psd in enumerate(raw_psd):
        psd_dir = os.path.join(output_dir, "PSD", f"PSD_{i + 1}")
        os.makedirs(psd_dir, exist_ok=True)
        dim = dims[i]

        if "Cons" not in each_psd:
            # 无常系数项，写入空矩阵
            write_mtx(
                os.path.join(psd_dir, "cons.mtx"),
                coo_matrix((dim, dim)),
            )

        for key, (rows, cols, vals) in each_psd.items():
            mat = coo_matrix(
                (np.array(vals, dtype=np.float64), (rows, cols)),
                shape=(dim, dim),
            )
            fname = "cons.mtx" if key == "Cons" else f"{key}.mtx"
            write_mtx(os.path.join(psd_dir, fname), mat)

    # LR/
    raw_lc = task.get("eqConstrains", [])
    if len(raw_lc) > 0:
        lr_dir = os.path.join(output_dir, "LR")
        os.makedirs(lr_dir, exist_ok=True)

        A_row, A_col, A_val, Ag_val = [], [], [], []
        b, bg = [], []

        for idx, each_lc in enumerate(raw_lc):
            b.append([-each_lc[0][0]])
            bg.append([-each_lc[0][1]])
            A_row.extend([idx] * len(each_lc[1:]))
            A_col.extend([_[1] for _ in each_lc[1:]])
            A_val.extend([_[0][0] for _ in each_lc[1:]])
            Ag_val.extend([_[0][1] for _ in each_lc[1:]])

        n_lc = len(raw_lc)
        shape = (n_lc, n_var)

        A = coo_matrix((A_val, (A_row, A_col)), shape=shape)
        Ag = coo_matrix((Ag_val, (A_row, A_col)), shape=shape)
        D1 = coo_matrix(-np.array(b))  # -(b) = D1
        D2 = coo_matrix(-np.array(bg))  # -(bg) = D2

        write_mtx(os.path.join(lr_dir, "C_0.mtx"), A)
        write_mtx(os.path.join(lr_dir, "C_1.mtx"), Ag)
        write_mtx(os.path.join(lr_dir, "D_0.mtx"), D1)
        write_mtx(os.path.join(lr_dir, "D_1.mtx"), D2)

    print(f"Converted: {input_json} -> {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
