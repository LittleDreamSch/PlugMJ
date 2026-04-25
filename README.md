# PlugMJ

从 `Mathematica` 中导出 SDP 问题的 JSON 文件，再应用 `python` 端调用 `cvxpy` 或者 `MOSEK Optimizer API` 执行求解。

## 安装

### Prerequisites

- `Python >= 3.11`
- [MOSEK License](https://www.mosek.com/products/academic-licenses/)

### Install

`Mathematica` 端程序安装：

- 将 `source/ToCVXPY.wl` 拷贝到 `Mathematica` 的 `Applications` 目录下

`python` 端程序安装：

```bash
# 从源码安装
pip install -e .

# 或从 wheel 安装
pip install plugmj-1.3.0-py3-none-any.whl
```

## 使用

示例求解问题

$$
\begin{align*}
\min\quad & x_2\\
\text{s.t.}\quad& \begin{pmatrix}
x_2 - x_4 + 1 & x_3 \\
x_3 & \frac{x_2}{4} + x_4
\quad&
\end{pmatrix}
\succeq 0\\
& x_3 = g * x_2 - x_4,\\
& g \in [.1,\ .2,\ .3,\ .4,\ .5,\ .6,\ .7,\ .8,\ .9,\ 1]
\end{align*}
$$

### Mathematica

```mathematica
<<ToCVXPY`
ToCVXPY 1.3.0
------------------
Use GenerateTask[target, allVars, sdpMatrix, loopEquations, para, lambda, eps, name]
to create name.json in order to transport the SDP question into CVXPY
Parameter:
  target: The coefficients of the target function.
          {1, 0, 2} will give allVars[1] + 2 allVars[3] as target.
  allVars: All the variables shown in this SDP problem.
  sdpMatrix: SDP Matrix as constrains.
  loopEquations: Equations constrains. Can include a parameter.
  para: The parameter shown in loopEquations.
  lambda: The discrete values of para.
  eps: The threshold of the SDP algorithm. It must be in the interval [10^-9, 10^-3]
  name: Name of the task. Task will be saved to 'name.json'. Task as default.
------------------
DREAM @ 20241023

In[0]:= corMatrix = {{x2 - x4 + 1, x3}, {x3, x2 / 4 + x4}}

In[1]:= GenerateTask[{1, 0, 0}, {x2, x3, x4},{corMatrix}, {x3 = g * x2 - x4}, g, Table[i,{i,.1, 1, .1}], 10^-6, "SDP_Task"]
[Matrix] Phasing Matrices.
[LoopEqs] Phasing LoopEqs.
Exporting...
The SDP_Task.json has been created.
Variables in allVars have been renamed.
```

便会在执行目录下生成 `Task.json` 文件。更多例子请参考 `source/example`。

### Python

```
 __________________________
< PlugMJ 1.3.0 @Dream >
 --------------------------
        \   ^__^
         \  (OO)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
Execute SDP Json exported from mathematica.

Example:
    PlugMJ -t Task.json -o output.csv -d min
                : Run Task.json and minimize the objective and save the result
                  as output.csv by using cvxpy as default interface

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Path of the task file or directory.
  -o OUTPUT, --output OUTPUT
                        Path of the output file.
  -d DIRECTION, --direction DIRECTION
                        Direction of the task. min or max. (minimize as default)
  -T THREADS, --threads THREADS
                        Number of threads. (0 as default)
  -e EPS, --eps EPS     Tolerence of the solver.
  -i INTERFACE, --interface INTERFACE
                        Interface to use, cvxpy or original. (cvxpy as default)
  -l LOG, --log LOG     Save log to path of the given file.
  -n NAME, --name NAME  Name of the task (disable in cvxpy).
```

示例：
- `PlugMJ -t Task_4D.json -o result_4D.csv -d max -i cvxpy -T 4`: 使用 `cvxpy` 接口，求解 `Task_4D.json` 问题，最大化目标函数，并将结果保存到 `result_4D.csv` 文件中，使用 4 个线程
- `PlugMJ -t task_folder/ -o result.csv`: 从文件夹格式加载任务（见下方说明）

#### 输出格式

`-o result.csv` 会将结果保存到 `result.csv` 文件中，每一行对应着一个参数值的解：

| 参数值 | 目标函数值 | x1 | x2 | .. | xn |
|--------|-----------|----|----|----|----|
| ...    | ...       | ...| ...| ...| ...|

同时会生成 `result_stats.json` 记录运行统计：

```json
{
    "optimization_time": 0.0757,
    "peak_memory_mb": 210.7,
    "avg_memory_mb": 210.7
}
```

运行日志默认保存在任务目录下的 `PlugMJ.log`。

## 文件夹输入格式

除了传统的单文件 JSON 格式外，PlugMJ 支持基于文件夹的输入格式，便于手动编辑和外部工具生成：

```
task_folder/
├── task.json              # 元数据
├── PSD/
│   ├── PSD_1/
│   │   ├── cons.mtx       # 常数项矩阵
│   │   ├── 0.mtx          # x[0] 的系数矩阵
│   │   ├── 1.mtx          # x[1] 的系数矩阵
│   │   └── ...
│   ├── PSD_2/
│   │   └── ...
│   └── ...
└── LR/
    ├── C_0.mtx            # λ^0 系数矩阵 (n_lc × n_var)
    ├── C_1.mtx            # λ^1 系数矩阵
    ├── C_2.mtx            # λ^2 系数矩阵（可选）
    ├── ...
    ├── D_0.mtx            # λ^0 常数项向量 (n_lc × 1)
    ├── D_1.mtx            # λ^1 常数项向量
    └── ...
```

### task.json

```json
{
    "taskname": "MyTask",
    "target": [[0, 2.0], [1, 3.0]],
    "eps": 1e-9,
    "para_value": [0.1, 0.2, 0.3],
    "variable_length": 10
}
```

### PSD 约束

每个 PSD 矩阵表示为 `cons + x[0]·A[0] + x[1]·A[1] + ...`，其中 `cons.mtx` 为常数项，`i.mtx` 为 `x[i]` 的系数矩阵。矩阵值支持有理数（如 `3/4`）。

### 线性约束

线性约束支持 λ 的多项式系数：

$$\sum_k \left(C_k \cdot \lambda^k\right) \cdot X + \sum_k \left(D_k \cdot \lambda^k\right) = 0$$

其中 `C_k.mtx` 和 `D_k.mtx` 分别为 λ^k 对应的系数矩阵和常数项向量。幂次可以不连续（如只有 `C_0.mtx`、`C_1.mtx`、`C_4.mtx`），缺失的幂次自动视为零矩阵。

当 `LR/` 目录不存在时，视为无线性约束。

### 格式转换

使用转换脚本将旧 JSON 格式转为文件夹格式：

```bash
python scripts/json_to_folder.py task/Task.json output_folder/
```

### 自动检测

如果当前目录包含 `task.json` 和 `PSD/` 子目录，可以省略 `-t` 参数：

```bash
cd task_folder/
PlugMJ -o result.csv
```

## 其他

`source` 文件夹内：
- `example/`: `Mathematica` 的例子
- `ToCVXPY.wl`: `Mathematica` 端程序
- `slurm.sh`: `Slurm` 脚本模板

## 问题

- cvxpy 接口给定精度和实际精度不一致，见 [cvxpy_issue_434](https://github.com/cvxpy/cvxpy/issues/434)
