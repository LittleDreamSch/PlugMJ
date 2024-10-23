# PlugMJ

从 `Mathematica` 中导出 SDP 问题的 JSON 文件，再应用 `python` 端调用 `cvxpy` 或者 `MOSEK Optimizer API` 执行求解。

## 安装

### Prerequisites

- `Python >= 3.11`

### Install

`Mathematica` 端程序安装：

- 将 `source/ToCVXPY.ml` 拷贝到 `Mathematica` 的 `Applications` 目录下

`python` 端程序安装：

- `pip install PlugMJ-1.1.1-py3-none-any.whl`

## 使用

### `Mathematica` 

```mathematica
<<ToCVXPY`
ToCVXPY beta 1.0.0
------------------
Use GenerateTask[target, allVars, sdpMatrix, loopEquations, para, lambda, eps]
to create Task.json in order to transport the SDP question into CVXPY
Parameter: 
  target: The coefficients of the target function.
          {1, 0, 2} will give allVars[1] + 2 allVars[3] as target.
  allVars: All the variables shown in this SDP problem. 
  sdpMatrix: SDP Matrix as constrains. 
  loopEquations: Equations constrains. Can include a parameter.
  para: The parameter shown in loopEquations. 
  lambda: The discrete values of para. 
  eps: The threshold of the SDP algorithm. It must be in the interval [10^-9, 10^-3]
------------------
DREAM @ 20240603

In[0]:= corMatrix = {{x2 - x4 + 1, x3}, {x3, x2 / 4 + x4}}

In[1]:= GenerateTask[{1, 0, 0}, {x2, x3, x4},{corMatrix}, {x3 = g * x2 - x4}, g, Table[i,{i,.1, 1, .1}], 10^-6]
[Matrix] Phasing Matrices.
[LoopEqs] Phasing LoopEqs.
Exporting...
The Task.json has been created.
Variables in allVars have been renamed. 
```

便会在执行目录下生成 `Task.json` 文件。更多例子请参考 `source/example`。

### `python`

```txt
 __________________________ 
< PlugMJ Beta 1.2.3 @Dream >
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
  -t TASK, --task TASK  Path of the task file.
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
  - `PlugMJ -t Task_4D.json -o result_4D.csv -d max -l log_4D.log -i cvxpy -T 4`: 使用 `cvxpy` 接口，求解 `Task_4D.json` 问题，最大化目标函数，并将结果保存到 `result_4D.csv` 文件中，并将日志保存到 `log_4D.log` 文件中，使用 4 个线程

#### 输出格式 

`-o result.csv` 会将结果保存到 `result.csv` 文件中，每一行对应着一个参数值的解，表头为:

|参数值 |目标函数值| x1 | x2 | .. | xn |
|-------|----------|----|----|----|----|
|...    |...       |... |... |... |... |
  
## 问题

- cvxpy 接口给定精度和实际精度不一致，见 [cvxpy_issue_434](https://github.com/cvxpy/cvxpy/issues/434)

