# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlugMJ is a bridge tool that exports Semidefinite Programming (SDP) problems from Mathematica as JSON, then solves them in Python using CVXPY or MOSEK Optimizer API. It supports parameterized problems with discrete parameter sweeps.

## Build & Run

```bash
# Install in dev mode (requires Python 3.11+)
pip install -e .

# Build distribution package
python -m build

# Run a solver task
PlugMJ -t Task.json -o output.csv -d min -i cvxpy -T 4

# CLI options
#   -t  Path to JSON task file (required)
#   -o  Output CSV path (default: output.csv)
#   -d  Direction: min or max (default: min)
#   -i  Interface: cvxpy or original (default: cvxpy)
#   -T  Thread count (0 = auto)
#   -e  Solver tolerance override
#   -l  Log file path
#   -n  Task name (MOSEK interface only)
```

## Architecture

The codebase lives under `src/plugmj/` with package root configured via `pyproject.toml` (`package-dir = {"" = "src"}`). Entry point is `plugmj.plugmj_shell:main`.

```
src/plugmj/
  __init__.py
  plugmj_shell.py          # CLI entry point (argparse → build_solver → optimize)
  data/
    task_loader.py          # Loads/parses JSON task files from Mathematica
    data_saver.py           # Saves results to CSV
  interface/
    interface.py            # Abstract base class (target, psd, lc, eps properties)
    cvxpy_interface.py      # CVXPY + MOSEK backend solver
    mosek_interface.py      # Direct MOSEK Fusion API solver
  utils/
    log.py                  # loguru-based logger with MOSEK log integration
```

### Data Flow

1. `TaskLoader` reads a JSON file describing SDP constraints (PSD matrices in sparse COO format, linear constraints with optional parameter `g`, objective coefficients)
2. `Interface` subclass builds the optimization problem — CVXPY uses `cp.Variable` + `cp.Parameter` for parameterized sweeps; MOSEK uses its Fusion API directly
3. `optimize()` loops over discrete parameter values `g_vals`, solves each, and `DataSaver` appends results to CSV

### Key Design Patterns

- **Abstract `Interface` base class**: defines properties (`target`, `psd`, `lc`, `eps`) with abstract getters/setters. Both `CvxpyInterface` and `MosekInterface` implement these to translate the JSON representation into solver-specific formulations.
- **Parameterized linear constraints**: form `(A + g*Ag) @ x == b + g*bg`, where `g` sweeps over discrete values. CVXPY uses `cp.Parameter` for this; MOSEK rebuilds the problem each iteration.
- **PSD matrix vectorization**: CVXPY uses `cp.vec_to_upper_tri()` with diagonal elements halved; MOSEK uses `sVec` format with sqrt(2) scaling for off-diagonals.

### Mathematica Side (`source/`)

`source/ToCVXPY.wl` is a Mathematica package that exports SDP problems as JSON. Users call `GenerateTask[target, allVars, sdpMatrix, loopEquations, para, lambda, eps, name]`. Examples are in `source/example/`.

## Known Issues

- CVXPY's `eps` parameter doesn't reliably control MOSEK tolerance — cvxpy may report `OPTIMAL` with large constraint violations. The CVXPY interface works around this by passing all MOSEK tolerance parameters directly. See [cvxpy#434](https://github.com/cvxpy/cvxpy/issues/434).
- `numpy` is pinned to `<2.0` due to compatibility constraints with MOSEK/cvxpy.

## Instructions

- If anything is unclear or ambiguous, ask the user for clarification rather than guessing.

## Dependencies

- `cvxpy>=1.5.3`, `mosek>=10.2.5` (requires MOSEK license), `numpy<2.0`, `pandas>=2.2.3`, `loguru>=0.7.2`
- Build: `setuptools` + `wheel`
