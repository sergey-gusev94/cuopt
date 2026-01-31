# Standalone Feasibility Jump Runner

This document describes how to run **only** the Feasibility Jump (FJ) heuristic on an MPS instance, with full control over mode, time/iteration limits, and optional initial solution. Use cases include: finding a first feasible solution quickly, running greedy descent from a given point, or benchmarking FJ in isolation.

## What is the standalone FJ runner?

The standalone FJ runner is the `run_fj` executable. It:

- Reads an MPS file (required).
- Optionally reads an initial solution from a `.sol` file or directory (same format as `solve_MIP`).
- Runs only the Feasibility Jump heuristic with the chosen mode and limits.
- Writes the result (objective, feasibility, optional `.sol` file) and prints machine-parseable lines (`FJ_OBJECTIVE=`, `FJ_FEASIBLE=`, `FJ_TIME=`, `FJ_SOLUTION_PATH=`).

Modes:

- **first_feasible** – Stop as soon as a feasible solution is found.
- **greedy_descent** – Single descent from the current point until no improving jump (local minimum or time/iteration limit).
- **exit_non_improving** – Run until no improvement for a number of local minimums (`--n-minimums-for-exit`).

## Build vs Conda

### Using Conda (prebuilt cuOpt)

- Use the existing conda environment (see [conda/README.md](../../conda/README.md)).
- The **run_fj** binary is only available if the conda package was built with MIP benchmarks enabled (`BUILD_MIP_BENCHMARKS=ON`). If your installed package does not ship `run_fj`, use “Build from source” below to obtain it.

### Build from source (when you need run_fj or modified FJ)

1. Build libcuopt with MIP and benchmarks enabled:
   ```bash
   ./build.sh libcuopt --cmake-args="-DBUILD_MIP_BENCHMARKS=ON"
   ```
   Do **not** use `--build-lp-only` so that MIP (and FJ) are built.

2. The `run_fj` executable is produced in the libcuopt build directory (e.g. `cpp/build/run_fj` when building from the repo root with default `LIBCUOPT_BUILD_DIR`).

3. Run it with the correct `LD_LIBRARY_PATH` if needed (e.g. point to the directory containing `libcuopt.so`). You can keep using the Python/cuopt package from conda for other APIs while using this binary.

## Running with a provided solution

- Use **--initial-solution-path** with a path to a single `.sol` file or to a directory.
- **Single file:** Path to a `.sol` file (variable names and values, one per line; same format as cuOpt solution output).
- **Directory:** A directory named after the MPS base name (e.g. `instance/` for `instance.mps`) containing one or more `.sol` files; the first valid solution found is used.
- Variable names in the solution file must match the variable names in the MPS (same order is not required; names are matched).
- The assignment is loaded, clamped to variable bounds, and FJ is run from that point.
- When an initial solution is provided, presolve is typically disabled (same behavior as `solve_MIP` with initial solutions).

## Terminate at first feasible solution

- Set **--mode first_feasible** (default).
- The runner exits as soon as the first feasible solution is found.
- If **--sol-out** is set, that solution is written to the given path.
- Example:
  ```bash
  run_fj instance.mps --mode first_feasible --time-limit 60 --sol-out out.sol
  ```

## Run GREEDY_DESCENT from an initial point

- Set **--mode greedy_descent** and optionally **--initial-solution-path** to start from that solution.
- Without **--initial-solution-path**, FJ starts from zero (or default assignment).
- FJ performs a single descent until a local minimum or time/iteration limit.
- Example:
  ```bash
  run_fj instance.mps --mode greedy_descent --initial-solution-path my.sol --time-limit 120 --sol-out fj_out.sol
  ```

## All FJ flags (CLI)

| CLI flag | Description | Default |
|----------|-------------|---------|
| `mps_path` | Path to MPS file (positional) | required |
| `--mode` | `first_feasible`, `greedy_descent`, `exit_non_improving` | `first_feasible` |
| `--time-limit` | Time limit in seconds | 60 |
| `--iteration-limit` | Maximum FJ iterations | no limit |
| `--n-minimums-for-exit` | Local minimums without improvement to exit (for `exit_non_improving`) | 7000 |
| `--initial-solution-path` | Path to initial solution file or directory | none |
| `--sol-out` | Path to write output solution (.sol) | none |
| `--seed` | Random seed | 0 |
| `--presolve` | Run presolve (t/f) | t |
| `--log-to-console` | Log to console (t/f) | t |

Advanced tuning (e.g. FJ hyper-parameters like `max_sampled_moves`, `random_var_probability`, tabu/load-balancing) is not exposed via CLI; modify defaults in `run_fj.cpp` and recompile if needed.

## Examples (copy-paste)

First feasible with time limit and solution output:

```bash
run_fj instance.mps --mode first_feasible --time-limit 60 --sol-out out.sol
```

Greedy descent from a solution file:

```bash
run_fj instance.mps --mode greedy_descent --initial-solution-path my.sol --time-limit 120 --sol-out fj_out.sol
```

EXIT_NON_IMPROVING with iteration limit:

```bash
run_fj instance.mps --mode exit_non_improving --iteration-limit 100000 --n-minimums-for-exit 5 --time-limit 300
```

## Feasibility jump (Python)

You can drive `run_fj` from Python using the `run_feasibility_jump()` wrapper in `cuopt.linear_programming`. See the example script `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/feasibility_jump_example.py` (or `python/cuopt/cuopt/linear_programming/examples/run_fj_with_solution.py`). Main arguments:

- **mps_path** – Path to MPS file.
- **initial_solution_path** – Optional path to a `.sol` file (solution “in a certain place”).
- **mode** – `"first_feasible"`, `"greedy_descent"`, or `"exit_non_improving"`.
- **time_limit** – Time limit in seconds.
- **sol_out_path** – Optional path to write the output solution.

The wrapper runs the `run_fj` binary, parses stdout, and returns a result object with `feasible`, `objective`, `solution_path`, and `time_seconds`.

## Note on CPU FJ

The standalone runner uses the GPU FJ implementation only. A CPU FJ path (`fj_cpu_climber_t` / `cpu_solve`) exists in the library but is not exposed by the `run_fj` executable.
