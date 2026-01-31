# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Feasibility Jump (FJ) runner wrapper.

Runs the run_fj executable via subprocess with a given MPS path, optional
initial solution path, and FJ mode. Returns a result object with feasible,
objective, solution_path, and time_seconds.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# Supported FJ modes (map to run_fj --mode)
FEASIBILITY_JUMP_MODES = ("first_feasible", "greedy_descent", "exit_non_improving")


@dataclass
class FeasibilityJumpResult:
    """Result of a standalone Feasibility Jump run."""

    feasible: bool
    """Whether a feasible solution was found."""
    objective: float
    """Objective value (NaN if infeasible)."""
    solution_path: Optional[str]
    """Path to written solution file, or None if not written."""
    time_seconds: float
    """Wall-clock time of the FJ run in seconds."""


def _find_run_fj_binary(run_fj_binary: Optional[str] = None) -> Optional[str]:
    """Resolve path to run_fj executable."""
    if run_fj_binary is not None:
        if os.path.isfile(run_fj_binary) and os.access(run_fj_binary, os.X_OK):
            return run_fj_binary
        return None
    # Check same prefix as Python (e.g. conda env bin)
    prefix_bin = os.path.join(sys.prefix, "bin", "run_fj")
    if os.path.isfile(prefix_bin) and os.access(prefix_bin, os.X_OK):
        return prefix_bin
    # Check PATH
    return shutil.which("run_fj")


def _parse_run_fj_stdout(stdout: str) -> FeasibilityJumpResult:
    """Parse run_fj stdout for FJ_OBJECTIVE=, FJ_FEASIBLE=, FJ_TIME=, FJ_SOLUTION_PATH=."""
    objective = float("nan")
    feasible = False
    time_seconds = 0.0
    solution_path = None

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("FJ_OBJECTIVE="):
            try:
                objective = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("FJ_FEASIBLE="):
            feasible = line.split("=", 1)[1].strip() == "1"
        elif line.startswith("FJ_TIME="):
            try:
                time_seconds = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("FJ_SOLUTION_PATH="):
            solution_path = line.split("=", 1)[1].strip()

    return FeasibilityJumpResult(
        feasible=feasible,
        objective=objective,
        solution_path=solution_path,
        time_seconds=time_seconds,
    )


def run_feasibility_jump(
    mps_path: str,
    initial_solution_path: Optional[str] = None,
    mode: str = "first_feasible",
    time_limit: float = 60.0,
    iteration_limit: Optional[int] = None,
    n_minimums_for_exit: Optional[int] = None,
    sol_out_path: Optional[str] = None,
    run_fj_binary: Optional[str] = None,
    seed: int = 0,
    presolve: bool = True,
    log_to_console: bool = True,
    **kwargs,
) -> FeasibilityJumpResult:
    """
    Run standalone Feasibility Jump on an MPS instance.

    Requires the run_fj executable (built with BUILD_MIP_BENCHMARKS=ON).
    Resolves run_fj from run_fj_binary, sys.prefix/bin/run_fj, or PATH.

    Parameters
    ----------
    mps_path : str
        Path to the MPS file (required).
    initial_solution_path : str, optional
        Path to initial solution file (.sol) or directory (same semantics as run_fj).
    mode : str, optional
        FJ mode: "first_feasible", "greedy_descent", or "exit_non_improving".
        Default "first_feasible".
    time_limit : float, optional
        Time limit in seconds. Default 60.0.
    iteration_limit : int, optional
        Maximum FJ iterations. Default no limit.
    n_minimums_for_exit : int, optional
        For exit_non_improving mode: number of local minimums without improvement to exit.
    sol_out_path : str, optional
        Path to write the output solution (.sol).
    run_fj_binary : str, optional
        Path to run_fj executable. If None, searched in sys.prefix/bin and PATH.
    seed : int, optional
        Random seed. Default 0.
    presolve : bool, optional
        Whether to run presolve. Default True.
    log_to_console : bool, optional
        Whether run_fj logs to console. Default True.
    **kwargs
        Ignored (for future options).

    Returns
    -------
    FeasibilityJumpResult
        Result with feasible, objective, solution_path, time_seconds.

    Raises
    ------
    FileNotFoundError
        If run_fj executable is not found.
    RuntimeError
        If run_fj fails (non-zero exit and no parseable output).
    """
    if mode not in FEASIBILITY_JUMP_MODES:
        raise ValueError(
            f"mode must be one of {FEASIBILITY_JUMP_MODES}, got {mode!r}"
        )

    binary = _find_run_fj_binary(run_fj_binary)
    if binary is None:
        raise FileNotFoundError(
            "run_fj executable not found. Build with BUILD_MIP_BENCHMARKS=ON "
            "or set run_fj_binary to the path of run_fj."
        )

    argv = [binary, mps_path, "--mode", mode, "--time-limit", str(time_limit), "--seed", str(seed)]
    if initial_solution_path is not None:
        argv.extend(["--initial-solution-path", initial_solution_path])
    if sol_out_path is not None:
        argv.extend(["--sol-out", sol_out_path])
    if iteration_limit is not None:
        argv.extend(["--iteration-limit", str(iteration_limit)])
    if n_minimums_for_exit is not None:
        argv.extend(["--n-minimums-for-exit", str(n_minimums_for_exit)])
    argv.append("--presolve")
    argv.append("t" if presolve else "f")
    argv.append("--log-to-console")
    argv.append("t" if log_to_console else "f")

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=time_limit + 60,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"run_fj timed out: {e}") from e

    out = result.stdout or ""
    parsed = _parse_run_fj_stdout(out)
    if result.returncode != 0 and result.returncode != 1:
        err = result.stderr or ""
        raise RuntimeError(
            f"run_fj failed with exit code {result.returncode}. stderr: {err[:500]}"
        )
    return parsed
