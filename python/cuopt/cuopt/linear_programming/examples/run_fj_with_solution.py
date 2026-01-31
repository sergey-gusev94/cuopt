# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run Feasibility Jump from an existing solution with a chosen mode.

This example shows how to call the Python wrapper when you have a solution
file (e.g. .sol) in a known path and want to run a specific FJ mode:
first_feasible, greedy_descent, or exit_non_improving.

Requires the run_fj executable (build with BUILD_MIP_BENCHMARKS=ON).
"""

from cuopt import linear_programming as lp


def main():
    # Paths: MPS instance and initial solution (solution in a certain place)
    mps_path = "instance.mps"
    initial_solution_path = "/path/to/my_solution.sol"  # or None to start from zero
    mode = "greedy_descent"  # or "first_feasible", "exit_non_improving"
    time_limit = 60.0
    sol_out_path = "fj_output.sol"

    result = lp.run_feasibility_jump(
        mps_path,
        initial_solution_path=initial_solution_path,
        mode=mode,
        time_limit=time_limit,
        sol_out_path=sol_out_path,
    )

    print(f"Feasible: {result.feasible}, Objective: {result.objective}")
    print(f"Time (s): {result.time_seconds}")
    if result.solution_path:
        print(f"Solution written to: {result.solution_path}")


if __name__ == "__main__":
    main()
