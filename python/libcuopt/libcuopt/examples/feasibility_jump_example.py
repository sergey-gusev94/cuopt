# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libcuopt.feasibility_jump_cli import run


def main() -> int:
    args = [
        "problem.mps",
        "--initial-solution",
        "initial.sol",
        "--mode",
        "first-feasible",
        "--candidate-selection",
        "feasible-first",
        "--load-balancing-mode",
        "auto",
        "--time-limit",
        "30",
        "--iteration-limit",
        "5000",
        "--objective-weight",
        "0.01",
        "--output-solution",
        "fj_solution.sol",
    ]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
