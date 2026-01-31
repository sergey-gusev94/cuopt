# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def _cli_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "feasibility_jump_cli")


def run(args: list[str]) -> int:
    """
    Run feasibility_jump_cli with the provided arguments.
    """
    return subprocess.call([_cli_path(), *args])


def main() -> None:
    """
    This connects to feasibility_jump_cli binary which is located under libcuopt/bin.
    """
    sys.exit(run(sys.argv[1:]))
