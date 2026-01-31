=====================
Feasibility Jump CLI
=====================

The ``feasibility_jump_cli`` binary runs the feasibility jump (FJ) heuristic directly on an MPS
model, without invoking the full MILP solve loop. It exposes all FJ settings and hyperparameters
as command-line flags, so you can experiment with different FJ modes, candidate-selection
strategies, and load-balancing behaviors.

This CLI is intended for:

- Running FJ from a provided initial solution and stopping on the first feasible solution.
- Running ``GREEDY_DESCENT`` from an initial point to locally improve an assignment.
- Tuning FJ hyperparameters and logging outcomes for debugging or research.

Build from a Conda Environment
==============================

If you already have a conda environment with the current cuOpt version installed and want to
compile the FJ CLI (or rebuild after changes), you can build it from source inside that env.
The build script will use ``$CONDA_PREFIX`` as the installation prefix.

1. Activate your conda environment:

   .. code-block:: bash

      conda activate <your-cuopt-env>

2. Build the C++ artifacts (including ``feasibility_jump_cli``):

   .. code-block:: bash

      ./build.sh libcuopt -n

   The binary will be placed under ``cpp/build/feasibility_jump_cli``. The ``-n`` flag skips the
   install step; omit it if you want the binary installed into ``$CONDA_PREFIX/bin``.

3. (Optional) Rebuild after changes:

   .. code-block:: bash

      ./build.sh libcuopt -n

If you need to use a specific CUDA compiler or toolchain, set ``CUDACXX`` before running
``build.sh``. For additional build options, see ``./build.sh --help``.

Quick Start
===========

Run FJ and stop at the first feasible solution (using an initial solution from a ``.sol`` file):

.. code-block:: bash

   feasibility_jump_cli problem.mps \
     --initial-solution initial.sol \
     --mode first-feasible \
     --time-limit 60 \
     --output-solution fj_first_feasible.sol

Run FJ in ``GREEDY_DESCENT`` mode from an initial solution:

.. code-block:: bash

   feasibility_jump_cli problem.mps \
     --initial-solution initial.sol \
     --mode greedy-descent \
     --iteration-limit 5000

Run FJ in rounding mode (useful when starting from an LP relaxation solution):

.. code-block:: bash

   feasibility_jump_cli problem.mps \
     --initial-solution lp_solution.sol \
     --mode rounding

Python Wrapper Example
======================

The ``libcuopt`` wheel installs a small Python wrapper that forwards arguments to the
``feasibility_jump_cli`` binary. You can call it directly as a script entry point or use the
helper in Python code.

CLI entry point from Python:

.. code-block:: bash

   python -m libcuopt.feasibility_jump_cli problem.mps \
     --initial-solution initial.sol \
     --mode first-feasible \
     --candidate-selection feasible-first \
     --time-limit 30 \
     --iteration-limit 5000 \
     --output-solution fj_solution.sol

Python helper example:

.. code-block:: python

   from libcuopt.feasibility_jump_cli import run

   args = [
       "problem.mps",
       "--initial-solution",
       "initial.sol",
       "--mode",
       "first-feasible",
       "--candidate-selection",
       "feasible-first",
       "--time-limit",
       "30",
       "--iteration-limit",
       "5000",
       "--output-solution",
       "fj_solution.sol",
   ]
   run(args)

See ``python/libcuopt/libcuopt/examples/feasibility_jump_example.py`` for a ready-to-run example.

Flags Reference
===============

General FJ settings
-------------------

- ``--mode``: FJ mode. Valid values:
  ``first-feasible``, ``greedy-descent``, ``tree``, ``rounding``, ``exit-non-improving``.
- ``--candidate-selection``: ``weighted-score`` or ``feasible-first``.
- ``--load-balancing-mode``: ``auto``, ``always-on``, ``always-off``.
- ``--time-limit``: wall-clock limit (seconds).
- ``--iteration-limit``: maximum number of FJ iterations.
- ``--seed``: RNG seed.
- ``--feasibility-run`` / ``--no-feasibility-run``: toggle feasibility-first behavior.
- ``--update-weights`` / ``--no-update-weights``: enable/disable dynamic constraint weights.
- ``--n-of-minimums-for-exit``: number of local minimums before exit.
- ``--infeasibility-weight``: weight applied to constraint infeasibility in the score.
- ``--baseline-objective-for-longer-run``: baseline objective threshold for longer runs.
- ``--objective-weight``: initial objective weight scalar.
- ``--initial-weight``: initial constraint weight value when resetting weights.
- ``--randomize-weights``: randomize constraint weights before running.

Input/output and environment
----------------------------

- ``--initial-solution``: path to an initial ``.sol`` file to seed FJ.
- ``--output-solution``: write the resulting solution to a ``.sol`` file.
- ``--mip-scaling``: apply MIP scaling before running FJ (disabled by default).
- ``--device``: CUDA device id to use.
- ``--log-file``: write logs to a file.
- ``--log-to-console`` / ``--no-log-to-console``: toggle console logging.

Hyperparameters (``fj_hyper_parameters_t``)
------------------------------------------

- ``--max-sampled-moves``: maximum number of sampled moves.
- ``--random-var-probability``: probability of choosing a random positive-score variable.
- ``--random-cstr-probability``: probability of choosing a variable via a random constraint.
- ``--global-move-update-period``: global move update period.
- ``--heavy-move-update-period``: heavy move update period.
- ``--sync-period``: synchronization period.
- ``--lhs-refresh-period``: LHS refresh period.
- ``--allow-infeasibility-iterations``: iterations allowed for infeasibility.
- ``--objective-weight-increment``: increment applied to the objective weight when improving.
- ``--load-balancing-variable-threshold``: variable threshold for load balancing.
- ``--load-balancing-constraint-threshold``: constraint threshold for load balancing.
- ``--load-balancing-variable-split-size``: variable split size for load balancing.
- ``--breakthrough-move-epsilon``: epsilon for breakthrough move acceptance.
- ``--tabu-tenure-min`` / ``--tabu-tenure-max``: minimum/maximum tabu tenure.
- ``--excess-improvement-weight``: weight for excess improvement term.
- ``--weight-smoothing-probability``: smoothing probability for constraint weights.
- ``--fractional-score-multiplier``: multiplier for fractional variable scoring.
- ``--rounding-second-stage-split``: rounding second-stage split fraction.
- ``--small-move-tabu-threshold``: threshold for small-move tabu marking.
- ``--small-move-tabu-tenure``: tenure for small-move tabu marking.
- ``--old-codepath-total-var-to-relvar-ratio-threshold``: legacy ratio threshold for load
  balancing.
- ``--load-balancing-codepath-min-varcount``: minimum variable count to enable load balancing.

Notes
=====

- ``first-feasible`` mode returns as soon as a feasible solution is found. Combine it with an
  initial solution to prioritize quick feasibility.
- ``greedy-descent`` performs a single descent pass from the initial point until no improving
  jumps are available.
- If you enable ``--mip-scaling``, the CLI scales the problem and initial solution before running
  FJ, then unscales the final solution for output. Disable scaling to preserve the original scale
  throughout.
