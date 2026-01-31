/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "initial_solution_reader.hpp"
#include "initial_problem_check.hpp"

#include <argparse/argparse.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>

#include <math_optimization/solution_writer.hpp>

#include <raft/core/handle.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/detail/cublas_wrappers.h>

#include <thrust/fill.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

static void init_handler(const raft::handle_t* handle_ptr)
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

static void read_single_solution_from_path(const std::string& path,
                                           const std::vector<std::string>& var_names,
                                           std::vector<double>& assignment)
{
  solution_reader_t reader;
  bool success = reader.read_from_sol(path);
  if (!success) {
    CUOPT_LOG_ERROR("Initial solution reading error!");
    return;
  }
  CUOPT_LOG_INFO("Success reading file %s Number of var vals %lu", path.c_str(), reader.data_map.size());
  assignment.clear();
  assignment.reserve(var_names.size());
  for (const auto& name : var_names) {
    auto it = reader.data_map.find(name);
    double val = (it != reader.data_map.end()) ? it->second : 0.0;
    assignment.push_back(val);
  }
}

static std::vector<double> read_initial_solution(const std::string& file_path,
                                                 const std::string& mps_file_name,
                                                 const std::vector<std::string>& var_names)
{
  std::vector<double> assignment;
  std::string mps_file_name_no_ext = mps_file_name.substr(0, mps_file_name.find_last_of("."));
  std::string initial_solution_dir = file_path + "/" + mps_file_name_no_ext;
  if (std::filesystem::exists(initial_solution_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(initial_solution_dir)) {
      read_single_solution_from_path(entry.path(), var_names, assignment);
      if (!assignment.empty()) break;
    }
  } else {
    read_single_solution_from_path(file_path, var_names, assignment);
  }
  return assignment;
}

}  // namespace cuopt::linear_programming::detail

int main(int argc, char* argv[])
{
  using namespace cuopt::linear_programming;
  using namespace cuopt::linear_programming::detail;

  argparse::ArgumentParser program("run_fj", "Standalone Feasibility Jump runner");
  program.add_argument("mps_path").help("Path to MPS file").required();
  program.add_argument("--mode")
    .help("FJ mode: first_feasible, greedy_descent, exit_non_improving")
    .default_value(std::string("first_feasible"));
  program.add_argument("--time-limit").help("Time limit (seconds)").scan<'g', double>().default_value(60.0);
  program.add_argument("--iteration-limit").help("Iteration limit").scan<'i', int>().default_value(std::numeric_limits<int>::max());
  program.add_argument("--n-minimums-for-exit")
    .help("Number of local minimums without improvement to exit (exit_non_improving mode)")
    .scan<'i', int>()
    .default_value(7000);
  program.add_argument("--initial-solution-path").help("Path to initial solution file or directory");
  program.add_argument("--sol-out").help("Path to write output solution (.sol)");
  program.add_argument("--seed").help("Random seed").scan<'i', int>().default_value(0);
  program.add_argument("--presolve").help("Run presolve (t/f)").default_value(std::string("t"));
  program.add_argument("--log-to-console").help("Log to console (t/f)").default_value(std::string("t"));

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  std::string mps_path             = program.get<std::string>("mps_path");
  std::string mode_str             = program.get<std::string>("--mode");
  double time_limit                = program.get<double>("--time-limit");
  int iteration_limit              = program.get<int>("--iteration-limit");
  int n_minimums_for_exit          = program.get<int>("--n-minimums-for-exit");
  std::string initial_solution_path;
  if (program.present("--initial-solution-path")) {
    initial_solution_path = program.get<std::string>("--initial-solution-path");
  }
  std::string sol_out;
  if (program.present("--sol-out")) { sol_out = program.get<std::string>("--sol-out"); }
  int seed = program.get<int>("--seed");
  bool presolve = (program.get<std::string>("--presolve") == "t" || program.get<std::string>("--presolve") == "true");
  bool log_to_console = (program.get<std::string>("--log-to-console") == "t" || program.get<std::string>("--log-to-console") == "true");

  cuopt::init_logger_t log("", log_to_console);

  const raft::handle_t handle_{};
  constexpr bool input_mps_strict = false;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model;
  try {
    mps_data_model = cuopt::mps_parser::parse_mps<int, double>(mps_path, input_mps_strict);
  } catch (const std::logic_error& e) {
    CUOPT_LOG_ERROR("MPS parser exception: %s", e.what());
    return -1;
  }

  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = time_limit;
  settings.tolerances.absolute_tolerance = 1e-6;
  settings.tolerances.relative_tolerance = 1e-12;
  settings.tolerances.integrality_tolerance = 1e-5;

  detail::problem_t<int, double> problem(op_problem, settings.get_tolerances());
  problem.preprocess_problem();

  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);

  auto timer = cuopt::timer_t(time_limit);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);

  detail::solution_t<int, double> solution(*solver.context.problem_ptr);

  std::vector<double> initial_assignment;
  if (!initial_solution_path.empty()) {
    std::string base_filename = mps_path.substr(mps_path.find_last_of("/\\") + 1);
    initial_assignment =
      detail::read_initial_solution(initial_solution_path, base_filename, mps_data_model.get_variable_names());
    if (!initial_assignment.empty()) {
      if (static_cast<size_t>(problem.n_variables) != initial_assignment.size()) {
        CUOPT_LOG_ERROR("Initial solution size %lu does not match problem variables %d",
                        initial_assignment.size(),
                        problem.n_variables);
        return -1;
      }
      cuopt::expand_device_copy(solution.assignment, initial_assignment, solution.handle_ptr->get_stream());
    }
  }
  if (initial_assignment.empty()) {
    thrust::fill(solution.handle_ptr->get_thrust_policy(),
                 solution.assignment.begin(),
                 solution.assignment.end(),
                 0.0);
  }
  solution.clamp_within_bounds();

  detail::fj_settings_t fj_settings;
  fj_settings.seed                 = seed;
  fj_settings.time_limit           = time_limit;
  fj_settings.iteration_limit      = iteration_limit;
  fj_settings.n_of_minimums_for_exit = n_minimums_for_exit;

  if (mode_str == "first_feasible") {
    fj_settings.mode = detail::fj_mode_t::FIRST_FEASIBLE;
  } else if (mode_str == "greedy_descent") {
    fj_settings.mode = detail::fj_mode_t::GREEDY_DESCENT;
  } else if (mode_str == "exit_non_improving") {
    fj_settings.mode = detail::fj_mode_t::EXIT_NON_IMPROVING;
  } else {
    CUOPT_LOG_ERROR("Unknown mode: %s (use first_feasible, greedy_descent, or exit_non_improving)", mode_str.c_str());
    return -1;
  }

  detail::fj_t<int, double> fj(solver.context, fj_settings);
  fj.reset_weights(solution.handle_ptr->get_stream(), 1.);
  solution.handle_ptr->sync_stream();

  auto start = std::chrono::high_resolution_clock::now();
  fj.solve(solution);
  solution.handle_ptr->sync_stream();
  auto end       = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(end - start).count();

  bool feasible = solution.compute_feasibility();
  double objective =
    feasible ? solution.get_user_objective() : std::numeric_limits<double>::quiet_NaN();

  std::cout << "FJ_OBJECTIVE=" << objective << std::endl;
  std::cout << "FJ_FEASIBLE=" << (feasible ? "1" : "0") << std::endl;
  std::cout << "FJ_TIME=" << elapsed << std::endl;

  if (!sol_out.empty()) {
    const std::vector<std::string>& var_names = op_problem.get_variable_names();
    std::vector<double> sol_host = cuopt::host_copy(solution.assignment, solution.handle_ptr->get_stream());
    std::string status = feasible ? "Feasible" : "Infeasible";
    solution_writer_t::write_solution_to_sol_file(sol_out, status, objective, var_names, sol_host);
    std::cout << "FJ_SOLUTION_PATH=" << sol_out << std::endl;
  }

  return feasible ? 0 : 1;
}
