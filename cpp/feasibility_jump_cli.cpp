/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/version_config.hpp>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <math_optimization/solution_reader.hpp>
#include <math_optimization/solution_writer.hpp>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/problem/problem.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <argparse/argparse.hpp>
#include <thrust/fill.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string normalize_token(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  std::replace(value.begin(), value.end(), '_', '-');
  return value;
}

cuopt::linear_programming::detail::fj_mode_t parse_fj_mode(const std::string& value)
{
  const auto token = normalize_token(value);
  if (token == "first-feasible") { return cuopt::linear_programming::detail::fj_mode_t::FIRST_FEASIBLE; }
  if (token == "greedy-descent") { return cuopt::linear_programming::detail::fj_mode_t::GREEDY_DESCENT; }
  if (token == "tree") { return cuopt::linear_programming::detail::fj_mode_t::TREE; }
  if (token == "rounding") { return cuopt::linear_programming::detail::fj_mode_t::ROUNDING; }
  if (token == "exit-non-improving") {
    return cuopt::linear_programming::detail::fj_mode_t::EXIT_NON_IMPROVING;
  }
  throw std::invalid_argument("Invalid --mode value: " + value);
}

cuopt::linear_programming::detail::fj_candidate_selection_t parse_candidate_selection(
  const std::string& value)
{
  const auto token = normalize_token(value);
  if (token == "weighted-score") {
    return cuopt::linear_programming::detail::fj_candidate_selection_t::WEIGHTED_SCORE;
  }
  if (token == "feasible-first") {
    return cuopt::linear_programming::detail::fj_candidate_selection_t::FEASIBLE_FIRST;
  }
  throw std::invalid_argument("Invalid --candidate-selection value: " + value);
}

cuopt::linear_programming::detail::fj_load_balancing_mode_t parse_load_balancing_mode(
  const std::string& value)
{
  const auto token = normalize_token(value);
  if (token == "always-on") {
    return cuopt::linear_programming::detail::fj_load_balancing_mode_t::ALWAYS_ON;
  }
  if (token == "auto") { return cuopt::linear_programming::detail::fj_load_balancing_mode_t::AUTO; }
  if (token == "always-off") {
    return cuopt::linear_programming::detail::fj_load_balancing_mode_t::ALWAYS_OFF;
  }
  throw std::invalid_argument("Invalid --load-balancing-mode value: " + value);
}

}  // namespace

int main(int argc, char* argv[])
{
  const std::string version_string = std::string("cuOpt ") + std::to_string(CUOPT_VERSION_MAJOR) +
                                     "." + std::to_string(CUOPT_VERSION_MINOR) + "." +
                                     std::to_string(CUOPT_VERSION_PATCH);
  argparse::ArgumentParser program("feasibility_jump_cli", version_string);

  program.add_argument("filename").help("input mps file").nargs(1).required();

  program.add_argument("--initial-solution")
    .help("path to the initial solution .sol file")
    .default_value("");

  program.add_argument("--output-solution")
    .help("path to write the resulting solution (.sol)")
    .default_value("");

  program.add_argument("--mode")
    .help("feasibility jump mode: first-feasible | greedy-descent | tree | rounding | "
          "exit-non-improving")
    .default_value(std::string("first-feasible"));

  program.add_argument("--candidate-selection")
    .help("candidate selection: weighted-score | feasible-first")
    .default_value(std::string("weighted-score"));

  program.add_argument("--load-balancing-mode")
    .help("load balancing: auto | always-on | always-off")
    .default_value(std::string("auto"));

  program.add_argument("--time-limit")
    .help("time limit in seconds")
    .default_value(60.0);

  program.add_argument("--iteration-limit")
    .help("maximum number of iterations")
    .default_value(std::numeric_limits<int>::max());

  program.add_argument("--seed").help("random seed").default_value(0);

  program.add_argument("--feasibility-run")
    .help("enable feasibility-first behavior")
    .default_value(true)
    .implicit_value(true);

  program.add_argument("--no-feasibility-run")
    .help("disable feasibility-first behavior")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--update-weights")
    .help("enable constraint weight updates")
    .default_value(true)
    .implicit_value(true);

  program.add_argument("--no-update-weights")
    .help("disable constraint weight updates")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--mip-scaling")
    .help("apply MIP scaling before running feasibility jump")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--objective-weight")
    .help("initial objective weight")
    .default_value(0.0);

  program.add_argument("--initial-weight")
    .help("initial constraint weight value")
    .default_value(1.0);

  program.add_argument("--randomize-weights")
    .help("randomize constraint weights before running")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--n-of-minimums-for-exit")
    .help("number of local minimums before exit")
    .default_value(7000);

  program.add_argument("--infeasibility-weight")
    .help("infeasibility weight")
    .default_value(1.0);

  program.add_argument("--baseline-objective-for-longer-run")
    .help("baseline objective threshold for longer runs")
    .default_value(std::numeric_limits<double>::lowest());

  program.add_argument("--max-sampled-moves")
    .help("maximum number of sampled moves")
    .default_value(raft::WarpSize * 16);

  program.add_argument("--random-var-probability")
    .help("probability of choosing a random positive-score variable")
    .default_value(0.04);

  program.add_argument("--random-cstr-probability")
    .help("probability of choosing a random constraint variable")
    .default_value(0.16);

  program.add_argument("--global-move-update-period")
    .help("global move update period")
    .default_value(10);

  program.add_argument("--heavy-move-update-period")
    .help("heavy move update period")
    .default_value(50);

  program.add_argument("--sync-period").help("synchronization period").default_value(200);

  program.add_argument("--lhs-refresh-period")
    .help("lhs refresh period")
    .default_value(500);

  program.add_argument("--allow-infeasibility-iterations")
    .help("iterations allowed for infeasibility")
    .default_value(200);

  program.add_argument("--objective-weight-increment")
    .help("objective weight increment")
    .default_value(0.01);

  program.add_argument("--load-balancing-variable-threshold")
    .help("load balancing variable threshold")
    .default_value(300);

  program.add_argument("--load-balancing-constraint-threshold")
    .help("load balancing constraint threshold")
    .default_value(5000);

  program.add_argument("--load-balancing-variable-split-size")
    .help("load balancing variable split size")
    .default_value(50);

  program.add_argument("--breakthrough-move-epsilon")
    .help("breakthrough move epsilon")
    .default_value(1e-4);

  program.add_argument("--tabu-tenure-min").help("tabu tenure min").default_value(3);

  program.add_argument("--tabu-tenure-max").help("tabu tenure max").default_value(13);

  program.add_argument("--excess-improvement-weight")
    .help("excess improvement weight")
    .default_value(0.5);

  program.add_argument("--weight-smoothing-probability")
    .help("weight smoothing probability")
    .default_value(0.0003);

  program.add_argument("--fractional-score-multiplier")
    .help("fractional score multiplier")
    .default_value(100.0);

  program.add_argument("--rounding-second-stage-split")
    .help("rounding second stage split")
    .default_value(0.1);

  program.add_argument("--small-move-tabu-threshold")
    .help("small move tabu threshold")
    .default_value(1e-6);

  program.add_argument("--small-move-tabu-tenure")
    .help("small move tabu tenure")
    .default_value(4);

  program.add_argument("--old-codepath-total-var-to-relvar-ratio-threshold")
    .help("legacy load balancing ratio threshold")
    .default_value(200);

  program.add_argument("--load-balancing-codepath-min-varcount")
    .help("load balancing codepath minimum variable count")
    .default_value(3200);

  program.add_argument("--device")
    .help("CUDA device id to use (default: current device)")
    .default_value(-1);

  program.add_argument("--log-file")
    .help("path to write log output")
    .default_value(std::string(""));

  program.add_argument("--log-to-console")
    .help("enable log output to console")
    .default_value(true)
    .implicit_value(true);

  program.add_argument("--no-log-to-console")
    .help("disable log output to console")
    .default_value(false)
    .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const auto device_id = program.get<int>("--device");
  std::optional<raft::device_setter> device_guard;
  if (device_id >= 0) { device_guard.emplace(device_id); }

  const auto log_file       = program.get<std::string>("--log-file");
  bool log_to_console       = program.get<bool>("--log-to-console");
  const bool disable_console = program.get<bool>("--no-log-to-console");
  if (disable_console) { log_to_console = false; }
  cuopt::init_logger_t logger(log_file, log_to_console);

  const std::string file_name            = program.get<std::string>("filename");
  const std::string initial_solution_file = program.get<std::string>("--initial-solution");
  const std::string output_solution_file  = program.get<std::string>("--output-solution");

  cuopt::linear_programming::detail::fj_settings_t fj_settings;
  try {
    fj_settings.mode = parse_fj_mode(program.get<std::string>("--mode"));
    fj_settings.candidate_selection =
      parse_candidate_selection(program.get<std::string>("--candidate-selection"));
    fj_settings.load_balancing_mode =
      parse_load_balancing_mode(program.get<std::string>("--load-balancing-mode"));
  } catch (const std::invalid_argument& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  fj_settings.time_limit      = program.get<double>("--time-limit");
  fj_settings.iteration_limit = program.get<int>("--iteration-limit");
  fj_settings.seed            = program.get<int>("--seed");
  fj_settings.n_of_minimums_for_exit = program.get<int>("--n-of-minimums-for-exit");
  fj_settings.infeasibility_weight   = program.get<double>("--infeasibility-weight");
  fj_settings.baseline_objective_for_longer_run =
    program.get<double>("--baseline-objective-for-longer-run");

  bool feasibility_run = program.get<bool>("--feasibility-run");
  if (program.get<bool>("--no-feasibility-run")) { feasibility_run = false; }
  fj_settings.feasibility_run = feasibility_run;

  bool update_weights = program.get<bool>("--update-weights");
  if (program.get<bool>("--no-update-weights")) { update_weights = false; }
  fj_settings.update_weights = update_weights;

  fj_settings.parameters.max_sampled_moves = program.get<int>("--max-sampled-moves");
  fj_settings.parameters.random_var_probability =
    program.get<double>("--random-var-probability");
  fj_settings.parameters.random_cstr_probability =
    program.get<double>("--random-cstr-probability");
  fj_settings.parameters.global_move_update_period =
    program.get<int>("--global-move-update-period");
  fj_settings.parameters.heavy_move_update_period =
    program.get<int>("--heavy-move-update-period");
  fj_settings.parameters.sync_period = program.get<int>("--sync-period");
  fj_settings.parameters.lhs_refresh_period = program.get<int>("--lhs-refresh-period");
  fj_settings.parameters.allow_infeasibility_iterations =
    program.get<int>("--allow-infeasibility-iterations");
  fj_settings.parameters.objective_weight_increment =
    program.get<double>("--objective-weight-increment");
  fj_settings.parameters.load_balancing_variable_threshold =
    program.get<int>("--load-balancing-variable-threshold");
  fj_settings.parameters.load_balancing_constraint_threshold =
    program.get<int>("--load-balancing-constraint-threshold");
  fj_settings.parameters.load_balancing_variable_split_size =
    program.get<int>("--load-balancing-variable-split-size");
  fj_settings.parameters.breakthrough_move_epsilon =
    program.get<double>("--breakthrough-move-epsilon");
  fj_settings.parameters.tabu_tenure_min = program.get<int>("--tabu-tenure-min");
  fj_settings.parameters.tabu_tenure_max = program.get<int>("--tabu-tenure-max");
  fj_settings.parameters.excess_improvement_weight =
    program.get<double>("--excess-improvement-weight");
  fj_settings.parameters.weight_smoothing_probability =
    program.get<double>("--weight-smoothing-probability");
  fj_settings.parameters.fractional_score_multiplier =
    program.get<double>("--fractional-score-multiplier");
  fj_settings.parameters.rounding_second_stage_split =
    program.get<double>("--rounding-second-stage-split");
  fj_settings.parameters.small_move_tabu_threshold =
    program.get<double>("--small-move-tabu-threshold");
  fj_settings.parameters.small_move_tabu_tenure = program.get<int>("--small-move-tabu-tenure");
  fj_settings.parameters.old_codepath_total_var_to_relvar_ratio_threshold =
    program.get<int>("--old-codepath-total-var-to-relvar-ratio-threshold");
  fj_settings.parameters.load_balancing_codepath_min_varcount =
    program.get<int>("--load-balancing-codepath-min-varcount");

  const bool mip_scaling = program.get<bool>("--mip-scaling");

  const auto objective_weight = program.get<double>("--objective-weight");
  const auto initial_weight   = program.get<double>("--initial-weight");
  const auto randomize_weights = program.get<bool>("--randomize-weights");

  const raft::handle_t handle_{};

  constexpr bool input_mps_strict = false;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model;
  try {
    CUOPT_LOG_INFO("Reading file %s", file_name.c_str());
    mps_data_model = cuopt::mps_parser::parse_mps<int, double>(file_name, input_mps_strict);
  } catch (const std::exception& err) {
    CUOPT_LOG_ERROR("MPS parser exception: %s", err.what());
    return 1;
  }

  auto op_problem =
    cuopt::linear_programming::mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_checking_t<int, double>::check_problem_representation(
    op_problem);

  cuopt::linear_programming::detail::problem_t<int, double> original_problem(op_problem);
  cuopt::linear_programming::detail::problem_t<int, double> scaled_problem(original_problem);

  auto hyper_params = cuopt::linear_programming::pdlp_hyper_params::pdlp_hyper_params_t{};
  cuopt::linear_programming::detail::pdlp_initial_scaling_strategy_t<int, double> scaling(
    scaled_problem.handle_ptr,
    scaled_problem,
    hyper_params.default_l_inf_ruiz_iterations,
    static_cast<double>(hyper_params.default_alpha_pock_chambolle_rescaling),
    scaled_problem.reverse_coefficients,
    scaled_problem.reverse_offsets,
    scaled_problem.reverse_constraints,
    nullptr,
    hyper_params,
    true);

  if (mip_scaling) { scaling.scale_problem(); }

  scaled_problem.preprocess_problem();
  cuopt::linear_programming::detail::trivial_presolve(scaled_problem);

  cuopt::linear_programming::mip_solver_settings_t<int, double> solver_settings;
  solver_settings.time_limit = fj_settings.time_limit;
  solver_settings.mip_scaling = mip_scaling;

  cuopt::timer_t timer(fj_settings.time_limit);
  cuopt::linear_programming::detail::mip_solver_t<int, double> solver(
    scaled_problem, solver_settings, scaling, timer);

  cuopt::linear_programming::detail::solution_t<int, double> solution(*solver.context.problem_ptr);

  std::vector<double> initial_solution;
  if (!initial_solution_file.empty()) {
    initial_solution =
      cuopt::linear_programming::solution_reader_t::get_variable_values_from_sol_file(
        initial_solution_file, mps_data_model.get_variable_names());
  }

  if (!initial_solution.empty()) {
    cuopt::expand_device_copy(
      solution.assignment, initial_solution, solution.handle_ptr->get_stream());
  } else {
    thrust::fill(solution.handle_ptr->get_thrust_policy(),
                 solution.assignment.begin(),
                 solution.assignment.end(),
                 0.0);
  }

  if (mip_scaling) { scaling.scale_primal(solution.assignment); }
  solution.clamp_within_bounds();

  cuopt::linear_programming::detail::fj_t<int, double> fj(solver.context, fj_settings);
  if (randomize_weights) {
    fj.randomize_weights(solution.handle_ptr);
  } else {
    fj.reset_weights(solution.handle_ptr->get_stream(), initial_weight);
  }
  if (objective_weight != 0.0) {
    fj.objective_weight.set_value_async(objective_weight, solution.handle_ptr->get_stream());
  }
  solution.handle_ptr->sync_stream();

  fj.solve(solution);

  if (mip_scaling) {
    scaling.unscale_solutions(solution);
    solution.problem_ptr = &original_problem;
  }
  solution.compute_objective();
  solution.compute_feasibility();

  CUOPT_LOG_INFO("Feasibility jump finished. Feasible=%d Objective=%g",
                 solution.get_feasible(),
                 solution.get_user_objective());

  if (!output_solution_file.empty()) {
    const auto host_assignment = solution.get_host_assignment();
    cuopt::linear_programming::solution_writer_t::write_solution_to_sol_file(
      output_solution_file,
      solution.get_feasible() ? "feasible" : "infeasible",
      solution.get_user_objective(),
      mps_data_model.get_variable_names(),
      host_assignment);
  }

  return 0;
}
