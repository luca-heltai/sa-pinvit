#ifndef laplace_problem_settings_h
#define laplace_problem_settings_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/solver_control.h>

#include "inner_control.h"

using namespace dealii;

template <int dim>
class LaplaceProblemSettings : public ParameterAcceptor
{
public:
  LaplaceProblemSettings();

  std::string problem_type  = "source";
  std::string smoother_type = "gmg";

  double              gmg_smoother_dampen     = 1.0;
  unsigned int        gmg_smoother_steps      = 1;
  unsigned int        n_cycles                = 7;
  unsigned int        degree                  = 2;
  bool                write_high_order_output = true;
  std::vector<double> exact_eigenvalues =
    {2.0, 5.0, 5.0, 8.0, 10.0, 10.0, 13.0, 13.0, 17.0, 17.0};

  unsigned int initial_refinement = 1;
  std::string  output_directory   = "";

  unsigned int           number_of_eigenvalues = 1;
  std::set<unsigned int> eigen_estimators      = {0};


  /**
   * By default, we create a hyper_L without colorization, and we use
   * homogeneous Dirichlet boundary conditions. In this set we store the
   * boundary ids to use when setting the boundary conditions:
   */
  std::set<types::boundary_id> homogeneous_dirichlet_ids{0};

  std::string name_of_grid       = "hyper_cube";
  std::string arguments_for_grid = "0: 3.14159265359: false";

  std::string refinement_strategy = "fixed_number";

  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> coefficient;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs;

  mutable ParameterAcceptorProxy<ReductionControl>
                                               first_and_last_solver_control;
  mutable ParameterAcceptorProxy<InnerControl> intermediate_solver_control;

  mutable ParsedConvergenceTable error_table;
};

#endif