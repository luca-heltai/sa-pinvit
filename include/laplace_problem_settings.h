#ifndef laplace_problem_settings_h
#define laplace_problem_settings_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/types.h>

using namespace dealii;

template <int dim>
class LaplaceProblemSettings : public ParameterAcceptor
{
public:
  LaplaceProblemSettings();

  std::string problem_type = "source";

  double       smoother_dampen         = 1.0;
  unsigned int smoother_steps          = 1;
  unsigned int n_steps                 = 7;
  unsigned int degree                  = 2;
  bool         write_high_order_output = true;

  unsigned int initial_refinement = 1;
  std::string  output_directory   = "";

  unsigned int pinvit_intermediate_max_iterations      = 100;
  double       pinvit_intermediate_tolerance           = 1e-6;
  unsigned int pinvit_initial_and_final_max_iterations = 0;
  double       pinvit_initial_and_final_tolerance      = 0;
  unsigned int number_of_eigenvalues                   = 1;

  //! By default, we create a hyper_L without colorization, and we use
  // homogeneous Dirichlet boundary conditions. In this set we store the
  // boundary ids to use when setting the boundary conditions:
  std::set<types::boundary_id> homogeneous_dirichlet_ids{0};

  std::string name_of_grid       = "hyper_L";
  std::string arguments_for_grid = "-1.: 1.: false";

  std::string refinement_strategy = "fixed_number";

  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> coefficient;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs;
};

#endif