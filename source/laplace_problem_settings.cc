#include "laplace_problem.h"

using namespace dealii;

template <int dim>
LaplaceProblemSettings<dim>::LaplaceProblemSettings()
  : ParameterAcceptor("/")
  , exact("Exact solution")
  , coefficient("Coefficient")
  , rhs("Forcing term")
{
  add_parameter("Problem type",
                problem_type,
                "Source problem or Eigenvalues problem (source|eigenvalues).",
                this->prm,
                Patterns::Selection("source|eigenvalues"));
  add_parameter("n_steps", n_steps, "Number of adaptive refinement steps.");
  add_parameter("smoother dampen",
                smoother_dampen,
                "Dampen factor for the smoother.");
  add_parameter("smoother steps", smoother_steps, "Number of smoother steps.");
  add_parameter("degree", degree, "Degree of the finite element space.");
  add_parameter(
    "Output directory",
    output_directory,
    "Directory where we want to save output files. Leave empty for no output.");

  enter_my_subsection(this->prm);
  this->prm.enter_subsection("Grid parameters");
  {
    this->prm.add_parameter("Grid generator", name_of_grid);
    this->prm.add_parameter("Grid generator arguments", arguments_for_grid);
    this->prm.add_parameter("Initial refinement",
                            initial_refinement,
                            "Initial refinement of the triangulation.");
    this->prm.add_parameter("Refinement strategy",
                            refinement_strategy,
                            "",
                            Patterns::Selection(
                              "fixed_fraction|fixed_number|global"));


    add_parameter(
      "Homogeneous Dirichlet boundary ids",
      homogeneous_dirichlet_ids,
      "Boundary Ids over which homogeneous Dirichlet boundary conditions are applied");
  }
  this->prm.leave_subsection();
  leave_my_subsection(this->prm);

  coefficient.declare_parameters_call_back.connect(
    [&]() { this->prm.set("Function expression", "1"); });
  rhs.declare_parameters_call_back.connect(
    [&]() { this->prm.set("Function expression", "1"); });
}

template class LaplaceProblemSettings<2>;
template class LaplaceProblemSettings<3>;