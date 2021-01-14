#include "laplace_problem.h"

using namespace dealii;

template <int dim>
LaplaceProblemSettings<dim>::LaplaceProblemSettings()
  : ParameterAcceptor("/")
  , exact("Exact solution")
  , coefficient("Coefficient")
  , rhs("Forcing term")
  , first_and_last_solver_control("/Solver/First and last cycle")
  , intermediate_solver_control("/Solver/Intermediate cycles")
  , error_table({"u"}, {{}})
{
  enter_subsection("Global parameters");
  add_parameter("Problem type",
                problem_type,
                "Source problem or Eigenvalues problem (source|pinvit).",
                this->prm,
                Patterns::Selection("source|pinvit"));
  add_parameter("Smoother type",
                smoother_type,
                "Smoother to use in intermediate cycles.",
                this->prm,
                Patterns::Selection("gmg|chebyshev|richardson"));
  add_parameter("Finite element degree",
                degree,
                "Degree of the finite element space.");
  add_parameter("Output directory",
                output_directory,
                "Directory where we want to save output files. "
                "Leave empty for no output.");
  add_parameter("Write high order output",
                write_high_order_output,
                "Write output using high order vtu format.");
  add_parameter("Number of cycles",
                n_cycles,
                "Number of adaptive refinement steps.");
  leave_subsection();



  enter_subsection("Smoothers");
  enter_subsection("GMG");
  add_parameter("Smoother dampen",
                gmg_smoother_dampen,
                "Dampen factor for the smoother.");
  add_parameter("smoother steps",
                gmg_smoother_steps,
                "Number of smoother steps.");
  leave_subsection();
  leave_subsection();



  enter_subsection("PINVIT parameters");
  add_parameter("Number of eigenvalues to compute",
                number_of_eigenvalues,
                "",
                this->prm,
                Patterns::Integer(1));
  add_parameter("Eigenvectors indices for estimator",
                eigen_estimators,
                "All indices must be lower than the number of "
                "computed eigenvalues.");
  add_parameter("Exact eigenvalues", exact_eigenvalues);

  leave_subsection();

  enter_subsection("Grid parameters");
  add_parameter("Grid generator", name_of_grid);
  add_parameter("Grid generator arguments", arguments_for_grid);
  add_parameter("Initial refinement",
                initial_refinement,
                "Initial refinement of the triangulation.");
  add_parameter("Refinement strategy",
                refinement_strategy,
                "",
                this->prm,
                Patterns::Selection("fixed_fraction|fixed_number|global"));

  add_parameter("Homogeneous Dirichlet boundary ids",
                homogeneous_dirichlet_ids,
                "Boundary Ids over which homogeneous Dirichlet boundary "
                "conditions are applied");
  leave_subsection();

  coefficient.declare_parameters_call_back.connect(
    [&]() { this->prm.set("Function expression", "1"); });
  rhs.declare_parameters_call_back.connect(
    [&]() { this->prm.set("Function expression", "1"); });

  first_and_last_solver_control.declare_parameters_call_back.connect(
    [&]() { this->prm.set("Reduction", "1e-6"); });

  intermediate_solver_control.declare_parameters_call_back.connect([&]() {
    this->prm.set("Reduction", "0");
    this->prm.set("Max steps", "0");
    this->prm.set("Tolerance", "0");
  });


  enter_subsection("Error parameters");
  enter_my_subsection(this->prm);
  error_table.add_parameters(this->prm);
  leave_my_subsection(this->prm);
  leave_subsection();
}

template class LaplaceProblemSettings<2>;
template class LaplaceProblemSettings<3>;