#ifndef laplace_problem_h
#define laplace_problem_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <fstream>

#include "laplace_problem_settings.h"

using namespace dealii;

namespace LA
{
  using namespace LinearAlgebraTrilinos;
} // namespace LA

template <int dim, int degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const LaplaceProblemSettings<dim> &settings);
  void
  run();

  using VectorType         = LA::MPI::Vector;
  using PreconditionAMG    = LA::MPI::PreconditionAMG;
  using PreconditionJacobi = LA::MPI::PreconditionJacobi;

  using MatrixFreeLevelVector  = LinearAlgebra::distributed::Vector<float>;
  using MatrixFreeActiveVector = LinearAlgebra::distributed::Vector<double>;

  using MatrixFreeLevelMatrix = MatrixFreeOperators::
    LaplaceOperator<dim, degree, degree + 1, 1, MatrixFreeLevelVector>;

  using MatrixFreeActiveMatrix = MatrixFreeOperators::
    LaplaceOperator<dim, degree, degree + 1, 1, MatrixFreeActiveVector>;

  using MatrixFreeLevelMassMatrix = MatrixFreeOperators::
    MassOperator<dim, degree, degree + 1, 1, MatrixFreeLevelVector>;

  using MatrixFreeActiveMassMatrix = MatrixFreeOperators::
    MassOperator<dim, degree, degree + 1, 1, MatrixFreeActiveVector>;


  /**
   * Create the grid from the specification in the settings.
   */
  void
  make_grid();
  void
  setup_system(const unsigned int cycle);
  void
  setup_multigrid();
  void
  assemble_multigrid();
  void
  assemble_rhs();
  void
  solve(const unsigned int cycle);

  /**
   * According to the problem type, solve using the given preconditioner,
   * or perform on smoothing step using the given preconditioner.
   *
   * At the end of this call, @p locally_relevant_solution, and @p solution
   * will contain sensible data (both for source problem and for pinvit
   * problem). When calling pinvit, then also eigenvalues and eigenvectors
   * will be updated.
   */
  template <class PreconditionerType>
  void
  solve_system(const PreconditionerType &prec, SolverControl &control);

  /**
   * Compute a Residual based error estimator on the source or eigenvalue
   * problem.
   */
  void
  estimate();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle);
  void
  compute_errors();
  void
  print_grid_info() const;

  const LaplaceProblemSettings<dim> &settings;

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ1<dim>                      mapping;
  FE_Q<dim>                                 fe;

  DoFHandler<dim>                                          dof_handler;
  parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer;
  parallel::distributed::SolutionTransfer<dim, MatrixFreeActiveVector>
    eigenvectors_transfer;

  IndexSet                  locally_owned_dofs;
  IndexSet                  locally_relevant_dofs;
  AffineConstraints<double> constraints;

  MatrixFreeActiveMatrix              stiffness_operator;
  MatrixFreeActiveMassMatrix          mass_operator;
  VectorType                          solution;
  VectorType                          locally_relevant_solution;
  VectorType                          right_hand_side;
  std::vector<MatrixFreeActiveVector> eigenvectors;
  std::vector<double>                 eigenvalues;
  Vector<double>                      estimated_error_square_per_cell;
  double                              global_error_estimate;

  MGLevelObject<MatrixFreeLevelMatrix>     mg_stiffness_operator;
  MGLevelObject<MatrixFreeLevelMassMatrix> mg_mass_operator;
  MGConstrainedDoFs                        mg_constrained_dofs;
  MGConstrainedDoFs                        mg_constrained_mass_dofs;

  TimerOutput computing_timer;

  /**
   * Make sure google tests can access private members.
   */
  template <int, int>
  friend class TestBench;
};

#endif