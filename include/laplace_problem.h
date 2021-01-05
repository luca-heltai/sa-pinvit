/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified by: Luca Heltai, 2020
 */
#ifndef laplace_problem_h
#define laplace_problem_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
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
  setup_system();
  void
  setup_multigrid();
  void
  assemble_multigrid();
  void
  assemble_rhs();
  void
  solve();
  void
  estimate();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle);
  void
  print_grid_info() const;

  const LaplaceProblemSettings<dim> &settings;

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ1<dim>                      mapping;
  FE_Q<dim>                                 fe;

  DoFHandler<dim> dof_handler;

  IndexSet                  locally_owned_dofs;
  IndexSet                  locally_relevant_dofs;
  AffineConstraints<double> constraints;

  MatrixFreeActiveMatrix     mf_system_matrix;
  MatrixFreeActiveMassMatrix mf_mass_matrix;
  VectorType                 solution;
  VectorType                 right_hand_side;
  Vector<double>             estimated_error_square_per_cell;

  MGLevelObject<MatrixFreeLevelMatrix>     mf_mg_matrix;
  MGLevelObject<MatrixFreeLevelMassMatrix> mf_mg_mass_matrix;
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