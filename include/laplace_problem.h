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

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA


using namespace dealii;

template <int dim>
class LaplaceProblemSettings : public ParameterAcceptor
{
public:
  LaplaceProblemSettings();
  bool
  try_parse(const std::string &prm_filename);

  double       smoother_dampen = 1.0;
  unsigned int smoother_steps  = 1;
  unsigned int n_steps         = 10;
  bool         output          = true;
  unsigned int degree          = 2;

  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> coefficient;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs;
};

template <int dim, int degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const LaplaceProblemSettings<dim> &settings);
  void
  run();

private:
  // We will use the following types throughout the program. First the
  // matrix-based types, after that the matrix-free classes. For the
  // matrix-free implementation, we use @p float for the level operators.
  using MatrixType         = LA::MPI::SparseMatrix;
  using VectorType         = LA::MPI::Vector;
  using PreconditionAMG    = LA::MPI::PreconditionAMG;
  using PreconditionJacobi = LA::MPI::PreconditionJacobi;

  using MatrixFreeLevelMatrix = MatrixFreeOperators::LaplaceOperator<
    dim,
    degree,
    degree + 1,
    1,
    LinearAlgebra::distributed::Vector<float>>;
  using MatrixFreeActiveMatrix = MatrixFreeOperators::LaplaceOperator<
    dim,
    degree,
    degree + 1,
    1,
    LinearAlgebra::distributed::Vector<double>>;

  using MatrixFreeLevelVector  = LinearAlgebra::distributed::Vector<float>;
  using MatrixFreeActiveVector = LinearAlgebra::distributed::Vector<double>;

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

  MatrixType             system_matrix;
  MatrixFreeActiveMatrix mf_system_matrix;
  VectorType             solution;
  VectorType             right_hand_side;
  Vector<double>         estimated_error_square_per_cell;

  MGLevelObject<MatrixType> mg_matrix;
  MGLevelObject<MatrixType> mg_interface_in;
  MGConstrainedDoFs         mg_constrained_dofs;

  MGLevelObject<MatrixFreeLevelMatrix> mf_mg_matrix;

  TimerOutput computing_timer;
};

#endif