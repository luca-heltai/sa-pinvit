#include "laplace_problem.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "utilities.h"

using namespace dealii;

template <int dim>
LaplaceProblemSettings<dim>::LaplaceProblemSettings()
  : ParameterAcceptor("/")
  , exact("Exact solution")
  , coefficient("Coefficient")
  , rhs("Forcing term")
{
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

template <int dim>
bool
LaplaceProblemSettings<dim>::try_parse(const std::string &prm_filename)
{
  if (prm_filename.size() == 0)
    {
      std::cout << "****  Error: No input file provided!\n"
                << "****  Error: Call this program as './step-50 input.prm\n"
                << "\n"
                << "****  You may want to use one of the input files in this\n"
                << "****  directory, or use the following default values\n"
                << "****  to create an input file:\n";
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        this->prm.print_parameters(std::cout, ParameterHandler::Text);
      return false;
    }

  try
    {
      this->prm.parse_input(prm_filename);
    }
  catch (std::exception &e)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cerr << e.what() << std::endl;
      return false;
    }

  return true;
}


template <int dim, int degree>
LaplaceProblem<dim, degree>::LaplaceProblem(
  const LaplaceProblemSettings<dim> &settings)
  : settings(settings)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , triangulation(
      mpi_communicator,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , mapping()
  , fe(degree)
  , dof_handler(triangulation)
  , computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
{}



template <int dim, int degree>
void
LaplaceProblem<dim, degree>::make_grid()
{
  TimerOutput::Scope timing(computing_timer, "Make grid");
  try
    {
      GridGenerator::generate_from_name_and_arguments(
        triangulation, settings.name_of_grid, settings.arguments_for_grid);
    }
  catch (...)
    {
      pcout << "Generating from name and argument failed." << std::endl
            << "Trying to read from file name." << std::endl;
      read_grid_and_cad_files(settings.name_of_grid,
                              settings.arguments_for_grid,
                              triangulation);
    }
  if (settings.output_directory != "")
    {
      GridOut       go;
      std::ofstream outfile(settings.output_directory + "coarse_grid.vtk");
      go.write_vtk(triangulation, outfile);
    }
  triangulation.refine_global(settings.initial_refinement);
}

template <int dim, int degree>
void
LaplaceProblem<dim, degree>::setup_system()
{
  TimerOutput::Scope timing(computing_timer, "Setup");

  dof_handler.distribute_dofs(fe);

  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  locally_owned_dofs = dof_handler.locally_owned_dofs();

  solution.reinit(locally_owned_dofs, mpi_communicator);
  right_hand_side.reinit(locally_owned_dofs, mpi_communicator);
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points);
  std::shared_ptr<MatrixFree<dim, double>> mf_storage =
    std::make_shared<MatrixFree<dim, double>>();
  mf_storage->reinit(
    mapping, dof_handler, constraints, QGauss<1>(degree + 1), additional_data);

  mf_system_matrix.initialize(mf_storage);

  mf_system_matrix.set_coefficient(
    make_coefficient_table(settings.coefficient, *mf_storage));
}



template <int dim, int degree>
void
LaplaceProblem<dim, degree>::setup_multigrid()
{
  TimerOutput::Scope timing(computing_timer, "Setup multigrid");

  dof_handler.distribute_mg_dofs();

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);

  const std::set<types::boundary_id> boundary_ids = {types::boundary_id(0)};
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, boundary_ids);

  const unsigned int n_levels = triangulation.n_global_levels();


  mf_mg_matrix.resize(0, n_levels - 1);

  for (unsigned int level = 0; level < n_levels; ++level)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                    level,
                                                    relevant_dofs);
      AffineConstraints<double> level_constraints;
      level_constraints.reinit(relevant_dofs);
      level_constraints.add_lines(
        mg_constrained_dofs.get_boundary_indices(level));
      level_constraints.close();

      typename MatrixFree<dim, float>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, float>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mg_level = level;
      std::shared_ptr<MatrixFree<dim, float>> mf_storage_level(
        new MatrixFree<dim, float>());
      mf_storage_level->reinit(mapping,
                               dof_handler,
                               level_constraints,
                               QGauss<1>(degree + 1),
                               additional_data);

      mf_mg_matrix[level].initialize(mf_storage_level,
                                     mg_constrained_dofs,
                                     level);

      mf_mg_matrix[level].set_coefficient(
        make_coefficient_table(settings.coefficient, *mf_storage_level));

      mf_mg_matrix[level].compute_diagonal();
    }
}



template <int dim, int degree>
void
LaplaceProblem<dim, degree>::assemble_rhs()
{
  TimerOutput::Scope timing(computing_timer, "Assemble right-hand side");

  MatrixFreeActiveVector solution_copy;
  MatrixFreeActiveVector right_hand_side_copy;
  mf_system_matrix.initialize_dof_vector(solution_copy);
  mf_system_matrix.initialize_dof_vector(right_hand_side_copy);

  solution_copy = 0.;
  constraints.distribute(solution_copy);
  solution_copy.update_ghost_values();
  right_hand_side_copy = 0;
  const Table<2, VectorizedArray<double>> &coefficient =
    *(mf_system_matrix.get_coefficient());

  FEEvaluation<dim, degree, degree + 1, 1, double> phi(
    *mf_system_matrix.get_matrix_free());

  for (unsigned int cell = 0;
       cell < mf_system_matrix.get_matrix_free()->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(solution_copy);
      phi.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_gradient(-1.0 *
                                (coefficient(cell, 0) * phi.get_gradient(q)),
                              q);
          phi.submit_value(
            evaluate_function(settings.rhs, phi.quadrature_point(q), 0), q);
        }

      phi.integrate_scatter(EvaluationFlags::values |
                              EvaluationFlags::gradients,
                            right_hand_side_copy);
    }

  right_hand_side_copy.compress(VectorOperation::add);

  ChangeVectorTypes::copy(right_hand_side, right_hand_side_copy);
}



template <int dim, int degree>
void
LaplaceProblem<dim, degree>::solve()
{
  TimerOutput::Scope timing(computing_timer, "Solve");

  SolverControl solver_control(1000, 1.e-10 * right_hand_side.l2_norm());
  solver_control.enable_history_data();

  solution = 0.;

  computing_timer.enter_subsection("Solve: Preconditioner setup");

  MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  SolverControl coarse_solver_control(1000, 1e-12, false, false);
  SolverCG<MatrixFreeLevelVector> coarse_solver(coarse_solver_control);
  PreconditionIdentity            identity;
  MGCoarseGridIterativeSolver<MatrixFreeLevelVector,
                              SolverCG<MatrixFreeLevelVector>,
                              MatrixFreeLevelMatrix,
                              PreconditionIdentity>
    coarse_grid_solver(coarse_solver, mf_mg_matrix[0], identity);

  using Smoother = dealii::PreconditionJacobi<MatrixFreeLevelMatrix>;
  MGSmootherPrecondition<MatrixFreeLevelMatrix, Smoother, MatrixFreeLevelVector>
    smoother;
  smoother.initialize(
    mf_mg_matrix, typename Smoother::AdditionalData(settings.smoother_dampen));
  smoother.set_steps(settings.smoother_steps);

  mg::Matrix<MatrixFreeLevelVector> mg_m(mf_mg_matrix);

  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MatrixFreeLevelMatrix>>
    mg_interface_matrices;
  mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
  for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mf_mg_matrix[level]);
  mg::Matrix<MatrixFreeLevelVector> mg_interface(mg_interface_matrices);

  Multigrid<MatrixFreeLevelVector> mg(
    mg_m, coarse_grid_solver, mg_transfer, smoother, smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, MatrixFreeLevelVector, MGTransferMatrixFree<dim, float>>
    preconditioner(dof_handler, mg, mg_transfer);

  // Copy the solution vector and right-hand side from LA::MPI::Vector
  // to dealii::LinearAlgebra::distributed::Vector so that we can
  // solve.
  MatrixFreeActiveVector solution_copy;
  MatrixFreeActiveVector right_hand_side_copy;
  mf_system_matrix.initialize_dof_vector(solution_copy);
  mf_system_matrix.initialize_dof_vector(right_hand_side_copy);

  ChangeVectorTypes::copy(solution_copy, solution);
  ChangeVectorTypes::copy(right_hand_side_copy, right_hand_side);
  computing_timer.leave_subsection("Solve: Preconditioner setup");

  // Timing for 1 V-cycle.
  {
    TimerOutput::Scope timing(computing_timer, "Solve: 1 multigrid V-cycle");
    preconditioner.vmult(solution_copy, right_hand_side_copy);
  }
  solution_copy = 0.;

  // Solve the linear system, update the ghost values of the solution,
  // copy back to LA::MPI::Vector and distribute constraints.
  {
    SolverCG<MatrixFreeActiveVector> solver(solver_control);

    TimerOutput::Scope timing(computing_timer, "Solve: CG");
    solver.solve(mf_system_matrix,
                 solution_copy,
                 right_hand_side_copy,
                 preconditioner);
  }

  solution_copy.update_ghost_values();
  ChangeVectorTypes::copy(solution, solution_copy);
  constraints.distribute(solution);

  pcout << "   Number of CG iterations:      " << solver_control.last_step()
        << std::endl;
}


// @sect3{The error estimator}

// We use the FEInterfaceValues class to assemble an error estimator to decide
// which cells to refine. See the exact definition of the cell and face
// integrals in the introduction. To use the method, we define Scratch and
// Copy objects for the MeshWorker::mesh_loop() with much of the following
// code being in essence as was set up in step-12 already (or at least similar
// in spirit).
template <int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &      mapping,
              const FiniteElement<dim> &fe,
              const unsigned int        quadrature_degree,
              const UpdateFlags         update_flags,
              const UpdateFlags         interface_update_flags)
    : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
    , fe_interface_values(mapping,
                          fe,
                          QGauss<dim - 1>(quadrature_degree),
                          interface_update_flags)
  {}


  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(scratch_data.fe_values.get_mapping(),
                          scratch_data.fe_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {}

  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
};



struct CopyData
{
  CopyData()
    : cell_index(numbers::invalid_unsigned_int)
    , value(0.)
  {}

  CopyData(const CopyData &) = default;

  struct FaceData
  {
    unsigned int cell_indices[2];
    double       values[2];
  };

  unsigned int          cell_index;
  double                value;
  std::vector<FaceData> face_data;
};


template <int dim, int degree>
void
LaplaceProblem<dim, degree>::estimate()
{
  TimerOutput::Scope timing(computing_timer, "Estimate");

  VectorType temp_solution;
  temp_solution.reinit(locally_owned_dofs,
                       locally_relevant_dofs,
                       mpi_communicator);
  temp_solution = solution;

  estimated_error_square_per_cell.reinit(triangulation.n_active_cells());

  using Iterator = typename DoFHandler<dim>::active_cell_iterator;

  // Assembler for cell residual $h^2 \| f + \epsilon \triangle u \|_K^2$
  auto cell_worker = [&](const Iterator &  cell,
                         ScratchData<dim> &scratch_data,
                         CopyData &        copy_data) {
    FEValues<dim> &fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);

    const double rhs_value = settings.rhs.value(cell->center());

    const double nu = settings.coefficient.value(cell->center());

    std::vector<Tensor<2, dim>> hessians(fe_values.n_quadrature_points);
    fe_values.get_function_hessians(temp_solution, hessians);

    copy_data.cell_index = cell->active_cell_index();

    double residual_norm_square = 0.;
    for (unsigned k = 0; k < fe_values.n_quadrature_points; ++k)
      {
        const double residual = (rhs_value + nu * trace(hessians[k]));
        residual_norm_square += residual * residual * fe_values.JxW(k);
      }

    copy_data.value =
      cell->diameter() * cell->diameter() * residual_norm_square;
  };

  // Assembler for face term $\sum_F h_F \| \jump{\epsilon \nabla u \cdot n}
  // \|_F^2$
  auto face_worker = [&](const Iterator &    cell,
                         const unsigned int &f,
                         const unsigned int &sf,
                         const Iterator &    ncell,
                         const unsigned int &nf,
                         const unsigned int &nsf,
                         ScratchData<dim> &  scratch_data,
                         CopyData &          copy_data) {
    FEInterfaceValues<dim> &fe_interface_values =
      scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

    copy_data.face_data.emplace_back();
    CopyData::FaceData &copy_data_face = copy_data.face_data.back();

    copy_data_face.cell_indices[0] = cell->active_cell_index();
    copy_data_face.cell_indices[1] = ncell->active_cell_index();

    const double coeff1 = settings.coefficient.value(cell->center());
    const double coeff2 = settings.coefficient.value(ncell->center());

    std::vector<Tensor<1, dim>> grad_u[2];

    for (unsigned int i = 0; i < 2; ++i)
      {
        grad_u[i].resize(fe_interface_values.n_quadrature_points);
        fe_interface_values.get_fe_face_values(i).get_function_gradients(
          temp_solution, grad_u[i]);
      }

    double jump_norm_square = 0.;

    for (unsigned int qpoint = 0;
         qpoint < fe_interface_values.n_quadrature_points;
         ++qpoint)
      {
        const double jump =
          coeff1 * grad_u[0][qpoint] * fe_interface_values.normal(qpoint) -
          coeff2 * grad_u[1][qpoint] * fe_interface_values.normal(qpoint);

        jump_norm_square += jump * jump * fe_interface_values.JxW(qpoint);
      }

    const double h           = cell->face(f)->measure();
    copy_data_face.values[0] = 0.5 * h * jump_norm_square;
    copy_data_face.values[1] = copy_data_face.values[0];
  };

  auto copier = [&](const CopyData &copy_data) {
    if (copy_data.cell_index != numbers::invalid_unsigned_int)
      estimated_error_square_per_cell[copy_data.cell_index] += copy_data.value;

    for (auto &cdf : copy_data.face_data)
      for (unsigned int j = 0; j < 2; ++j)
        estimated_error_square_per_cell[cdf.cell_indices[j]] += cdf.values[j];
  };

  const unsigned int n_gauss_points = degree + 1;
  ScratchData<dim>   scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_hessians | update_quadrature_points |
                                  update_JxW_values,
                                update_values | update_gradients |
                                  update_JxW_values | update_normal_vectors);
  CopyData           copy_data;

  // We need to assemble each interior face once but we need to make sure that
  // both processes assemble the face term between a locally owned and a ghost
  // cell. This is achieved by setting the
  // MeshWorker::assemble_ghost_faces_both flag. We need to do this, because
  // we do not communicate the error estimator contributions here.
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_ghost_faces_both |
                          MeshWorker::assemble_own_interior_faces_once,
                        /*boundary_worker=*/nullptr,
                        face_worker);

  const double global_error_estimate =
    std::sqrt(Utilities::MPI::sum(estimated_error_square_per_cell.l1_norm(),
                                  mpi_communicator));
  pcout << "   Global error estimate:        " << global_error_estimate
        << std::endl;
}



template <int dim, int degree>
void
LaplaceProblem<dim, degree>::refine_grid()
{
  TimerOutput::Scope timing(computing_timer, "Refine grid");

  const double refinement_fraction = 1. / (std::pow(2.0, dim) - 1.);
  if (settings.refinement_strategy == "fixed_number")
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_square_per_cell, refinement_fraction, 0.0);
  else if (settings.refinement_strategy == "fixed_fraction")
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_square_per_cell, refinement_fraction, 0.0);
  else if (settings.refinement_strategy == "global")
    for (const auto &cell : triangulation.active_cell_iterators())
      cell->set_refine_flag();
  else
    Assert(false, ExcInternalError("Unknown refinement strategy."));

  triangulation.execute_coarsening_and_refinement();
}


// @sect4{LaplaceProblem::output_results()}

// The output_results() function is similar to the ones found in many of the
// tutorials (see step-40 for example).
template <int dim, int degree>
void
LaplaceProblem<dim, degree>::output_results(const unsigned int cycle)
{
  TimerOutput::Scope timing(computing_timer, "Output results");

  VectorType temp_solution;
  temp_solution.reinit(locally_owned_dofs,
                       locally_relevant_dofs,
                       mpi_communicator);
  temp_solution = solution;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(temp_solution, "solution");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  Vector<float> level(triangulation.n_active_cells());
  for (const auto &cell : triangulation.active_cell_iterators())
    level(cell->active_cell_index()) = cell->level();
  data_out.add_data_vector(level, "level");

  if (estimated_error_square_per_cell.size() > 0)
    data_out.add_data_vector(estimated_error_square_per_cell,
                             "estimated_error_square_per_cell");

  data_out.build_patches();

  const std::string pvtu_filename =
    data_out.write_vtu_with_pvtu_record(settings.output_directory,
                                        "solution",
                                        cycle,
                                        mpi_communicator,
                                        2 /*n_digits*/,
                                        1 /*n_groups*/);

  pcout << "   Wrote " << pvtu_filename << std::endl;

  GridOut go;
  auto    grid_name = "level_grid_" + Utilities::int_to_string(cycle, 2);
  go.write_mesh_per_processor_as_vtu(triangulation,
                                     settings.output_directory + grid_name,
                                     true);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      static std::vector<std::pair<double, std::string>> times_and_names;
      times_and_names.push_back(std::make_pair((double)cycle, pvtu_filename));
      std::ofstream pvd_output(settings.output_directory + "solution.pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);

      static std::vector<std::pair<double, std::string>> times_and_grid_names;
      times_and_grid_names.push_back(
        std::make_pair((double)cycle, grid_name + ".pvtu"));
      std::ofstream pvd_output_grids(settings.output_directory +
                                     "level_grid.pvd");
      DataOutBase::write_pvd_record(pvd_output_grids, times_and_grid_names);
    }
}


// @sect4{LaplaceProblem::run()}

// As in most tutorials, this function calls the various functions defined
// above to setup, assemble, solve, and output the results.
template <int dim, int degree>
void
LaplaceProblem<dim, degree>::run()
{
  for (unsigned int cycle = 0; cycle < settings.n_steps; ++cycle)
    {
      pcout << "Cycle " << cycle << ':' << std::endl;
      if (cycle == 0)
        make_grid();
      else
        refine_grid();

      pcout << "   Number of active cells:       "
            << triangulation.n_global_active_cells();

      // We only output level cell data for the GMG methods (same with DoF
      // data below). Note that the partition efficiency is irrelevant for AMG
      // since the level hierarchy is not distributed or used during the
      // computation.
      pcout << " (" << triangulation.n_global_levels() << " global levels)"
            << std::endl
            << "   Partition efficiency:         "
            << 1.0 / MGTools::workload_imbalance(triangulation);
      pcout << std::endl;

      setup_system();

      setup_multigrid();

      pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs();
      pcout << " (by level: ";
      for (unsigned int level = 0; level < triangulation.n_global_levels();
           ++level)
        pcout << dof_handler.n_dofs(level)
              << (level == triangulation.n_global_levels() - 1 ? ")" : ", ");

      pcout << std::endl;

      assemble_rhs();

      solve();
      estimate();

      if (settings.output_directory != "")
        output_results(cycle);

      computing_timer.print_summary();
      computing_timer.reset();
    }
}

template class LaplaceProblemSettings<2>;
template class LaplaceProblemSettings<3>;

// degree 1
template class LaplaceProblem<2, 1>;
template class LaplaceProblem<3, 1>;

// degree 2
template class LaplaceProblem<2, 2>;
template class LaplaceProblem<3, 2>;

// degree 3
template class LaplaceProblem<2, 3>;
template class LaplaceProblem<3, 3>;