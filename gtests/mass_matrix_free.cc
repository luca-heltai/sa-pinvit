#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <type_traits>

#include "pinvit.h"
#include "test_bench.h"
#include "utilities.h"

using namespace dealii;

TEST_F(TestBench2D, IntegrateWithMassMatrix)
{
  setup(R"(
    set Problem type                         = pinvit
    subsection Grid parameters
      set Grid generator                     = hyper_cube
      set Grid generator arguments           = 0.: 1.: false
      set Homogeneous Dirichlet boundary ids = 0
      set Initial refinement                 = 3
    end
    )");

  pb.make_grid();
  pb.setup_system(0);
  pb.setup_multigrid();

  // ref 3 = 8 cells x 8 cells = 64 cells.
  ASSERT_EQ(pb.triangulation.n_global_active_cells(), 64);

  MatrixFreeActiveVector ones;
  MatrixFreeActiveVector M_ones;
  pb.mass_operator.initialize_dof_vector(ones);
  pb.mass_operator.initialize_dof_vector(M_ones);
  ones = 1.0;

  // One in each support point: 17*17 = 289
  ASSERT_NEAR(ones.l1_norm(), 289, 1e-10);

  pb.mass_operator.vmult(M_ones, ones);
  // one = \int_\Omega ones * ones = 1.0
  auto one = M_ones * ones;
  pb.pcout << "one: " << one << std::endl;

  ASSERT_NEAR(one, 1.0, 1e-10);

  // Check trivial rayleigh quotient
  one = compute_rayleigh_quotient(pb.mass_operator, pb.mass_operator, ones);

  ASSERT_NEAR(one, 1.0, 1e-10);
}