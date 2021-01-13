#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <type_traits>

#include "test_bench.h"

using namespace dealii;

/* Default parameter file:
 *
 * subsection Coefficient
 *   set Function constants  =
 *   set Function expression = 1
 *   set Variable names      = x,y,t
 * end
 * subsection Exact solution
 *   set Function constants  =
 *   set Function expression = 0
 *   set Variable names      = x,y,t
 * end
 * subsection Forcing term
 *   set Function constants  =
 *   set Function expression = 1
 *   set Variable names      = x,y,t
 * end
 * subsection Global parameters
 *   set Finite element degree   = 2
 *   set Number of cycles        = 7
 *   set Output directory        =
 *   set Problem type            = source
 *   set Smoother type           = gmg
 *   set Write high order output = true
 * end
 * subsection Grid parameters
 *   set Grid generator                     = hyper_L
 *   set Grid generator arguments           = -1.: 1.: false
 *   set Homogeneous Dirichlet boundary ids = 0
 *   set Initial refinement                 = 1
 *   set Refinement strategy                = fixed_number
 * end
 * subsection PINVIT parameters
 *   set Eigenvectors indices for estimator = 0
 *   set Number of eigenvalues to compute   = 1
 * end
 * subsection Smoothers
 *   subsection GMG
 *     set Smoother dampen = 1
 *     set smoother steps  = 1
 *   end
 * end
 * subsection Solver
 *   subsection First and last cycle
 *     set Log frequency = 1
 *     set Log history   = false
 *     set Log result    = true
 *     set Max steps     = 100
 *     set Reduction     = 1e-6
 *     set Tolerance     = 1.e-10
 *   end
 *   subsection Intermediate cycles
 *     set Log frequency = 1
 *     set Log history   = false
 *     set Log result    = true
 *     set Max steps     = 0
 *     set Reduction     = 0
 *     set Tolerance     = 0
 *   end
 * end
 */


TEST_F(TestBench2D, CheckStandardSettings)
{
  setup(R"(
     subsection Global parameters
       set Number of cycles  = 1
     end
     subsection Smoothers
       subsection GMG
         set Smoother dampen = 2
         set smoother steps  = 3
       end
     end
    )");

  ASSERT_EQ(settings.n_cycles, 1);
  ASSERT_EQ(settings.gmg_smoother_dampen, 2.0);
  ASSERT_EQ(settings.gmg_smoother_steps, 3);
}

TEST_F(TestBench2D, CheckFunctionSettings)
{
  setup(R"(
    subsection Coefficient
      set Function constants  = 
      set Function expression = x+y
      set Variable names      = x,y,t
    end
    subsection Exact solution
      set Function constants  = 
      set Function expression = 2*x+2*y
      set Variable names      = x,y,t
    end
    subsection Forcing term
      set Function constants  = 
      set Function expression = 3*x+3*y
      set Variable names      = x,y,t
    end
    )");

  ASSERT_EQ(settings.coefficient.value({1.0, 1.0}), 2.0);
  ASSERT_EQ(settings.exact.value({1.0, 1.0}), 4.0);
  ASSERT_EQ(settings.rhs.value({1.0, 1.0}), 6.0);
}