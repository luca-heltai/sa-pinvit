#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <type_traits>

#include "test_bench.h"

using namespace dealii;

/* Default parameter file:

set degree          = 2
set n_steps         = 10
set output          = true
set smoother dampen = 1
set smoother steps  = 1
subsection Coefficient
  set Function constants  =
  set Function expression = 0
  set Variable names      = x,y,t
end
subsection Exact solution
  set Function constants  =
  set Function expression = 0
  set Variable names      = x,y,t
end
subsection Forcing term
  set Function constants  =
  set Function expression = 0
  set Variable names      = x,y,t
end

*/


TEST_F(TestBench2D, CheckStandardSettings)
{
  setup(R"(
    set n_steps = 1
    set smoother dampen = 2
    set smoother steps = 3
    )");

  ASSERT_EQ(settings.n_steps, 1);
  ASSERT_EQ(settings.smoother_dampen, 2.0);
  ASSERT_EQ(settings.smoother_steps, 3);
}

TEST_F(TestBench2D, CheckFunctionSettings)
{
  setup(R"(
    set degree          = 2
    set n_steps         = 10
    set output          = true
    set smoother dampen = 1
    set smoother steps  = 1
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