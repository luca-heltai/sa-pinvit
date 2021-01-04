#include <gtest/gtest.h>

#include "laplace_problem.h"
using namespace dealii;

template <int dim, int degree>
class TestBench : public ::testing::Test
{
protected:
  ParameterHandler &          prm = ParameterAcceptor::prm;
  LaplaceProblemSettings<dim> settings;
  LaplaceProblem<dim, degree> pb;

  void
  setup(const std::string &parameters)
  {
    prm.parse_input_from_string(parameters);
    ParameterAcceptor::parse_all_parameters();
  }

  TestBench()
    : pb(settings)
  {
    ParameterAcceptor::declare_all_parameters();
  }
};

using TestBench2D = TestBench<2, 2>;
using TestBench3D = TestBench<3, 2>;