#include <gtest/gtest.h>

#include "laplace_problem.h"
using namespace dealii;

template <int dim, int degree>
class TestBench : public ::testing::Test
{
public:
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