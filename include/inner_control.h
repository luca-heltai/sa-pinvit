#ifndef lh_inner_control_h
#define lh_inner_control_h

#include <deal.II/lac/solver_control.h>

using namespace dealii;


class InnerControl : public ReductionControl
{
public:
  InnerControl() = default;

  InnerControl(const unsigned int max_iterations,
               const double       tolerance,
               const double       reduction,
               bool               log,
               bool               loglast);

  virtual SolverControl::State
  check(const unsigned int iteration, const double check_value) override;
};

#endif