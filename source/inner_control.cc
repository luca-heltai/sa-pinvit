#include "inner_control.h"

InnerControl::InnerControl(const unsigned int max_iterations,
                           const double       tolerance,
                           const double       reduction,
                           bool               log,
                           bool               loglast)
  : ReductionControl(max_iterations, tolerance, reduction, log, loglast)
{}



SolverControl::State
InnerControl::check(const unsigned int iteration, const double check_value)
{
  if (iteration >= max_steps())
    return SolverControl::success;
  else
    return ReductionControl::check(iteration, check_value);
}