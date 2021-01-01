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

#include "laplace_problem.h"

// @sect3{The main() function}

// This is a similar main function to step-40, with the exception that
// we require the user to pass a .prm file as a sole command line
// argument (see step-29 and the documentation of the ParameterHandler
// class for a complete discussion of parameter files).
int
main(int argc, char *argv[])
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  LaplaceProblemSettings<DEAL_DIMENSION> settings;
  if (!settings.try_parse((argc > 1) ? (argv[1]) : ""))
    return 0;

  try
    {
      constexpr unsigned int fe_degree = 2;

      LaplaceProblem<DEAL_DIMENSION, fe_degree> test(settings);
      test.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
      return 1;
    }

  return 0;
}