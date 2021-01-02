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

int
main(int argc, char *argv[])
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  LaplaceProblemSettings<DEAL_DIMENSION> settings;
  std::string filename = (argc > 1) ? (argv[1]) : "parameters.prm";
  ParameterAcceptor::initialize(filename, "used_" + filename);

  try
    {
      switch (settings.degree)
        {
          case 1:
            {
              constexpr unsigned int                    fe_degree = 1;
              LaplaceProblem<DEAL_DIMENSION, fe_degree> test(settings);
              test.run();
            }
            break;
          case 2:
            {
              constexpr unsigned int                    fe_degree = 2;
              LaplaceProblem<DEAL_DIMENSION, fe_degree> test(settings);
              test.run();
            }
            break;
          case 3:
            {
              constexpr unsigned int                    fe_degree = 3;
              LaplaceProblem<DEAL_DIMENSION, fe_degree> test(settings);
              test.run();
            }
            break;
          default:
            AssertThrow(
              false,
              ExcMessage(
                "The selected degree was not compiled. Please choose a lower one."));
        }
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