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
#ifndef lh_utilities_h
#define lh_utilities_h


using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>


template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized,
                  const unsigned int                         component)
{
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      result[v] = function.value(p, component);
    }
  return result;
}

template <int dim, typename Number, int n_components = dim>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, n_components);
  Tensor<1, n_components, VectorizedArray<Number>> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        result[d][v] = function.value(p, d);
    }
  return result;
}


// MatrixFree operators must use the
// dealii::LinearAlgebra::distributed::Vector vector type. Here we define
// operations which copy to and from Trilinos vectors for compatibility with
// the matrix-based code. Note that this functionality does not currently
// exist for PETSc vector types, so Trilinos must be installed to use the
// MatrixFree solver in this tutorial.
namespace ChangeVectorTypes
{
  template <typename number>
  void
  copy(LA::MPI::Vector &                                         out,
       const dealii::LinearAlgebra::distributed::Vector<number> &in)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rwv(
      out.locally_owned_elements());
    rwv.import(in, VectorOperation::insert);
#ifdef USE_PETSC_LA
    AssertThrow(false,
                ExcMessage("CopyVectorTypes::copy() not implemented for "
                           "PETSc vector types."));
#else
    out.import(rwv, VectorOperation::insert);
#endif
  }



  template <typename number>
  void
  copy(dealii::LinearAlgebra::distributed::Vector<number> &out,
       const LA::MPI::Vector &                             in)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rwv;
#ifdef USE_PETSC_LA
    (void)in;
    AssertThrow(false,
                ExcMessage("CopyVectorTypes::copy() not implemented for "
                           "PETSc vector types."));
#else
    rwv.reinit(in);
#endif
    out.import(rwv, VectorOperation::insert);
  }
} // namespace ChangeVectorTypes



template <int dim, typename number>
std::shared_ptr<Table<2, VectorizedArray<number>>>
make_coefficient_table(
  const Function<dim> &                                   function,
  const MatrixFree<dim, number, VectorizedArray<number>> &mf_storage)
{
  auto coefficient_table =
    std::make_shared<Table<2, VectorizedArray<number>>>();
  FEEvaluation<dim, -1, 0, 1, number> fe_eval(mf_storage);
  const unsigned int                  n_cells    = mf_storage.n_macro_cells();
  const unsigned int                  n_q_points = fe_eval.n_q_points;
  coefficient_table->reinit(n_cells, 1);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      fe_eval.reinit(cell);
      VectorizedArray<number> average_value = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
        average_value +=
          evaluate_function(function, fe_eval.quadrature_point(q));
      average_value /= n_q_points;
      (*coefficient_table)(cell, 0) = average_value;
    }
  return coefficient_table;
}



#endif