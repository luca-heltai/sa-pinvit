#ifndef lh_pinvit_h
#define lh_pinvit_h

#include <deal.II/base/exceptions.h>

using namespace dealii;

template <class VectorType, class StiffnessMatrixType, class MassMatrixType>
typename VectorType::value_type
compute_rayleigh_quotient(const StiffnessMatrixType &stiffness_matrix,
                          const MassMatrixType &     mass_matrix,
                          const VectorType &         vector)
{
  VectorType dst_tmp(vector);
  stiffness_matrix.vmult(dst_tmp, vector);
  const auto vtAv = vector * dst_tmp;
  mass_matrix.vmult(dst_tmp, vector);
  const auto vtMv = vector * dst_tmp;
  Assert(vtMv > 0.0,
         ExcInternalError("Mass matrix must be positive definite."));
  return vtAv / vtMv;
}



template <class VectorType,
          class StiffnessMatrixType,
          class MassMatrixType,
          class PreconditionerType,
          class ConstraintsType>
void
one_step_pinvit(typename VectorType::value_type &mu,
                VectorType &                     vector,
                const StiffnessMatrixType &      stiffness_matrix,
                const MassMatrixType &           mass_matrix,
                const PreconditionerType &       preconditioner,
                const ConstraintsType &          constraints)
{
  VectorType v1(vector);
  VectorType v2(vector);
  // v1 = A v
  stiffness_matrix.vmult(v1, vector);
  // v2 = M v
  mass_matrix.vmult(v2, vector);

  // v^T A v
  const auto vtAv = vector * v1;
  // v^T M v
  const auto vtMv = vector * v2;

  Assert(vtMv > 0.0,
         ExcInternalError("Mass matrix must be positive definite."));

  // mu(v) = (v^T A v) / (v^T M v)
  const auto mu_old = vtAv / vtMv;

  // v = v - P^{-1}(A v - mu(v) M v)
  // v2 = mu(v) M v
  v2 *= mu_old;
  // v1 = A v - mu(v) M v
  v1 -= v2;
  // v2 = P^{-1}(A v - mu(v) M v)
  preconditioner.vmult(v2, v1);
  // vnew = v - P^{-1}(A v - mu(v) M v)
  vector -= v2;
  constraints.distribute(vector);

  // vnew = vnew / ||vnew||
  mass_matrix.vmult(v2, vector);
  const auto norm = std::sqrt(v2 * vector);
  vector *= 1 / norm;

  // Since now v^T M V = 1, mu = v^T A v
  v2 = vector;
  constraints.set_zero(v2);
  stiffness_matrix.vmult(v1, v2);
  mu = v1 * vector;
}



template <class VectorType,
          class StiffnessMatrixType,
          class MassMatrixType,
          class PreconditionerType,
          class ConstraintsType>
void
one_step_pinvit(std::vector<typename VectorType::value_type> &mus,
                std::vector<VectorType> &                     vectors,
                const StiffnessMatrixType &                   stiffness_matrix,
                const MassMatrixType &                        mass_matrix,
                const PreconditionerType &                    preconditioner,
                const ConstraintsType &                       constraints)
{
  AssertDimension(mus.size(), vectors.size());
  if (mus.size() == 0)
    return;
  unsigned int n = mus.size();

  LAPACKFullMatrix<typename VectorType::value_type>    A(n, n);
  LAPACKFullMatrix<typename VectorType::value_type>    M(n, n);
  std::vector<Vector<typename VectorType::value_type>> W(
    n, Vector<typename VectorType::value_type>(n));

  std::vector<VectorType> AV(n, vectors[0]);
  std::vector<VectorType> MV(n, vectors[0]);

  // Compute \tilde V
  for (unsigned int i = 0; i < n; ++i)
    {
      stiffness_matrix.vmult(AV[i], vectors[i]);
      constraints.set_zero(AV[i]);
      const auto vitAvi = AV[i] * vectors[i];
      mass_matrix.vmult(MV[i], vectors[i]);
      const auto vitMvi = MV[i] * vectors[i];
      Assert(vitMvi > 0,
             ExcInternalError("Mass matrix must be positive definite."));
      mus[i] = vitAvi / vitMvi;

      // vnew = v - P^{-1}(A v - mu(v) M v)
      MV[i] *= mus[i];
      AV[i] -= MV[i];
      preconditioner.vmult(MV[i], AV[i]);
      vectors[i] -= MV[i];
      // Precompute vector matrix products.
      stiffness_matrix.vmult(AV[i], vectors[i]);
      constraints.set_zero(AV[i]);
      mass_matrix.vmult(MV[i], vectors[i]);
    }

  // Assemble A and M.
  for (unsigned int i = 0; i < n; ++i)
    for (unsigned int j = 0; j < n; ++j)
      {
        A(i, j) = AV[i] * vectors[j];
        M(i, j) = MV[i] * vectors[j];
      }

  // Compute generalized eigenvalues and eigenvectors
  A.compute_generalized_eigenvalues_symmetric(M, W);

  // Now: V(i,j) = sum_k V(i,k) W[k,j]
  for (unsigned int i = 0; i < n; ++i)
    {
      mus[i] = A.eigenvalue(i).real();
      AV[i]  = vectors[i];
    }
  for (unsigned int i = 0; i < n; ++i)
    {
      vectors[i] = 0;
      for (unsigned int j = 0; j < n; ++j)
        {
          MV[j] = AV[j];
          MV[j] *= W[i][j];
          vectors[i] += MV[j];
        }
      constraints.distribute(vectors[i]);
      mass_matrix.vmult(MV[i], vectors[i]);
      vectors[i] /= std::sqrt(MV[i] * vectors[i]);
    }
}
#endif