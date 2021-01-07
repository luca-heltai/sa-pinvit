#ifndef lh_utilities_h
#define lh_utilities_h


using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>
#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/lapack_full_matrix.h>

template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized,
                  const unsigned int                         component)
{
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < p_vectorized[0].size(); ++v)
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
  for (unsigned int v = 0; v < p_vectorized[0].size(); ++v)
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
  const unsigned int                  n_cells    = mf_storage.n_cell_batches();
  const unsigned int                  n_q_points = fe_eval.n_q_points;
  coefficient_table->reinit(n_cells, 1);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      fe_eval.reinit(cell);
      VectorizedArray<number> average_value = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
        average_value +=
          evaluate_function(function, fe_eval.quadrature_point(q), 0);
      average_value /= n_q_points;
      (*coefficient_table)(cell, 0) = average_value;
    }
  return coefficient_table;
}


template <int dim, int spacedim>
void
read_grid_and_cad_files(const std::string &           grid_file_name,
                        const std::string &           ids_and_cad_file_names,
                        Triangulation<dim, spacedim> &tria)
{
  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read(grid_file_name);

  // If we got to this point, then the Triangulation has been read, and we are
  // ready to attach to it the correct manifold descriptions. We perform the
  // next lines of code only if deal.II has been built with OpenCASCADE
  // support. For each entry in the map, we try to open the corresponding CAD
  // file, we analyze it, and according to its content, opt for either a
  // OpenCASCADE::ArcLengthProjectionLineManifold (if the CAD file contains a
  // single `TopoDS_Edge` or a single `TopoDS_Wire`) or a
  // OpenCASCADE::NURBSPatchManifold, if the file contains a single face.
  // Notice that if the CAD files do not contain single wires, edges, or
  // faces, an assertion will be throw in the generation of the Manifold.
  //
  // We use the Patterns::Tools::Convert class to do the conversion from the
  // string to a map between manifold ids and file names for us:
#ifdef DEAL_II_WITH_OPENCASCADE
  using map_type  = std::map<types::manifold_id, std::string>;
  using Converter = Patterns::Tools::Convert<map_type>;

  for (const auto &pair : Converter::to_value(ids_and_cad_file_names))
    {
      const auto &manifold_id   = pair.first;
      const auto &cad_file_name = pair.second;

      const auto extension = boost::algorithm::to_lower_copy(
        cad_file_name.substr(cad_file_name.find_last_of('.') + 1));

      TopoDS_Shape shape;
      if (extension == "iges" || extension == "igs")
        shape = OpenCASCADE::read_IGES(cad_file_name);
      else if (extension == "step" || extension == "stp")
        shape = OpenCASCADE::read_STEP(cad_file_name);
      else
        AssertThrow(false,
                    ExcNotImplemented("We found an extension that we "
                                      "do not recognize as a CAD file "
                                      "extension. Bailing out."));

      // Now we check how many faces are contained in the `Shape`. OpenCASCADE
      // is intrinsically 3D, so if this number is zero, we interpret this as
      // a line manifold, otherwise as a
      // OpenCASCADE::NormalToMeshProjectionManifold in `spacedim` = 3, or
      // OpenCASCADE::NURBSPatchManifold in `spacedim` = 2.
      const auto n_elements = OpenCASCADE::count_elements(shape);
      if ((std::get<0>(n_elements) == 0))
        tria.set_manifold(
          manifold_id,
          OpenCASCADE::ArclengthProjectionLineManifold<dim, spacedim>(shape));
      else if (spacedim == 3)
        {
          // We use this trick, because
          // OpenCASCADE::NormalToMeshProjectionManifold is only implemented
          // for spacedim = 3. The check above makes sure that things actually
          // work correctly.
          const auto t = reinterpret_cast<Triangulation<dim, 3> *>(&tria);
          t->set_manifold(manifold_id,
                          OpenCASCADE::NormalToMeshProjectionManifold<dim, 3>(
                            shape));
        }
      else
        // We also allow surface descriptions in two dimensional spaces based
        // on single NURBS patches. For this to work, the CAD file must
        // contain a single `TopoDS_Face`.
        tria.set_manifold(manifold_id,
                          OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                            TopoDS::Face(shape)));
    }
#else
  (void)ids_and_cad_file_names;
  AssertThrow(false, ExcNotImplemented("Generation of the grid failed."));
#endif
}
#endif