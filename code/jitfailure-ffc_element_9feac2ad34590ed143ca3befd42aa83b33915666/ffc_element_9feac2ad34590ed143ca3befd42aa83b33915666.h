// This code conforms with the UFC specification version 2018.1.0
// and was automatically generated by FFC version 2018.1.0.
//
// This code was generated with the following parameters:
//

//  add_tabulate_tensor_timing:     False
//  convert_exceptions_to_warnings: False
//  cpp_optimize:                   True
//  cpp_optimize_flags:             '-O2'
//  epsilon:                        1e-14
//  error_control:                  False
//  external_include_dirs:          ''
//  external_includes:              ''
//  external_libraries:             ''
//  external_library_dirs:          ''
//  form_postfix:                   False
//  format:                         'ufc'
//  generate_dummy_tabulate_tensor: False
//  max_signature_length:           0
//  no-evaluate_basis_derivatives:  True
//  optimize:                       True
//  precision:                      None
//  quadrature_degree:              None
//  quadrature_rule:                None
//  representation:                 'auto'
//  split:                          False

#ifndef __FFC_ELEMENT_9FEAC2AD34590ED143CA3BEFD42AA83B33915666_H
#define __FFC_ELEMENT_9FEAC2AD34590ED143CA3BEFD42AA83B33915666_H
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <ufc.h>

class ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_finite_element_main: public ufc::finite_element
{
public:

  ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_finite_element_main();

  ~ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_finite_element_main() override;

  const char * signature() const final override;

  ufc::shape cell_shape() const final override;

  std::size_t topological_dimension() const final override;

  std::size_t geometric_dimension() const final override;

  std::size_t space_dimension() const final override;

  std::size_t value_rank() const final override;

  std::size_t value_dimension(std::size_t i) const final override;

  std::size_t value_size() const final override;

  std::size_t reference_value_rank() const final override;

  std::size_t reference_value_dimension(std::size_t i) const final override;

  std::size_t reference_value_size() const final override;

  std::size_t degree() const final override;

  const char * family() const final override;

  void evaluate_reference_basis(double * reference_values,
                                std::size_t num_points,
                                const double * X) const final override;

  void evaluate_reference_basis_derivatives(double * reference_values,
                                            std::size_t order,
                                            std::size_t num_points,
                                            const double * X) const final override;

  void transform_reference_basis_derivatives(double * values,
                                             std::size_t order,
                                             std::size_t num_points,
                                             const double * reference_values,
                                             const double * X,
                                             const double * J,
                                             const double * detJ,
                                             const double * K,
                                             int cell_orientation) const final override;

  void evaluate_basis(std::size_t i,
                      double * values,
                      const double * x,
                      const double * coordinate_dofs,
                      int cell_orientation,
                      const ufc::coordinate_mapping * cm=nullptr
                      ) const final override;

  void evaluate_basis_all(double * values,
                          const double * x,
                          const double * coordinate_dofs,
                          int cell_orientation,
                          const ufc::coordinate_mapping * cm=nullptr
                          ) const final override;

  void evaluate_basis_derivatives(std::size_t i,
                                  std::size_t n,
                                  double * values,
                                  const double * x,
                                  const double * coordinate_dofs,
                                  int cell_orientation,
                                  const ufc::coordinate_mapping * cm=nullptr
                                  ) const final override;

  void evaluate_basis_derivatives_all(std::size_t n,
                                      double * values,
                                      const double * x,
                                      const double * coordinate_dofs,
                                      int cell_orientation,
                                      const ufc::coordinate_mapping * cm=nullptr
                                      ) const final override;

  double evaluate_dof(std::size_t i,
                      const ufc::function& f,
                      const double * coordinate_dofs,
                      int cell_orientation,
                      const ufc::cell& c,
                      const ufc::coordinate_mapping * cm=nullptr
                      ) const final override;

  void evaluate_dofs(double * values,
                     const ufc::function& f,
                     const double * coordinate_dofs,
                     int cell_orientation,
                     const ufc::cell& c,
                     const ufc::coordinate_mapping * cm=nullptr
                     ) const final override;

  void interpolate_vertex_values(double * vertex_values,
                                 const double * dof_values,
                                 const double * coordinate_dofs,
                                 int cell_orientation,
                                 const ufc::coordinate_mapping * cm=nullptr
                                 ) const final override;

  void tabulate_dof_coordinates(double * dof_coordinates,
                                const double * coordinate_dofs,
                                const ufc::coordinate_mapping * cm=nullptr
                                ) const final override;

  void tabulate_reference_dof_coordinates(double * reference_dof_coordinates) const final override;

  std::size_t num_sub_elements() const final override;

  ufc::finite_element * create_sub_element(std::size_t i) const final override;

  ufc::finite_element * create() const final override;

};

extern "C" ufc::finite_element * create_ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_finite_element_main();


class ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_dofmap_main: public ufc::dofmap
{
public:

  ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_dofmap_main();

  ~ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_dofmap_main() override;

  const char * signature() const final override;

  bool needs_mesh_entities(std::size_t d) const final override;

  std::size_t topological_dimension() const final override;

  std::size_t global_dimension(const std::vector<std::size_t>&
                               num_global_entities) const final override;

  std::size_t num_global_support_dofs() const final override;

  std::size_t num_element_support_dofs() const final override;

  std::size_t num_element_dofs() const final override;

  std::size_t num_facet_dofs() const final override;

  std::size_t num_entity_dofs(std::size_t d) const final override;

  std::size_t num_entity_closure_dofs(std::size_t d) const final override;

  void tabulate_dofs(std::size_t * dofs,
                     const std::vector<std::size_t>& num_global_entities,
                     const std::vector<std::vector<std::size_t>>& entity_indices) const final override;

  void tabulate_facet_dofs(std::size_t * dofs,
                           std::size_t facet) const final override;

  void tabulate_entity_dofs(std::size_t * dofs,
                            std::size_t d, std::size_t i) const final override;

  void tabulate_entity_closure_dofs(std::size_t * dofs,
                            std::size_t d, std::size_t i) const final override;

  std::size_t num_sub_dofmaps() const final override;

  ufc::dofmap * create_sub_dofmap(std::size_t i) const final override;

  ufc::dofmap * create() const final override;

};

extern "C" ufc::dofmap * create_ffc_element_9feac2ad34590ed143ca3befd42aa83b33915666_dofmap_main();

#endif
