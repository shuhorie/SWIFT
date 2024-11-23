/*******************************************************************************
 * This file is part of SWIFT.
 * This file is developed for the on-the-fly cloud finding with FoF
 * based on Horie+2024.
 * Copyright (c) 2024 Shu Horie (shorie@ccs.tsukuba.ac.jp).
 ******************************************************************************/
#ifndef SWIFT_FOF_CLOUD_H
#define SWIFT_FOF_CLOUD_H

/* Config parameters. */
#include <config.h>

/* Local headers */
#include "cosmology.h"
#include "engine.h"
#include "error.h"
#include "part.h"
#include "units.h"
#include "align.h"
#include "part_type.h"

struct fof_cloud_props {

  /* ----------- Parameters of the FOF search ------- */

  /*! The absolute linking length in internal units. */
  double l_x_absolute;

  /*! The square of the linking length. */
  double l_x2;

  /*! The minimum gas density of a candidate particle that is a member of a cloud */
  double rho_min;

  /*! Minimal number of particles in a group */
  int min_group_size;

  /*! Default group ID to give to particles not in a group */
  size_t group_id_default;

  /*! ID of the first (largest) group. */
  size_t group_id_offset;

  /*! The types of particles to use for linking */
  int fof_linking_types[swift_type_count];

  /* ------------  Group properties ----------------- */

  /*! Number of groups */
  long long num_groups;

  /*! Number of local black holes that belong to groups whose roots are on a
   * different node. */
  int extra_bh_seed_count;

  /*! Index of the root particle of the group a given gpart belongs to. */
  size_t *group_index;

  /*! Is the group purely local after linking the foreign particles? */
  char *is_purely_local;

  /*! For attachable particles: distance to the current nearest linkable part */
  float *distance_to_link;

  /*! Size of the group a given gpart belongs to. */
  size_t *group_size;

  /*! Final size of the group a given gpart belongs to. */
  long long *final_group_size;

  /*! Mass of the group a given gpart belongs to. */
  double *group_mass;

  /*! Centre of mass of the group a given gpart belongs to. */
  double *group_centre_of_mass;

  /*! Position of the first particle of a given group. */
  double *group_first_position;

  /*! Index of the part with the maximal density of each group. */
  long long *max_part_density_index;

  /*! Maximal density of all parts of each group. */
  float *max_part_density;

  /* ------------ MPI-related arrays --------------- */

  /*! The number of links between pairs of particles on this node and
   * a foreign node */
  int group_link_count;

  /*! The allocated size of the links array */
  int group_links_size;

  /*! The links between pairs of particles on this node and a foreign
   * node */
  struct fof_cloud_mpi *group_links;
};

/* Store group size and offset into array. */
struct cloud_group_length {

  size_t index, size;

} SWIFT_STRUCT_ALIGN;

#ifdef WITH_MPI

/* MPI message required for FOF. */
struct fof_cloud_mpi {

  /* The local particle's root ID.*/
  size_t group_i;

  /* The local group's size.*/
  size_t group_i_size;

  /* The foreign particle's root ID.*/
  size_t group_j;

  /* The local group's size.*/
  size_t group_j_size;
};

/* Struct used to find final group ID when using MPI */
struct fof_cloud_final_index {
  size_t local_root;
  size_t global_root;
};

/* Struct used to find the total mass of a group when using MPI */
struct fof_cloud_final_mass {
  size_t global_root;
  double group_mass;
  long long final_group_size;
  double first_position[3];
  double centre_of_mass[3];
  long long max_part_density_index;
  float max_part_density;
};

/* Struct used to iterate over the hash table and unpack the mass fragments of a
 * group when using MPI */
struct fof_cloud_mass_send_hashmap {
  struct fof_final_mass *mass_send;
  size_t nsend;
};

/* Store local and foreign cell indices that touch. */
struct cloud_cell_pair_indices {
  struct cell *local, *foreign;
};
#endif /* WITH_MPI */

/* Function prototypes. */
void fof_cloud_init(struct fof_cloud_props *fcp,
                    struct swift_params *params,
                    const struct phys_const *phys_const,
                    const struct unit_system *us);
void fof_cloud_allocate(const struct space *s, struct fof_cloud_props *props);
void fof_cloud_search_self_cell(const struct fof_cloud_props *props,
                                const double l_x2,
                                const struct part *const space_parts,
                                const struct cell *c);
void fof_cloud_search_pair_cells(const struct fof_cloud_props *props,
                                 const double dim[3], const double l_x2,
                                 const int periodic,
                                 const struct part *const space_parts,
                                 const struct cell *restrict ci,
                                 const struct cell *restrict cj);
void rec_fof_cloud_search_self(const struct fof_cloud_props *props,
                               const double dim[3], const double search_r2,
                               const int periodic,
                               const struct part *const space_parts,
                               struct cell *c);
void rec_fof_cloud_search_pair(const struct fof_cloud_props *props,
                               const double dim[3], const double search_r2,
                               const int periodic, const struct part *const space_parts,
                               struct cell *restrict ci, struct cell *restrict cj);
void fof_cloud_search_foreign_cells(struct fof_cloud_props *props,
                                    const struct space *s);
void fof_cloud_compute_local_sizes(struct fof_cloud_props *props,
                                   struct space *s);
void fof_cloud_link_foreign_fragments(struct fof_cloud_props *props,
                                      const struct space *s);
void fof_cloud_compute_group_props(struct fof_cloud_props *props,
                                   const struct phys_const *constants,
                                   const struct cosmology *cosmo,
                                   struct space *s);

#endif /* SWIFT_FOF_CLOUD_H */