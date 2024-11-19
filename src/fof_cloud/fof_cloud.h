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

  /*! The types of particles to use for attaching */
  int fof_attach_types[swift_type_count];

  /* ------------  Group properties ----------------- */

  /*! Number of groups */
  long long num_groups;

  /*! Number of local black holes that belong to groups whose roots are on a
   * different node. */
  int extra_bh_seed_count;

  /*! Index of the root particle of the group a given gpart belongs to. */
  size_t *group_index;

  /*! Index of the root particle of the group a given gpart is attached to. */
  size_t *attach_index;

  /*! Has the particle found a linkable to attach to? */
  char *found_attachable_link;

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
  struct fof_mpi *group_links;
};

/* Function prototypes. */
void fof_cloud_init(struct fof_cloud_props *fcp,
                    struct swift_params *params,
                    const struct phys_const *phys_const,
                    const struct unit_system *us);
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