/*******************************************************************************
 * This file is part of SWIFT.
 * This file is developed for the on-the-fly cloud finding with FoF
 * based on Horie+2024.
 * Copyright (c) 2024 Shu Horie (shorie@ccs.tsukuba.ac.jp).
 ******************************************************************************/

/* Config parameters. */
#include <config.h>

#ifdef WITH_FOF_CLOUD

/* Some standard headers. */
#include <errno.h>
#include <libgen.h>
#include <unistd.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* This object's header. */
#include "fof_cloud.h"

/* Local includes. */
#include "inline.h"
#include "timers.h"

/* Constants. */
#define FOF_CLOUD_COMPRESS_PATHS_MIN_LENGTH (2)

/**
 * @brief Initialise the properties of the FOF code for cloud finding.
 *
 * @param props the #fof_cloud_props structure to fill.
 * @param params the parameter file parser.
 * @param phys_const The physical constants in internal units.
 * @param us The internal unit system.
 */
void fof_cloud_init(struct fof_cloud_props *props,
                    struct swift_params *params,
                    const struct phys_const *phys_const,
                    const struct unit_system *us) {

  /* Main operation modes ------------------------------------------------- */

  props->l_x_absolute =
      parser_get_param_double(params, "FOFCloud:Linking_Length_in_cgs") /
      units_cgs_conversion_factor(us, UNIT_CONV_LENGTH);

  props->rho_min =
      parser_get_param_double(params, "FOFCloud:Density_Threshold_in_cgs") /
      units_cgs_conversion_factor(us, UNIT_CONV_DENSITY);

  props->min_group_size =
      parser_get_param_int(params, "FOFCloud:min_group_size");


  if (engine_rank == 0) {
    message("Properties of FoF for cloud finding (code units)");
    message("LinkingLength          = %g", props->l_x_absolute);
    message("Density Threshold      = %g", props->rho_min);
    message("Min group size         = %d", props->min_group_size);
  }

}

/**
 * @brief Allocate the memory and initialise the arrays for a FOF cloud calculation.
 *
 * @param s The #space to act on.
 * @param props The properties of the FOF cloud structure.
 */
void fof_cloud_allocate(const struct space *s, struct fof_cloud_props *props) {

}

/**
 * @brief Finds the local root ID of the group a particle exists in.
 *
 * We follow the group_index array until reaching the root of the group.
 *
 * Also performs path compression if the path is long.
 *
 * This is almost the same as fof_find() in fof.c
 *
 * @param i The index of the particle.
 * @param group_index Array of group root indices.
 */
__attribute__((always_inline)) INLINE static size_t fof_cloud_find(
    const size_t i, size_t *group_index) {

  size_t root = i;
  int tree_depth = 0;

  while (root != group_index[root]) {
#ifdef PATH_HALVING
    atomic_cas(&group_index[root], group_index[root],
               group_index[group_index[root]]);
#endif
    root = group_index[root];
    tree_depth++;
  }

  /* Only perform path compression on trees with a depth of
   * FOF_CLOUD_COMPRESS_PATHS_MIN_LENGTH or higher. */
  if (tree_depth >= FOF_CLOUD_COMPRESS_PATHS_MIN_LENGTH)
    atomic_cas(&group_index[i], group_index[i], root);

  return root;
}

/**
 * @brief Atomically update the root of a group
 *
 * This is exactly the same as atomic_update_root() in fof.c
 *
 * @param address The address of the value to update.
 * @param y The new value to write.
 *
 * @return 1 If successful, 0 otherwise.
 */
__attribute__((always_inline)) INLINE static int atomic_update_root_fof_cloud(
    volatile size_t *address, const size_t y) {

  size_t *size_t_ptr = (size_t *)address;

  size_t old_val = *address;
  size_t test_val = old_val;
  size_t new_val = y;

  /* atomic_cas returns old_val if *size_t_ptr has not changed since being
   * read.*/
  old_val = atomic_cas(size_t_ptr, test_val, new_val);

  if (test_val == old_val)
    return 1;
  else
    return 0;
}

/**
 * @brief Unifies two groups by setting them to the same root.
 *
 * This is almost the same as fof_union() in fof.c
 *
 * @param root_i The root of the first group. Will be updated.
 * @param root_j The root of the second group.
 * @param group_index The list of group roots.
 */
__attribute__((always_inline)) INLINE static void fof_cloud_union(
    size_t *restrict root_i, const size_t root_j,
    size_t *restrict group_index) {

  int result = 0;

  /* Loop until the root can be set to a new value. */
  do {
    size_t root_i_new = fof_cloud_find(*root_i, group_index);
    const size_t root_j_new = fof_cloud_find(root_j, group_index);

    /* Skip particles in the same group. */
    if (root_i_new == root_j_new) return;

    /* If the root ID of pj is lower than pi's root ID set pi's root to point to
     * pj's. Otherwise set pj's root to point to pi's. */
    if (root_j_new < root_i_new) {

      /* Updates the root and checks that its value has not been changed since
       * being read. */
      result = atomic_update_root_fof_cloud(&group_index[root_i_new], root_j_new);

      /* Update root_i on the fly. */
      *root_i = root_j_new;
    } else {

      /* Updates the root and checks that its value has not been changed since
       * being read. */
      result = atomic_update_root_fof_cloud(&group_index[root_j_new], root_i_new);

      /* Update root_i on the fly. */
      *root_i = root_i_new;
    }
  } while (result != 1);
}

/**
 * @brief Perform a FOF cloud search using union-find on a given leaf-cell
 *
 * @param props The properties fof the FOF cloud scheme.
 * @param l_x2 The square of the FOF cloud linking length.
 * @param space_parts The start of the #part array in the #space structure.
 * @param c The #cell in which to perform FOF cloud.
 */
void fof_cloud_search_self_cell(const struct fof_cloud_props *props,
                                const double l_x2,
                                const struct part *const space_parts,
                                const struct cell *c) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->split) error("Performing the FOF search at a non-leaf level!");
#endif

  const size_t count = c->grav.count;
  const struct part *parts = c->hydro.parts;

  /* Index of particles in the global group list */
  size_t *const group_index = props->group_index;

  /* Make a list of particle offsets into the global parts array. */
  size_t *const offset = group_index + (ptrdiff_t)(parts - space_parts);

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != engine_rank)
    error("Performing self FOF search on foreign cell.");
#endif

  /* Loop over particles and find which particles belong in the same group. */
  for (size_t i = 0; i < count; i++) {

    const struct part *pi = &parts[i];

    /* Ignore inhibited particles */
    if (pi->time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (pi->rho < props->rho_min) continue;

#ifdef SWIFT_DEBUG_CHECKS
    if (pi->ti_drift != ti_current)
      error("Running FOF on an un-drifted particle!");
#endif

    const double pix = pi->x[0];
    const double piy = pi->x[1];
    const double piz = pi->x[2];

    /* Find the root of pi. */
    size_t root_i = fof_cloud_find(offset[i], group_index);

    for (size_t j = i + 1; j < count; j++) {

      const struct part *pj = &parts[j];

      /* Ignore inhibited particles */
      if (pj->time_bin >= time_bin_inhibited) continue;

      /* Check density threshold */
      if (pj->rho < props->rho_min) continue;

#ifdef SWIFT_DEBUG_CHECKS
      if (pj->ti_drift != ti_current)
        error("Running FOF on an un-drifted particle!");
#endif

      /* Find the root of pj. */
      const size_t root_j = fof_cloud_find(offset[j], group_index);

      /* Skip particles in the same group. */
      if (root_i == root_j) continue;

      const double pjx = pj->x[0];
      const double pjy = pj->x[1];
      const double pjz = pj->x[2];

      /* Compute the pairwise distance */
      float dx[3], r2 = 0.0f;
      dx[0] = pix - pjx;
      dx[1] = piy - pjy;
      dx[2] = piz - pjz;

      for (int k = 0; k < 3; k++) r2 += dx[k] * dx[k];

      /* Hit or miss? */
      if (r2 < l_x2) {

        /* Merge the groups` */
        fof_cloud_union(&root_i, root_j, group_index);
      }
    }
  }
}


/**
 * @brief Recursively perform a union-find FOF cloud on a cell.
 *
 * @param props The properties fof the FOF cloud scheme.
 * @param dim The dimension of the space.
 * @param space_parts The start of the #part array in the #space structure.
 * @param search_r2 the square of the FOF cloud linking length.
 * @param periodic Are we using periodic BCs?
 * @param c The #cell in which to perform FOF cloud.
 */
void rec_fof_cloud_search_self(const struct fof_cloud_props *props,
                               const double dim[3], const double search_r2,
                               const int periodic,
                               const struct part *const space_parts,
                               struct cell *c) {

  /* Recurse? */
  if (c->split) {

    /* Loop over all progeny. Perform pair and self recursion on progenies.*/
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {

        rec_fof_cloud_search_self(props, dim, search_r2, periodic, space_parts,
                                  c->progeny[k]);

        for (int l = k + 1; l < 8; l++) {
          if (c->progeny[l] != NULL)
            rec_fof_cloud_search_pair(props, dim, search_r2, periodic, space_parts,
                                      c->progeny[k], c->progeny[l]);
        }
      }
    }
  }
  /* Otherwise, compute self-interaction. */
  else
    fof_cloud_search_self_cell(props, search_r2, space_parts, c);

}

/**
 * @brief Recursively perform a union-find FOF cloud between two cells.
 *
 * If cells are more distant than the linking length, we abort early.
 *
 * @param props The properties fof the FOF cloud scheme.
 * @param dim The dimension of the space.
 * @param search_r2 the square of the FOF cloud linking length.
 * @param periodic Are we using periodic BCs?
 * @param space_parts The start of the #gpart array in the #space structure.
 * @param ci The first #cell in which to perform FOF cloud.
 * @param cj The second #cell in which to perform FOF cloud.
 */
void rec_fof_cloud_search_pair(const struct fof_cloud_props *props,
                               const double dim[3], const double search_r2,
                               const int periodic, const struct part *const space_parts,
                               struct cell *restrict ci, struct cell *restrict cj) {

  /* Find the shortest distance between cells, remembering to account for
   * boundary conditions. */

}

/**
 * @brief Search foreign cells for links and communicate any found to the
 * appropriate node.
 *
 * @param props the properties of the FOF cloud scheme.
 * @param s Pointer to a #space.
 */
void fof_cloud_search_foreign_cells(struct fof_cloud_props *props,
                                    const struct space *s) {

#ifdef WITH_MPI
  printf("fof_cloud_search_foreign_cells\n");
#endif /* WITH_MPI */
}

/**
 * @brief Compute the local size of each FOF cloud group fragment.
 *
 * @param props The properties of the FOF cloud scheme.
 * @param s The #space containing the particles.
 */
void fof_cloud_compute_local_sizes(struct fof_cloud_props *props,
                                   struct space *s) {

//   const int verbose = s->e->verbose;

  printf("fof_cloud_compute_local_sizes\n");
}

/**
 * @brief Process all the group fragments spanning more than
 * one rank to link them.
 *
 * This is the final global union-find pass which concludes
 * the MPI-FOF-algorithm.
 *
 * @param props The properties fof the FOF cloud scheme.
 * @param s The #space we work with.
 */
void fof_cloud_link_foreign_fragments(struct fof_cloud_props *props,
                                      const struct space *s) {

#ifdef WITH_MPI
  printf("fof_cloud_link_foreign_fragments\n");

#endif /* WITH_MPI */
}

/**
 * @brief Compute all the group properties
 *
 * @param props The properties of the FOF cloud scheme.
 * @param constants The physical constants in internal units.
 * @param cosmo The current cosmological model.
 * @param s The #space containing the particles.
 */
void fof_cloud_compute_group_props(struct fof_cloud_props *props,
                                   const struct phys_const *constants,
                                   const struct cosmology *cosmo,
                                   struct space *s) {

//   const int verbose = s->e->verbose;
  printf("fof_cloud_group_props\n");
}

#endif /* WITH_FOF_CLOUD */
