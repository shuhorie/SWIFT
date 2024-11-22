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
#include "hashmap.h"

/* Constants. */
#define FOF_CLOUD_COMPRESS_PATHS_MIN_LENGTH (2)

/*! Offset between the first particle on this MPI rank and the first particle in
 * the global order */
size_t node_offset_cloud;

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

  props->l_x2 = props->l_x_absolute * props->l_x_absolute;

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
 * @brief Mapper function to set the initial group indices.
 *
 * This is exactly the same as fof_set_initial_group_index_mapper()
 * in fof.c.
 *
 * @param map_data The array of group indices.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to first group index.
 */
void fof_cloud_set_initial_group_index_mapper(void *map_data, int num_elements,
                                              void *extra_data) {
  size_t *group_index = (size_t *)map_data;
  size_t *group_index_start = (size_t *)extra_data;

  const ptrdiff_t offset = group_index - group_index_start;

  for (int i = 0; i < num_elements; ++i) {
    group_index[i] = i + offset;
  }
}

/**
 * @brief Mapper function to set the initial distances.
 *
 * This is exactly the same as fof_set_initial_part_distances_mapper()
 * in fof.c.
 *
 * @param map_data The array of distance.
 * @param num_elements Chunk size.
 * @param extra_data N/A.
 */
void fof_cloud_set_initial_part_distances_mapper(void *map_data,
                                                 int num_elements,
                                                 void *extra_data) {

  float *distance = (float *)map_data;
  for (int i = 0; i < num_elements; ++i) {
    distance[i] = FLT_MAX;
  }
}

/**
 * @brief Mapper function to set the initial group sizes.
 *
 * This is exactly the same as fof_set_initial_group_size_mapper()
 * in fof.c.
 *
 * @param map_data The array of group sizes.
 * @param num_elements Chunk size.
 * @param extra_data N/A.
 */
void fof_cloud_set_initial_group_size_mapper(void *map_data, int num_elements,
                                             void *extra_data) {

  size_t *group_size = (size_t *)map_data;
  for (int i = 0; i < num_elements; ++i) {
    group_size[i] = 1;
  }
}

/**
 * @brief Allocate the memory and initialise the arrays for a FOF cloud calculation.
 *
 * @param s The #space to act on.
 * @param props The properties of the FOF cloud structure.
 */
void fof_cloud_allocate(const struct space *s, struct fof_cloud_props *props) {

  const int verbose = s->e->verbose;
  const ticks total_tic = getticks();

#ifdef WITH_MPI
  /* Check size of linking length against the top-level cell dimensions. */
  if (props->l_x2 > s->width[0] * s->width[0])
    error(
        "Linking length for FoF cloud is greater than the width of a top-level"
        "cell. Need to check more than one layer of top-level cells for links.");
#endif

  /* Allocate and initialise a group index array. */
  if (swift_memalign("fof_group_index", (void **)&props->group_index, 64,
                     s->nr_gparts * sizeof(size_t)) != 0)
    error("Failed to allocate list of particle group indices for FoF cloud search.");

  /* Allocate and initialise the closest distance array. */
  if (swift_memalign("fof_distance", (void **)&props->distance_to_link, 64,
                     s->nr_gparts * sizeof(float)) != 0)
    error(
        "Failed to allocate list of particle distances array for FoF cloud search.");

  /* Allocate and initialise a group size array. */
  if (swift_memalign("fof_group_size", (void **)&props->group_size, 64,
                     s->nr_gparts * sizeof(size_t)) != 0)
    error("Failed to allocate list of group size for FoF cloud search.");

  ticks tic = getticks();

  /* Set initial group index */
  threadpool_map(&s->e->threadpool, fof_cloud_set_initial_group_index_mapper,
                 props->group_index, s->nr_parts, sizeof(size_t),
                 threadpool_auto_chunk_size, props->group_index);

  if (verbose)
    message("Setting initial group index took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  tic = getticks();

  /* Set initial distances */
  threadpool_map(&s->e->threadpool, fof_cloud_set_initial_part_distances_mapper,
                 props->distance_to_link, s->nr_parts, sizeof(float),
                 threadpool_auto_chunk_size, NULL);

  if (verbose)
    message("Setting initial distances took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  tic = getticks();

  /* Set initial group sizes */
  threadpool_map(&s->e->threadpool, fof_cloud_set_initial_group_size_mapper,
                 props->group_size, s->nr_parts, sizeof(size_t),
                 threadpool_auto_chunk_size, NULL);

  if (verbose)
    message("Setting initial group sizes took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

#ifdef SWIFT_DEBUG_CHECKS
  ti_current = s->e->ti_current;
#endif

  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - total_tic),
            clocks_getunit());
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
 * @brief Compute th minimal distance between any two points in two cells.
 *
 * This is exactly the same as cell_min_dist() in fof.c
 *
 * @param ci The first #cell.
 * @param cj The second #cell.
 * @param dim The size of the simulation domain.
 */
__attribute__((always_inline)) INLINE static double cell_min_dist_fof_cloud(
    const struct cell *restrict ci, const struct cell *restrict cj,
    const double dim[3]) {

  /* Get cell locations. */
  const double cix_min = ci->loc[0];
  const double ciy_min = ci->loc[1];
  const double ciz_min = ci->loc[2];
  const double cjx_min = cj->loc[0];
  const double cjy_min = cj->loc[1];
  const double cjz_min = cj->loc[2];

  const double cix_max = ci->loc[0] + ci->width[0];
  const double ciy_max = ci->loc[1] + ci->width[1];
  const double ciz_max = ci->loc[2] + ci->width[2];
  const double cjx_max = cj->loc[0] + cj->width[0];
  const double cjy_max = cj->loc[1] + cj->width[1];
  const double cjz_max = cj->loc[2] + cj->width[2];

  double not_same_range[3];

  /* If two cells are in the same range of coordinates along
     any of the 3 axis, the distance along this axis is 0 */
  if (ci->width[0] > cj->width[0]) {
    if ((cix_min <= cjx_min) && (cjx_max <= cix_max))
      not_same_range[0] = 0.;
    else
      not_same_range[0] = 1.;
  } else {
    if ((cjx_min <= cix_min) && (cix_max <= cjx_max))
      not_same_range[0] = 0.;
    else
      not_same_range[0] = 1.;
  }
  if (ci->width[1] > cj->width[1]) {
    if ((ciy_min <= cjy_min) && (cjy_max <= ciy_max))
      not_same_range[1] = 0.;
    else
      not_same_range[1] = 1.;
  } else {
    if ((cjy_min <= ciy_min) && (ciy_max <= cjy_max))
      not_same_range[1] = 0.;
    else
      not_same_range[1] = 1.;
  }
  if (ci->width[2] > cj->width[2]) {
    if ((ciz_min <= cjz_min) && (cjz_max <= ciz_max))
      not_same_range[2] = 0.;
    else
      not_same_range[2] = 1.;
  } else {
    if ((cjz_min <= ciz_min) && (ciz_max <= cjz_max))
      not_same_range[2] = 0.;
    else
      not_same_range[2] = 1.;
  }

  /* Find the shortest distance between cells, remembering to account for
   * periodic boundary conditions. */
  double dx[3];
  dx[0] = min4(fabs(nearest(cix_min - cjx_min, dim[0])),
               fabs(nearest(cix_min - cjx_max, dim[0])),
               fabs(nearest(cix_max - cjx_min, dim[0])),
               fabs(nearest(cix_max - cjx_max, dim[0])));

  dx[1] = min4(fabs(nearest(ciy_min - cjy_min, dim[1])),
               fabs(nearest(ciy_min - cjy_max, dim[1])),
               fabs(nearest(ciy_max - cjy_min, dim[1])),
               fabs(nearest(ciy_max - cjy_max, dim[1])));

  dx[2] = min4(fabs(nearest(ciz_min - cjz_min, dim[2])),
               fabs(nearest(ciz_min - cjz_max, dim[2])),
               fabs(nearest(ciz_max - cjz_min, dim[2])),
               fabs(nearest(ciz_max - cjz_max, dim[2])));

  double r2 = 0.;
  for (int k = 0; k < 3; k++) r2 += dx[k] * dx[k] * not_same_range[k];

  return r2;
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
  if (c->split) error("Performing the FOF cloud search at a non-leaf level!");
#endif

  const size_t count = c->hydro.count;
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

// #ifdef SWIFT_DEBUG_CHECKS
//     if (pi->ti_drift != ti_current)
//       error("Running FOF on an un-drifted particle!");
// #endif

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

// #ifdef SWIFT_DEBUG_CHECKS
//       if (pj->ti_drift != ti_current)
//         error("Running FOF on an un-drifted particle!");
// #endif

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

        /* Merge the groups */
        fof_cloud_union(&root_i, root_j, group_index);
      }
    }
  }
}

/**
 * @brief Perform a FOF cloud search using union-find between two cells
 *
 * @param props The properties fof the FOF cloud scheme.
 * @param dim The dimension of the simulation volume.
 * @param l_x2 The square of the FOF cloud linking length.
 * @param periodic Are we using periodic BCs?
 * @param space_parts The start of the #part array in the #space structure.
 * @param ci The first #cell in which to perform FOF cloud.
 * @param cj The second #cell in which to perform FOF cloud.
 */
void fof_cloud_search_pair_cells(const struct fof_cloud_props *props,
                                 const double dim[3], const double l_x2,
                                 const int periodic,
                                 const struct part *const space_parts,
                                 const struct cell *restrict ci,
                                 const struct cell *restrict cj) {

  const size_t count_i = ci->hydro.count;
  const size_t count_j = cj->hydro.count;
  const struct part *parts_i = ci->hydro.parts;
  const struct part *parts_j = cj->hydro.parts;

  /* Index of particles in the global group list */
  size_t *const group_index = props->group_index;

  /* Make a list of particle offsets into the global parts array. */
  size_t *const offset_i = group_index + (ptrdiff_t)(parts_i - space_parts);
  size_t *const offset_j = group_index + (ptrdiff_t)(parts_j - space_parts);

#ifdef SWIFT_DEBUG_CHECKS
  if (offset_j > offset_i && (offset_j < offset_i + count_i))
    error("Overlapping cells");
  if (offset_i > offset_j && (offset_i < offset_j + count_j))
    error("Overlapping cells");
  if (ci->nodeID != cj->nodeID) error("Searching foreign cells!");
#endif

  /* Account for boundary conditions.*/
  double shift[3] = {0.0, 0.0, 0.0};

  /* Get the relative distance between the pairs, wrapping. */
  double diff[3];
  for (int k = 0; k < 3; k++) {
    diff[k] = cj->loc[k] - ci->loc[k];
    if (periodic && diff[k] < -dim[k] * 0.5)
      shift[k] = dim[k];
    else if (periodic && diff[k] > dim[k] * 0.5)
      shift[k] = -dim[k];
    else
      shift[k] = 0.0;
    diff[k] += shift[k];
  }

  /* Loop over particles and find which particles belong in the same group. */
  for (size_t i = 0; i < count_i; i++) {

    const struct part *restrict pi = &parts_i[i];

    /* Ignore inhibited particles */
    if (pi->time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (pi->rho < props->rho_min) continue;

// #ifdef SWIFT_DEBUG_CHECKS
//     if (pi->ti_drift != ti_current)
//       error("Running FOF on an un-drifted particle!");
// #endif

    const double pix = pi->x[0] - shift[0];
    const double piy = pi->x[1] - shift[1];
    const double piz = pi->x[2] - shift[2];

    /* Find the root of pi. */
    size_t root_i = fof_cloud_find(offset_i[i], group_index);

    for (size_t j = i + 1; j < count_j; j++) {

      const struct part *restrict pj = &parts_j[j];

      /* Ignore inhibited particles */
      if (pi->time_bin >= time_bin_inhibited) continue;

      /* Check density threshold */
      if (pj->rho < props->rho_min) continue;

      /* Find the root of pj. */
      const size_t root_j = fof_cloud_find(offset_j[j], group_index);

      /* Skip particles in the same group. */
      if (root_i == root_j) continue;

      const double pjx = pj->x[0];
      const double pjy = pj->x[1];
      const double pjz = pj->x[2];

      /* Compute pairwise distance (periodic BCs were accounted
       for by the shift vector) */
      float dx[3], r2 = 0.0f;
      dx[0] = pix - pjx;
      dx[1] = piy - pjy;
      dx[2] = piz - pjz;

      for (int k = 0; k < 3; k++) r2 += dx[k] * dx[k];

      /* Hit or miss? */
      if (r2 < l_x2) {

        /* Merge the groups */
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
 * @param space_parts The start of the #part array in the #space structure.
 * @param ci The first #cell in which to perform FOF cloud.
 * @param cj The second #cell in which to perform FOF cloud.
 */
void rec_fof_cloud_search_pair(const struct fof_cloud_props *props,
                               const double dim[3], const double search_r2,
                               const int periodic,
                               const struct part *const space_parts,
                               struct cell *restrict ci, struct cell *restrict cj) {

  /* Find the shortest distance between cells, remembering to account for
   * boundary conditions. */
  const double r2 = cell_min_dist_fof_cloud(ci, cj, dim);

#ifdef SWIFT_DEBUG_CHECKS
  if (ci == cj) error("Pair FoF cloud called on same cell!!!");
#endif

  /* Return if cells are out of range of each other. */
  if (r2 > search_r2) return;

  /* Recurse on both cells if they are both split. */
  if (ci->split && cj->split) {
    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL) {

        for (int l = 0; l < 8; l++)
          if (cj->progeny[l] != NULL)
            rec_fof_cloud_search_pair(props, dim, search_r2, periodic, space_parts,
                                      ci->progeny[k], cj->progeny[l]);
      }
    }
  } else if (ci->split) {
    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL)
        rec_fof_cloud_search_pair(props, dim, search_r2, periodic, space_parts,
                                  ci->progeny[k], cj);
    }
  } else if (cj->split) {
    for (int k = 0; k < 8; k++) {
      if (cj->progeny[k] != NULL)
        rec_fof_cloud_search_pair(props, dim, search_r2, periodic, space_parts, ci,
                                  cj->progeny[k]);
    }
  } else {
    /* Perform FOF cloud search between pairs of cells that are within the linking
     * length and not the same cell. */
    fof_cloud_search_pair_cells(props, dim, search_r2, periodic, space_parts, ci,
                                cj);
  }
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

/* Mapper function to atomically update the group size array.
 *
 * This is exactly the same as fof_update_group_size_mapper() in fof.c.
 */
void fof_cloud_update_group_size_mapper(hashmap_key_t key, hashmap_value_t *value,
                                        void *data) {

  size_t *group_size = (size_t *)data;

  /* Use key to index into group size array. */
  atomic_add(&group_size[key], value->value_st);
}

/**
 * @brief Mapper function to calculate the group sizes of clouds.
 *
 * @param map_data An array of #part%s.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to a #space.
 */
void fof_calc_cloud_group_size_mapper(void *map_data, int num_elements,
                                      void *extra_data) {

  /* Retrieve mapped data. */
  struct space *s = (struct space *)extra_data;
  struct part *parts = (struct part *)map_data;
  size_t *restrict group_index = s->e->fof_cloud_properties->group_index;
  size_t *restrict group_size = s->e->fof_cloud_properties->group_size;

  /* Offset into gparts array. */
  const ptrdiff_t parts_offset = (ptrdiff_t)(parts - s->parts);
  size_t *const group_index_offset = group_index + parts_offset;

  /* Create hash table. */
  hashmap_t map;
  hashmap_init(&map);

  for (int ind = 0; ind < num_elements; ind++) {

    const hashmap_key_t root =
        (hashmap_key_t)fof_cloud_find(group_index_offset[ind], group_index);
    const size_t part_index = parts_offset + ind;

    /* Only add particles which aren't the root of a group. Stops groups of size
     * 1 being added to the hash table. */
    if (root != part_index) {
      hashmap_value_t *size = hashmap_get(&map, root);

      if (size != NULL)
        (*size).value_st++;
      else
        error("Couldn't find key (%zu) or create new one.", root);
    }
  }

  /* Update the group size array. */
  if (map.size > 0)
    hashmap_iterate(&map, fof_cloud_update_group_size_mapper, group_size);

  hashmap_free(&map);
}

/**
 * @brief Compute the local size of each FOF cloud group fragment.
 *
 * @param props The properties of the FOF cloud scheme.
 * @param s The #space containing the particles.
 */
void fof_cloud_compute_local_sizes(struct fof_cloud_props *props,
                                   struct space *s) {

  const int verbose = s->e->verbose;

  struct part *parts = s->parts;
  const size_t nr_parts = s->nr_parts;

  const ticks tic_total = getticks();

  if (engine_rank == 0 && verbose)
    message("Size of hash table element: %ld", sizeof(hashmap_element_t));

#ifdef WITH_MPI

  const ticks comms_tic = getticks();

  /* Determine number of parts on lower numbwer MPI ranks */
  const long long nr_parts_local = s->nr_parts;
  long long nr_parts_cumulative;
  MPI_Scan(&nr_parts_local, &nr_parts_cumulative, 1, MPI_LONG_LONG, MPI_SUM,
           MPI_COMM_WORLD);

  if (verbose)
    message("MPI_Scan Imbalance took: %.3f %s.",
            clocks_from_ticks(getticks() - comms_tic), clocks_getunit());

  /* Reset global variable containing the rank particle count offset */
  node_offset_cloud = nr_parts_cumulative - nr_parts_local;
#endif /* WITH_MPI */

  /* Compute the group sizes of the local fragments
   * (in non-MPI land that is the final group size of the clouds) */
  const ticks tic_calc_group_size = getticks();

  threadpool_map(&s->e->threadpool, fof_calc_cloud_group_size_mapper, parts,
                 nr_parts, sizeof(struct part), threadpool_auto_chunk_size,
                 s);
  if (verbose)
    message("FOF cloud calc group size took (FOF_CLOUD SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic_calc_group_size),
            clocks_getunit());

  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic_total),
            clocks_getunit());
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

  swift_free("fof_group_index", props->group_index);
  swift_free("fof_distance", props->distance_to_link);
  swift_free("fof_group_size", props->group_size);
}

#endif /* WITH_FOF_CLOUD */
