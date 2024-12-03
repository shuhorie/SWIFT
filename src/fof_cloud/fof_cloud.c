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
#include "proxy.h"

#define fof_cloud_props_default_group_id 2147483647
#define fof_cloud_props_default_group_id_offset 1
#define fof_cloud_props_default_group_link_size 20000

/* Constants. */
#define CLOUD_UNION_BY_SIZE_OVER_MPI (1)
#define FOF_CLOUD_COMPRESS_PATHS_MIN_LENGTH (2)

#ifdef WITH_MPI

/* MPI types used for communications */
MPI_Datatype fof_cloud_mpi_type;
MPI_Datatype cloud_group_length_mpi_type;
MPI_Datatype fof_cloud_final_index_type;
MPI_Datatype fof_cloud_final_mass_type;

/*! Offset between the first particle on this MPI rank and the first particle in
 * the global order */
size_t node_offset_cloud;
#endif /* WITH_MPI */

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

  /* Read value of absolute linking length asked by the user */
  props->l_x_absolute =
      parser_get_param_double(params, "FOFCloud:Linking_Length_in_cgs") /
      units_cgs_conversion_factor(us, UNIT_CONV_LENGTH);

  props->l_x2 = props->l_x_absolute * props->l_x_absolute;

  /* Read value of minimum hydro density asked by the user */
  props->rho_min =
      parser_get_param_double(params, "FOFCloud:Density_Threshold_in_cgs") /
      units_cgs_conversion_factor(us, UNIT_CONV_DENSITY);

  /* Read the minimum group size. */
  props->min_group_size =
      parser_get_param_int(params, "FOFCloud:min_group_size");

  /* Read the default group ID of particles in groups below the minimum group
   * size. */
  props->group_id_default = parser_get_opt_param_int(
      params, "FOFCloud:group_id_default", fof_cloud_props_default_group_id);

  /* Read the starting group ID. */
  props->group_id_offset = parser_get_opt_param_int(
      params, "FOFCloud:group_id_offset", fof_cloud_props_default_group_id_offset);

  if (props->l_x_absolute <= 0.)
    error("The FOF cloud linking length can't be negative!");

  if (props->rho_min <= 0.)
    error("The FOF cloud density threshold can't be negative!");

  if (props->min_group_size <= 0.)
    error("The FOF cloud group size can't be negative!");

  if (engine_rank == 0) {
    message("Properties of FoF for cloud finding (code units)");
    message("LinkingLength          = %g", props->l_x_absolute);
    message("Density Threshold      = %g", props->rho_min);
    message("Min group size         = %d", props->min_group_size);
  }

#if defined(WITH_MPI) && defined(UNION_BY_SIZE_OVER_MPI)
  if (engine_rank == 0)
    message(
        "Performing FOF cloud over MPI using union by size and union by rank "
        "locally.");
#else
  message("Performing FOF cloud using union by rank.");
#endif
}

/**
 * @brief Registers MPI types used by FOF cloud.
 */
void fof_cloud_create_mpi_types(void) {

#ifdef WITH_MPI
  if (MPI_Type_contiguous(sizeof(struct fof_cloud_mpi) / sizeof(unsigned char),
                          MPI_BYTE, &fof_cloud_mpi_type) != MPI_SUCCESS ||
      MPI_Type_commit(&fof_cloud_mpi_type) != MPI_SUCCESS) {
    error("Failed to create MPI type for fof_cloud.");
  }
  if (MPI_Type_contiguous(sizeof(struct cloud_group_length) / sizeof(unsigned char),
                          MPI_BYTE, &cloud_group_length_mpi_type) != MPI_SUCCESS ||
      MPI_Type_commit(&cloud_group_length_mpi_type) != MPI_SUCCESS) {
    error("Failed to create MPI type for cloud_group_length.");
  }
  /* Define type for sending fof_final_index struct */
  if (MPI_Type_contiguous(sizeof(struct fof_cloud_final_index), MPI_BYTE,
                          &fof_cloud_final_index_type) != MPI_SUCCESS ||
      MPI_Type_commit(&fof_cloud_final_index_type) != MPI_SUCCESS) {
    error("Failed to create MPI type for fof_cloud_final_index.");
  }
  /* Define type for sending fof_final_mass struct */
  if (MPI_Type_contiguous(sizeof(struct fof_cloud_final_mass), MPI_BYTE,
                          &fof_cloud_final_mass_type) != MPI_SUCCESS ||
      MPI_Type_commit(&fof_cloud_final_mass_type) != MPI_SUCCESS) {
    error("Failed to create MPI type for fof_cloud_final_mass.");
  }
#else
  error("Calling an MPI function in non-MPI code.");
#endif
}

#ifdef WITH_MPI

/**
 * @brief Check whether a given group ID is on the local node.
 *
 * This function only makes sense in MPI mode.
 *
 * @param group_id The ID to check.
 * @param nr_parts The number of parts on this node.
 */
__attribute__((always_inline)) INLINE static int is_local_fof_cloud(
    const size_t group_id, const size_t nr_parts) {
#ifdef WITH_MPI
  return (group_id >= node_offset_cloud && group_id < node_offset_cloud + nr_parts);
#else
  error("Calling MPI function in non-MPI mode");
  return 1;
#endif
}

/**
 * @brief Find the global root ID of a given particle
 *
 * This function only makes sense in MPI mode.
 *
 * @param i Index of the particle.
 * @param group_index Array of group root indices.
 * @param nr_parts The number of hydro-particles on this node.
 */
__attribute__((always_inline)) INLINE static size_t fof_cloud_find_global(
    const size_t i, const size_t *group_index, const size_t nr_parts) {

#ifdef WITH_MPI
  size_t root = node_offset_cloud + i;
  if (!is_local_fof_cloud(root, nr_parts)) {

    /* Non local --> This is the root */
    return root;
  } else {

    /* Local --> Follow the links until we find the root */
    while (root != group_index[root - node_offset_cloud]) {
      root = group_index[root - node_offset_cloud];
      if (!is_local_fof_cloud(root, nr_parts)) break;
    }
  }

  /* Perform path compression. */
  // int index = i;
  // while(index != root) {
  //  int next = group_index[index];
  //  group_index[index] = root;
  //  index = next;
  //}

  return root;
#else
  error("Calling MPI function in non-MPI mode");
  return -1;
#endif
}

#endif /* WITH_MPI */

/**
 * @brief   Finds the local root ID of the group a particle exists in
 * when group_index contains globally unique identifiers -
 * i.e. we stop *before* we advance to a foreign root.
 *
 * This is almost the same as fof_find_local() in fof.c.
 *
 * Here we assume that the input i is a local index and we
 * return the local index of the root.
 *
 * @param i Index of the particle.
 * @param nr_parts The number of hydro-particles on this node.
 * @param group_index Array of group root indices.
 */
__attribute__((always_inline)) INLINE static size_t fof_cloud_find_local(
    const size_t i, const size_t nr_parts, const size_t *group_index) {
#ifdef WITH_MPI
  size_t root = node_offset_cloud + i;

  while ((group_index[root - node_offset_cloud] != root) &&
         (group_index[root - node_offset_cloud] >= node_offset_cloud) &&
         (group_index[root - node_offset_cloud] < node_offset_cloud + nr_parts)) {
    root = group_index[root - node_offset_cloud];
  }

  return root - node_offset_cloud;
#else
  size_t root = i;

  while ((group_index[root] != root) && (group_index[root] < nr_parts)) {
    root = group_index[root];
  }

  return root;
#endif
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

  for (int i = 0; i < num_elements; i++) {
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
  for (int i = 0; i < num_elements; i++) {
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
  for (int i = 0; i < num_elements; i++) {
    group_size[i] = 1;
  }
}

/**
 * @brief Mapper function to set the initial group IDs.
 *
 * This is almost the same as fof_set_initial_group_id_mappe()
 * in fof.c.
 *
 * @param map_data The array of #part%s.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to the default group ID.
 */
void fof_cloud_set_initial_group_id_mapper(void *map_data, int num_elements,
                                           void *extra_data) {

  /* Unpack the information */
  struct part *parts = (struct part *)map_data;
  const size_t group_id_default = *((size_t *)extra_data);

  for (int i = 0; i < num_elements; ++i) {
    parts[i].fof_cloud_data.group_id = group_id_default;
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
  if (swift_memalign("fof_cloud_group_index", (void **)&props->group_index, 64,
                     s->nr_parts * sizeof(size_t)) != 0)
    error("Failed to allocate list of particle group indices for FoF cloud search.");

  /* Allocate and initialise the closest distance array. */
  if (swift_memalign("fof_cloud_distance", (void **)&props->distance_to_link, 64,
                     s->nr_parts * sizeof(float)) != 0)
    error(
        "Failed to allocate list of particle distances array for FoF cloud search.");

  /* Allocate and initialise a group size array. */
  if (swift_memalign("fof_cloud_group_size", (void **)&props->group_size, 64,
                     s->nr_parts * sizeof(size_t)) != 0)
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
 * @brief Comparison function for qsort call comparing group sizes.
 *
 * This is almost the same as cmp_func_group_size() in fof.c
 *
 * @param a The first #cloud_group_length object.
 * @param b The second #cloud_group_length object.
 * @return 1 if the size of the group b is larger than the size of group a, -1
 * if a is the largest and 0 if they are equal.
 */
int cmp_func_cloud_group_size(const void *a, const void *b) {
  struct cloud_group_length *a_group_size = (struct cloud_group_length *)a;
  struct cloud_group_length *b_group_size = (struct cloud_group_length *)b;
  if (b_group_size->size > a_group_size->size)
    return 1;
  else if (b_group_size->size < a_group_size->size)
    return -1;
  else
    return 0;
}

#ifdef WITH_MPI

/**
 * @brief Comparison function for qsort call comparing group global roots.
 *
 * This is almost the same as compare_fof_final_index_global_root() in fof.c
 *
 * @param a The first #fof_cloud_final_index object.
 * @param b The second #fof_cloud_final_index object.
 * @return 1 if the global of the group b is *smaller* than the global group of
 * group a, -1 if a is the smaller one and 0 if they are equal.
 */
int compare_fof_cloud_final_index_global_root(const void *a, const void *b) {
  struct fof_cloud_final_index *fof_final_index_a = (struct fof_cloud_final_index *)a;
  struct fof_cloud_final_index *fof_final_index_b = (struct fof_cloud_final_index *)b;
  if (fof_final_index_b->global_root < fof_final_index_a->global_root)
    return 1;
  else if (fof_final_index_b->global_root > fof_final_index_a->global_root)
    return -1;
  else
    return 0;
}

/**
 * @brief Comparison function for qsort call comparing group global roots
 *
 * This is almost the same as compare_fof_final_mass_global_root() in fof.c
 *
 * @param a The first #fof_cloud_final_mass object.
 * @param b The second #fof_cloud_final_mass object.
 * @return 1 if the global of the group b is *smaller* than the global group of
 * group a, -1 if a is the smaller one and 0 if they are equal.
 */
int compare_fof_cloud_final_mass_global_root(const void *a, const void *b) {
  struct fof_cloud_final_mass *fof_final_mass_a = (struct fof_cloud_final_mass *)a;
  struct fof_cloud_final_mass *fof_final_mass_b = (struct fof_cloud_final_mass *)b;
  if (fof_final_mass_b->global_root < fof_final_mass_a->global_root)
    return 1;
  else if (fof_final_mass_b->global_root > fof_final_mass_a->global_root)
    return -1;
  else
    return 0;
}

#endif

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

#ifdef WITH_MPI

/* Add a group to the hash table.
 * This is exactly the same as hashmap_add_group() in fof.c.
 */
__attribute__((always_inline)) INLINE static void hashmap_add_cloud_group(
    const size_t group_id, const size_t group_offset, hashmap_t *map) {

  int created_new_element = 0;
  hashmap_value_t *offset =
      hashmap_get_new(map, group_id, &created_new_element);

  if (offset != NULL) {

    /* If the element is a new entry set its value. */
    if (created_new_element) {
      (*offset).value_st = group_offset;
    }
  } else
    error("Couldn't find key (%zu) or create new one.", group_id);
}

/* Find a group in the hash table.
 * This is exactly the same as hashmap_find_group_offset() in fof.c.
 */
__attribute__((always_inline)) INLINE static size_t hashmap_find_cloud_group_offset(
    const size_t group_id, hashmap_t *map) {

  hashmap_value_t *group_offset = hashmap_get(map, group_id);

  if (group_offset == NULL)
    error("Couldn't find key (%zu) or create new one.", group_id);

  return (size_t)(*group_offset).value_st;
}

/* Compute send/recv offsets for MPI communication.
 * This is exactly the same as fof_compute_send_recv_offsets() in fof.c.
 */
__attribute__((always_inline)) INLINE static void fof_cloud_compute_send_recv_offsets(
    const int nr_nodes, int *sendcount, int **recvcount, int **sendoffset,
    int **recvoffset, size_t *nrecv) {

  /* Determine number of entries to receive */
  *recvcount = (int *)malloc(nr_nodes * sizeof(int));
  MPI_Alltoall(sendcount, 1, MPI_INT, *recvcount, 1, MPI_INT, MPI_COMM_WORLD);

  /* Compute send/recv offsets */
  *sendoffset = (int *)malloc(nr_nodes * sizeof(int));

  (*sendoffset)[0] = 0;
  for (int i = 1; i < nr_nodes; i++)
    (*sendoffset)[i] = (*sendoffset)[i - 1] + sendcount[i - 1];

  *recvoffset = (int *)malloc(nr_nodes * sizeof(int));

  (*recvoffset)[0] = 0;
  for (int i = 1; i < nr_nodes; i++)
    (*recvoffset)[i] = (*recvoffset)[i - 1] + (*recvcount)[i - 1];

  /* Allocate receive buffer */
  *nrecv = 0;
  for (int i = 0; i < nr_nodes; i++) (*nrecv) += (*recvcount)[i];
}

#endif /* WITH_MPI */

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

#ifdef WITH_MPI

/**
 * @brief Add a local<->foreign pair in range to the list of links
 *
 * Possibly reallocates the local_group_links if we run out of space.
 */
static INLINE void add_foreign_link_to_list_fof_cloud(
    int *local_link_count, int *group_links_size, struct fof_cloud_mpi **group_links,
    struct fof_cloud_mpi **local_group_links, const size_t root_i,
    const size_t root_j, const size_t size_i, const size_t size_j) {

  /* If the group_links array is not big enough re-allocate it. */
  if (*local_link_count + 1 > *group_links_size) {

    const int new_size = 2 * (*group_links_size);

    *group_links_size = new_size;

    (*group_links) = (struct fof_cloud_mpi *)realloc(
        *group_links, new_size * sizeof(struct fof_cloud_mpi));

    /* Reset the local pointer */
    (*local_group_links) = *group_links;

    message("Re-allocating local group links from %d to %d elements.",
            *local_link_count, new_size);

    if (new_size < 0) error("Overflow in size of list of foreign links");
  }

  /* Store the particle group properties for communication. */
  (*local_group_links)[*local_link_count].group_i = root_i;
  (*local_group_links)[*local_link_count].group_i_size = size_i;

  (*local_group_links)[*local_link_count].group_j = root_j;
  (*local_group_links)[*local_link_count].group_j_size = size_j;

  (*local_link_count)++;
}
#endif /* WITH_MPI */

/* Perform a FOF cloud search between a local and foreign cell using the Union-Find
 * algorithm. Store any links found between particles.*/
void fof_cloud_search_pair_cells_foreign(
    const struct fof_cloud_props *props, const double dim[3], const double l_x2,
    const int periodic, const struct part *const space_parts,
    const size_t nr_parts, const struct cell *restrict ci,
    const struct cell *restrict cj, int *restrict link_count,
    struct fof_cloud_mpi **group_links, int *restrict group_links_size) {

#ifdef WITH_MPI

  const size_t count_i = ci->hydro.count;
  const size_t count_j = cj->hydro.count;
  const struct part *parts_i = ci->hydro.parts;
  const struct part *parts_j = cj->hydro.parts;

  /* Get local pointers */
  const size_t *restrict group_index = props->group_index;
  const size_t *restrict group_size = props->group_size;

  /* Values local to this function to avoid dereferencing */
  struct fof_cloud_mpi *local_group_links = *group_links;
  int local_link_count = *link_count;

  /* Make a list of particle offsets into the global parts array. */
  const size_t *const offset_i =
      group_index + (ptrdiff_t)(parts_i - space_parts);

#ifdef SWIFT_DEBUG_CHECKS

  /* Check whether cells are local to the node. */
  const int ci_local = (ci->nodeID == engine_rank);
  const int cj_local = (cj->nodeID == engine_rank);

  if ((ci_local && cj_local) || (!ci_local && !cj_local))
    error(
        "FOF cloud search of foreign cells called on two local cells or two foreign "
        "cells.");

  if (!ci_local) {
    error("Cell ci, is not local.");
  }
#endif

  double shift[3] = {0.0, 0.0, 0.0};

  /* Get the relative distance between the pairs, wrapping. */
  for (int k = 0; k < 3; k++) {
    const double diff = cj->loc[k] - ci->loc[k];
    if (periodic && diff < -dim[k] / 2)
      shift[k] = dim[k];
    else if (periodic && diff > dim[k] / 2)
      shift[k] = -dim[k];
    else
      shift[k] = 0.0;
  }

  /* Loop over particles and find which particles belong in the same group. */
  for (size_t i = 0; i < count_i; i++) {

    const struct part *pi = &parts_i[i];

    /* Ignore inhibited particles */
    if (pi->time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (pi->rho < props->rho_min) continue;

    const double pix = pi->x[0] - shift[0];
    const double piy = pi->x[1] - shift[1];
    const double piz = pi->x[2] - shift[2];

    /* Find the root of pi. */
    const size_t root_i =
        fof_cloud_find_global(offset_i[i] - node_offset_cloud, group_index, nr_parts);

    for (size_t j = 0; j < count_j; j++) {

      const struct part *pj = &parts_j[j];

      /* Ignore inhibited particles */
      if (pj->time_bin >= time_bin_inhibited) continue;

      /* Check whether we ignore this particle type altogether */
      // Here we do not use if-statement since pi is already comfirmed to be
      // a hydro particle

      /* Check density threshold */
      if (pi->rho < props->rho_min) continue;

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

        /* Check that the links have not already been added to the list. */
        for (int l = 0; l < local_link_count; l++) {
          if (local_group_links[l].group_i == root_i &&
              local_group_links[l].group_j == pj->fof_cloud_data.group_id) {
            continue;
          }
        }

        /* Add a possible link to the list */
        add_foreign_link_to_list_fof_cloud(
            &local_link_count, group_links_size, group_links,
            &local_group_links, root_i, pj->fof_cloud_data.group_id,
            group_size[root_i - node_offset_cloud], pj->fof_cloud_data.group_size);
      }
    }
  }

  /* Update the returned values */
  *link_count = local_link_count;

#else
  error("Calling MPI function in non-MPI mode.");
#endif /* WITH_MPI */
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

#ifdef WITH_MPI

/* Recurse on a pair of cells (one local, one foreign) and perform a FOF cloud search
 * between cells that are within range. */
void rec_fof_cloud_search_pair_foreign(
    const struct fof_cloud_props *props, const double dim[3], const double search_r2,
    const int periodic, const struct part *const space_parts,
    const size_t nr_parts, const struct cell *ci, const struct cell *cj,
    int *restrict link_count, struct fof_cloud_mpi **group_links,
    int *restrict group_links_size) {

#ifdef SWIFT_DEBUG_CHECKS
  if (ci == cj) error("Pair FOF cloud called on same cell!!!");
  if (ci->nodeID == cj->nodeID) error("Fully local pair!");
#endif

  /* Find the shortest distance between cells, remembering to account for
   * boundary conditions. */
  const double r2 = cell_min_dist_fof_cloud(ci, cj, dim);

  /* Return if cells are out of range of each other */
  if (r2 > search_r2) return;

  /* Recurse on both cells if they are both split */
  if (ci->split && cj->split) {
    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL) {

        for (int l = 0; l < 8; l++) {
          if (cj->progeny[l] != NULL) {
            rec_fof_cloud_search_pair_foreign(props, dim, search_r2, periodic,
                                              space_parts, nr_parts, ci->progeny[k],
                                              cj->progeny[l], link_count, group_links,
                                              group_links_size);
          }
        }
      }
    }
  } else if (ci->split) {

    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL) {
        rec_fof_cloud_search_pair_foreign(props, dim, search_r2, periodic,
                                          space_parts, nr_parts, ci->progeny[k], cj,
                                          link_count, group_links, group_links_size);
      }
    }
  } else if (cj->split) {

    for (int k = 0; k < 8; k++) {
      if (cj->progeny[k] != NULL) {
        rec_fof_cloud_search_pair_foreign(props, dim, search_r2, periodic,
                                          space_parts, nr_parts, ci, cj->progeny[k],
                                          link_count, group_links, group_links_size);
      }
    }
  } else {
    /* Perform FOF cloud search between pairs of cells that are within the linking
     * length and not the same cell. */
    fof_cloud_search_pair_cells_foreign(props, dim, search_r2, periodic, space_parts,
                                        nr_parts, ci, cj, link_count, group_links,
                                        group_links_size);
  }
}
#endif

/* Mapper function to atomically update the group mass array. */
static INLINE void fof_cloud_update_group_mass_iterator(hashmap_key_t key,
                                                        hashmap_value_t *value,
                                                        void *data) {

  double *group_mass = (double *)data;

  /* Use key to index into group mass array. */
  atomic_add_d(&group_mass[key], value->value_dbl);
}

/* Mapper function to atomically update the group size array. */
static INLINE void fof_cloud_update_group_size_iterator(hashmap_key_t key,
                                                        hashmap_value_t *value,
                                                        void *data) {
  long long *group_size = (long long *)data;

  /* Use key to index into group mass array. */
  atomic_add(&group_size[key], value->value_st);
}

/**
 * @brief Mapper function to calculate the group masses.
 *
 * @param map_data An array of #part%s.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to a #space.
 */
void fof_cloud_calc_group_mass_mapper(void *map_data, int num_elements,
                                      void *extra_data) {

  /* Retrieve mapped data. */
  struct space *s = (struct space *)extra_data;
  struct part *parts = (struct part *)map_data;
  double *group_mass = s->e->fof_cloud_properties->group_mass;
  long long *group_size = s->e->fof_cloud_properties->final_group_size;
  const size_t group_id_default = s->e->fof_cloud_properties->group_id_default;
  const size_t group_id_offset = s->e->fof_cloud_properties->group_id_offset;

  /* Create hash table */
  hashmap_t map;
  hashmap_init(&map);

  /* Loop over particles and increment the group mass for groups above
   * min_group_size. */
  for (int ind = 0; ind < num_elements; ind++) {

    /* Only check groups above the minimum size. */
    if (parts[ind].fof_cloud_data.group_id != group_id_default) {

      hashmap_key_t index =
          parts[ind].fof_cloud_data.group_id - group_id_offset;
      hashmap_value_t *data = hashmap_get(&map, index);

      /* Update group mass */
      if (data != NULL) {
        (*data).value_dbl += parts[ind].mass;
        (*data).value_st++;
      } else
        error("Couldn't find key (%zu) or create new one.", index);
    }
  }

  /* Update the group mass array. */
  if (map.size > 0) {
    hashmap_iterate(&map, fof_cloud_update_group_mass_iterator, group_mass);
    hashmap_iterate(&map, fof_cloud_update_group_size_iterator, group_size);
  }

  hashmap_free(&map);
}

#ifdef WITH_MPI
/* Mapper function to unpack hash table into array. */
void fof_cloud_unpack_group_mass_mapper(hashmap_key_t key, hashmap_value_t *value,
                                        void *data) {

  struct fof_cloud_mass_send_hashmap *fof_cloud_mass_send =
      (struct fof_cloud_mass_send_hashmap *)data;
  struct fof_cloud_final_mass *mass_send = fof_cloud_mass_send->mass_send;
  size_t *nsend = &fof_cloud_mass_send->nsend;

  /* Store elements from hash table in array. */
  mass_send[*nsend].global_root = key;
  mass_send[*nsend].group_mass = value->value_dbl;
  mass_send[*nsend].final_group_size = value->value_ll;
  mass_send[*nsend].first_position[0] = value->value_array2_dbl[0];
  mass_send[*nsend].first_position[1] = value->value_array2_dbl[1];
  mass_send[*nsend].first_position[2] = value->value_array2_dbl[2];
  mass_send[*nsend].centre_of_mass[0] = value->value_array_dbl[0];
  mass_send[*nsend].centre_of_mass[1] = value->value_array_dbl[1];
  mass_send[*nsend].centre_of_mass[2] = value->value_array_dbl[2];
  mass_send[*nsend].max_part_density_index = value->value_st;
  mass_send[*nsend].max_part_density = value->value_flt;

  (*nsend)++;
}
#endif /* WITH_MPI */

/**
 * @brief Calculates the total mass and CoM of each group above min_group_size
 * and finds the densest particle.
 */
void fof_cloud_calc_group_mass(struct fof_cloud_props *props, const struct space *s,
                               const size_t num_groups_local,
                               const size_t num_groups_prev,
                               size_t *restrict num_on_node,
                               size_t *restrict first_on_node,
                               double *restrict group_mass) {

  const size_t nr_parts = s->nr_parts;
  struct part *parts = s->parts;
  const size_t group_id_offset = props->group_id_offset;
  const size_t group_id_default = props->group_id_default;
  const int periodic = s->periodic;
  const double dim[3] = {s->dim[0], s->dim[1], s->dim[2]};

#ifdef WITH_MPI
  size_t *group_index = props->group_index;
  const int nr_nodes = s->e->nr_nodes;

  /* Direct pointers to the arrays */
  long long *max_part_density_index = props->max_part_density_index;
  float *max_part_density = props->max_part_density;
  double *centre_of_mass = props->group_centre_of_mass;
  double *first_position = props->group_first_position;
  long long *final_group_size = props->final_group_size;

  /* Start the hash map */
  hashmap_t map;
  hashmap_init(&map);

  /* Collect information about the local particles and update the local AND
   * foreign group fragments */
  for (size_t i = 0; i < nr_parts; i++) {

    /* Ignore inhibited particles */
    if (parts[i].time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (parts[i].rho < props->rho_min) continue;

    /* Check if the particle is in a group above the threshold. */
    if (parts[i].fof_cloud_data.group_id != group_id_default) {

      const size_t root = fof_cloud_find_global(i, group_index, nr_parts);

      if (is_local_fof_cloud(root, nr_parts)) {

        /* The root is local */

        const size_t index =
            parts[i].fof_cloud_data.group_id - group_id_offset - num_groups_prev;

        /* Updata group mass */
        group_mass[index] += parts[i].mass;

        /* Updata group size */
        final_group_size[index]++;
      } else {

        /* The root is *not* local */

        /* Get the root in the foreign hashmap (create if necessary) */
        hashmap_value_t *const data = hashmap_get(&map, (hashmap_key_t)root);
        if (data == NULL)
          error("Couldn't find key (%zu) or create new one.", root);

        /* Compute the centre of mass */
        const double mass = parts[i].mass;
        double x[3] = {parts[i].x[0], parts[i].x[1], parts[i].x[2]};

        /* Add mass fragments of groups */
        data->value_dbl += mass;

        /* Increase fragments size */
        data->value_ll++;

        /* Record the first particle of this fragment that we encounter so we
         * can use it as reference frame for the centre of mass calculation
         */
        if (data->value_array2_dbl[0] == (double)(-FLT_MAX)) {
          data->value_array2_dbl[0] = parts[i].x[0];
          data->value_array2_dbl[1] = parts[i].x[1];
          data->value_array2_dbl[2] = parts[i].x[2];
        }

        if (periodic) {
          x[0] = nearest(x[0] - data->value_array2_dbl[0], dim[0]);
          x[1] = nearest(x[1] - data->value_array2_dbl[1], dim[1]);
          x[2] = nearest(x[2] - data->value_array2_dbl[2], dim[2]);
        }

        data->value_array_dbl[0] += mass * x[0];
        data->value_array_dbl[1] += mass * x[1];
        data->value_array_dbl[2] += mass * x[2];

        /* Also accumulate the densest gas particle ans its index */
        /* Update index if a denser gas particle is found */
        if (parts[i].rho > data->value_flt) {
          data->value_flt = parts[i].rho;
          data->value_st = i;
        }

      } /* Foreign root */
    } /* Particle is in a group */
  } /* Loop over particles */

  size_t nsend = map.size;
  struct fof_cloud_mass_send_hashmap hashmap_mass_send = {NULL, 0};

  /* Allocate and initialise a mass array */
  if (posix_memalign((void **)&hashmap_mass_send.mass_send, 32,
                     nsend * sizeof(struct fof_cloud_final_mass)) != 0)
    error("Failed to allocate list of group masses for FOF cloud search.");

  struct fof_cloud_final_mass *fof_cloud_mass_send = hashmap_mass_send.mass_send;

  /* Unpack mass fragments and roots from hash table */
  if (map.size > 0)
    hashmap_iterate(&map, fof_cloud_unpack_group_mass_mapper, &hashmap_mass_send);

  nsend = hashmap_mass_send.nsend;

#ifdef SWIFT_DEBUG_CHECKS
  if (nsend != map.size)
    error("No. of mass fragments to send != elements in hash table.");
#endif

  hashmap_free(&map);

  /* Sort by global root - this puts the groups in order of which node they're
   * stored on */
  qsort(fof_cloud_mass_send, nsend, sizeof(struct fof_cloud_final_mass),
        compare_fof_cloud_final_mass_global_root);

  /* Determine how many entries go to each node */
  int *sendcount = (int *)calloc(nr_nodes, sizeof(int));
  int dest = 0;
  for (size_t i = 0; i < nsend; i++) {
    while ((fof_cloud_mass_send[i].global_root >=
            first_on_node[dest] + num_on_node[dest]) ||
           (num_on_node[dest] == 0))
      dest++;

    if (dest >= nr_nodes) error("Node index out of range!");

    sendcount[dest]++;
  }

  int *recvcount = NULL, *sendoffset = NULL, *recvoffset = NULL;
  size_t nrecv = 0;

  fof_cloud_compute_send_recv_offsets(nr_nodes, sendcount, &recvcount, &sendoffset,
                                      &recvoffset, &nrecv);

  struct fof_cloud_final_mass *fof_cloud_mass_recv =
      (struct fof_cloud_final_mass *)malloc(nrecv * sizeof(struct fof_cloud_final_mass));

  /* Exchange group mass */
  MPI_Alltoallv(fof_cloud_mass_send, sendcount, sendoffset, fof_cloud_final_mass_type,
                fof_cloud_mass_recv, recvcount, recvoffset, fof_cloud_final_mass_type,
                MPI_COMM_WORLD);

  /* For each received global root, look up the group ID we assigned and
   * increment the group mass */
  for (size_t i = 0; i < nrecv; i++) {
#ifdef SWIFT_DEBUG_CHECKS
    if ((fof_cloud_mass_recv[i].global_root < node_offset_cloud) ||
        (fof_cloud_mass_recv[i].global_root >= node_offset_cloud + nr_parts)) {
      error("Received global root index out of range!");
    }
#endif
    const size_t local_root_index = fof_cloud_mass_recv[i].global_root - node_offset_cloud;
    const size_t local_group_offset = group_id_offset + num_groups_prev;
    const size_t index =
        parts[local_root_index].fof_cloud_data.group_id - local_group_offset;
    group_mass[index] += fof_cloud_mass_recv[i].group_mass;
    final_group_size[index] += fof_cloud_mass_recv[i].final_group_size;
  }

  /* Loop over particles, densest particle in each *local* group.
   * We can do this now as we eventually have the total group mass */
  for (size_t i = 0; i < nr_parts; i++) {

    /* Ignore inhibited particles */
    if (parts[i].time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (parts[i].rho < props->rho_min) continue;

    /* Only check groups above the minimum mass threshold */
    if (parts[i].fof_cloud_data.group_id != group_id_default) {

      const size_t root = fof_cloud_find_global(i, group_index, nr_parts);

      if (is_local_fof_cloud(root, nr_parts)) {

        const size_t index =
            parts[i].fof_cloud_data. group_id - group_id_offset - num_groups_prev;

        /* Compute the center of mass */
        const double mass = parts[i].mass;
        double x[3] = {parts[i].x[0], parts[i].x[1], parts[i].x[2]};

        /* Record the first particle of this group that we encounter so we
         * can use it as reference frame for the centre of mass calculation */
        if (first_position[index * 3 + 0] == (double)(-FLT_MAX)) {
          first_position[index * 3 + 0] = x[0];
          first_position[index * 3 + 1] = x[1];
          first_position[index * 3 + 2] = x[2];
        }

        if (periodic) {
          x[0] = nearest(x[0] - first_position[index * 3 + 0], dim[0]);
          x[1] = nearest(x[1] - first_position[index * 3 + 1], dim[1]);
          x[2] = nearest(x[2] - first_position[index * 3 + 2], dim[2]);
        }

        centre_of_mass[index * 3 + 0] += mass * x[0];
        centre_of_mass[index * 3 + 1] += mass * x[1];
        centre_of_mass[index * 3 + 2] += mass * x[2];

        /* Update index if a denser gas particle is found. */
        if (parts[i].rho > max_part_density[index]) {
          max_part_density_index[index] = i;
          max_part_density[index] = parts[i].rho;
        }
      }
    }
  }

  /* For each received global root, look up the group ID we assigned and find
   * the global maximum gas density */
  for (size_t i = 0; i < nrecv; i++) {

    const size_t local_root_index =
        fof_cloud_mass_recv[i].global_root - node_offset_cloud;
    const size_t local_group_offset = group_id_offset + num_groups_prev;
    const size_t index =
        parts[local_root_index].fof_cloud_data.group_id - local_group_offset;

    double fragment_mass = fof_cloud_mass_recv[i].group_mass;
    double fragment_centre_of_mass[3] = {
        fof_cloud_mass_recv[i].centre_of_mass[0] / fof_cloud_mass_recv[i].group_mass,
        fof_cloud_mass_recv[i].centre_of_mass[1] / fof_cloud_mass_recv[i].group_mass,
        fof_cloud_mass_recv[i].centre_of_mass[2] / fof_cloud_mass_recv[i].group_mass};
    fragment_centre_of_mass[0] += fof_cloud_mass_recv[i].first_position[0];
    fragment_centre_of_mass[1] += fof_cloud_mass_recv[i].first_position[1];
    fragment_centre_of_mass[2] += fof_cloud_mass_recv[i].first_position[2];

    if (periodic) {
      fragment_centre_of_mass[0] = nearest(
          fragment_centre_of_mass[0] - first_position[3 * index + 0], dim[0]);
      fragment_centre_of_mass[1] = nearest(
          fragment_centre_of_mass[1] - first_position[3 * index + 1], dim[1]);
      fragment_centre_of_mass[2] = nearest(
          fragment_centre_of_mass[2] - first_position[3 * index + 2], dim[2]);
    }

    centre_of_mass[index * 3 + 0] += fragment_mass * fragment_centre_of_mass[0];
    centre_of_mass[index * 3 + 1] += fragment_mass * fragment_centre_of_mass[1];
    centre_of_mass[index * 3 + 2] += fragment_mass * fragment_centre_of_mass[2];
  }

  /* Send the result back */
  MPI_Alltoallv(fof_cloud_mass_recv, recvcount, recvoffset, fof_cloud_final_mass_type,
                fof_cloud_mass_send, sendcount, sendoffset, fof_cloud_final_mass_type,
                MPI_COMM_WORLD);

  free(sendcount);
  free(recvcount);
  free(sendoffset);
  free(recvoffset);
  free(fof_cloud_mass_send);
  free(fof_cloud_mass_recv);

#else

  /* Increment the group mass for groups above min_group_size. */
  threadpool_map(&s->e->threadpool, fof_cloud_calc_group_mass_mapper, parts,
                 nr_parts, sizeof(struct part), threadpool_auto_chunk_size,
                 (struct space *)s);

  /* Direct pointers to the arrays */
  long long *max_part_density_index = props->max_part_density_index;
  float *max_part_density = props->max_part_density;
  double *centre_of_mass = props->group_centre_of_mass;
  double *first_position = props->group_first_position;

  /* Loop over particles, compute CoM and find the densest particle in each
   * group. */
  for (size_t i = 0; i < nr_parts; i++) {

    /* Ignore inhibited particles */
    if (parts[i].time_bin >= time_bin_inhibited) continue;

    /* Check whether we ignore this particle type altogether */
    // Here we do not use if-statement since pi is already comfirmed to be
    // a hydro particle

    /* Check density threshold */
    if (parts[i].rho < props->rho_min) continue;

    const size_t index = parts[i].fof_cloud_data.group_id - group_id_offset;

    /* Only check groups above the minimum mass threshold. */
    if (parts[i].fof_cloud_data.group_id != group_id_default) {

      /* Compute the centre of mass */
      const double mass = parts[i].mass;
      double x[3] = {parts[i].x[0], parts[i].x[1], parts[i].x[2]};

      /* Record the first particle of this group that we encounter so we
       * can use it as reference frame for the centre of mass calculation */
      if (first_position[index * 3 + 0] == (double)(-FLT_MAX)) {
        first_position[index * 3 + 0] = x[0];
        first_position[index * 3 + 1] = x[1];
        first_position[index * 3 + 2] = x[2];
      }

      if (periodic) {
        x[0] = nearest(x[0] - first_position[index * 3 + 0], dim[0]);
        x[1] = nearest(x[1] - first_position[index * 3 + 1], dim[1]);
        x[2] = nearest(x[2] - first_position[index * 3 + 2], dim[2]);
      }

      centre_of_mass[index * 3 + 0] += mass * x[0];
      centre_of_mass[index * 3 + 1] += mass * x[1];
      centre_of_mass[index * 3 + 2] += mass * x[2];

      /* Update index if a denser gas particle is found. */
      if (parts[i].rho > max_part_density[index]) {
        max_part_density[index] = parts[i].rho;
        max_part_density_index[index] = i;
      }
    }
  }
#endif /* WITH_MPI */
}

/**
 * @brief Mapper function to perform FOF search.
 *
 * @param map_data An array of #cell pair indices.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to a #space.
 */
void fof_cloud_find_foreign_links_mapper(void *map_data, int num_elements,
                                         void *extra_data) {

#ifdef WITH_MPI

  /* Retrieve mapped data. */
  struct space *s = (struct space *)extra_data;
  const int periodic = s->periodic;
  const size_t nr_parts = s->nr_parts;
  const struct part *const parts = s->parts;
  const struct engine *e = s->e;
  struct fof_cloud_props *props = e->fof_cloud_properties;
  struct cloud_cell_pair_indices *cell_pairs =
      (struct cloud_cell_pair_indices *)map_data;

  const double dim[3] = {s->dim[0], s->dim[1], s->dim[2]};
  const double search_r2 = props->l_x2;

  /* Store links in an array local to this thread. */
  int local_link_count = 0;
  int local_group_links_size = props->group_links_size / e->nr_threads;

  /* Init the local group links buffer. */
  struct fof_cloud_mpi *local_group_links = (struct fof_cloud_mpi *)swift_calloc(
      "fof_cloud_group_links", sizeof(struct fof_cloud_mpi), local_group_links_size);
  if (local_group_links == NULL)
    error("Failed to allocate temporary group links buffer.");

  /* Loop over all pairs of local and foreign cells, recurse then perform a
   * FOF cloud search. */
  for (int ind = 0; ind < num_elements; ind++) {

    /* Get the local and foreign cells to recurse on */
    const struct cell *restrict local_cell = cell_pairs[ind].local;
    const struct cell *restrict foreign_cell = cell_pairs[ind].foreign;

    rec_fof_cloud_search_pair_foreign(props, dim, search_r2, periodic, parts,
                                      nr_parts, local_cell, foreign_cell,
                                      &local_link_count, &local_group_links,
                                      &local_group_links_size);
  }

  /* Add links found by this thread to the global link list. */
  /* Lock to prevent race conditions while adding to the global link list.*/
  if (lock_lock(&s->lock) == 0) {

    /* get pointers to global arrays */
    int *restrict group_links_size = &props->group_links_size;
    int *restrict group_link_count = &props->group_link_count;
    struct fof_cloud_mpi **group_links = &props->group_links;

    /* If the global group_links array is not big enough re-allocate it. */
    if (*group_link_count + local_link_count > *group_links_size) {

      const int old_size = *group_links_size;
      const int new_size =
          max(*group_link_count + local_link_count, 2 * old_size);

      (*group_links) = (struct fof_cloud_mpi *)realloc(
          *group_links, new_size * sizeof(struct fof_cloud_mpi));

      *group_links_size = new_size;

      message("Re-allocating global group links from %d to %d elements.",
              old_size, new_size);
    }

    /* Copy the local links to the global list */
    for (int i = 0; i < local_link_count; i++) {

      int found = 0;

      /* Check that the links have not already been added to the list by another
       * thread. */
      for (int l = 0; l < *group_link_count; l++) {
        if ((*group_links)[l].group_i == local_group_links[i].group_i &&
            (*group_links)[l].group_j == local_group_links[i].group_j) {
          found = 1;
          break;
        }
      }

      if (!found) {

        (*group_links)[*group_link_count].group_i =
            local_group_links[i].group_i;
        (*group_links)[*group_link_count].group_i_size =
            local_group_links[i].group_i_size;

        (*group_links)[*group_link_count].group_j =
            local_group_links[i].group_j;
        (*group_links)[*group_link_count].group_j_size =
            local_group_links[i].group_j_size;

        (*group_link_count) = (*group_link_count) + 1;
      }
    }
  }

  /* Release lock. */
  if (lock_unlock(&s->lock) != 0) error("Failed to unlock the space");

  swift_free("fof_local_group_links", local_group_links);
#endif /* WITH_MPI */
}

/*
 *
 */
void fof_cloud_finalise_group_data(struct fof_cloud_props *props,
                                   const struct cloud_group_length *group_sizes,
                                   const struct part *parts, const int periodic,
                                   const double dim[3], const int num_groups) {

  size_t *group_size =
      (size_t *)swift_malloc("fof_cloud_group_size", num_groups * sizeof(size_t));
  size_t *group_index =
      (size_t *)swift_malloc("fof_cloud_group_index", num_groups * sizeof(size_t));
  double *group_centre_of_mass = (double *)swift_malloc(
      "fof_cloud_group_centre_of_mass", 3 * num_groups * sizeof(double));

  for (int i = 0; i < num_groups; i++) {

    const size_t group_offset = group_sizes[i].index;

    /* Centre of mass, including possible box wrapping */
    double CoM[3] = {
        props->group_centre_of_mass[i * 3 + 0] / props->group_mass[i],
        props->group_centre_of_mass[i * 3 + 1] / props->group_mass[i],
        props->group_centre_of_mass[i * 3 + 2] / props->group_mass[i]};
    if (periodic) {
      CoM[0] =
          box_wrap(CoM[0] + props->group_first_position[i * 3 + 0], 0., dim[0]);
      CoM[1] =
          box_wrap(CoM[1] + props->group_first_position[i * 3 + 1], 0., dim[1]);
      CoM[2] =
          box_wrap(CoM[2] + props->group_first_position[i * 3 + 2], 0., dim[2]);
    }

#ifdef WITH_MPI
    group_index[i] = parts[group_offset - node_offset_cloud].fof_cloud_data.group_id;
    group_size[i] = props->group_size[group_offset - node_offset_cloud];
#else
    group_index[i] = parts[group_offset].fof_cloud_data.group_id;
    group_size[i] = props->group_size[group_offset];
#endif

    group_centre_of_mass[i * 3 + 0] = CoM[0];
    group_centre_of_mass[i * 3 + 1] = CoM[1];
    group_centre_of_mass[i * 3 + 2] = CoM[2];
  }

  swift_free("fof_cloud_group_centre_of_mass", props->group_centre_of_mass);
  swift_free("fof_cloud_group_size", props->group_size);
  swift_free("fof_cloud_group_index", props->group_index);

  props->group_centre_of_mass = group_centre_of_mass;
  props->group_size = group_size;
  props->group_index = group_index;
}

struct mapper_data_fof_cloud {
  size_t *group_index;
  size_t *group_size;
  float *distance_to_link;
  size_t nr_parts;
  struct part *space_parts;
};

/**
 * @brief Mapper function to set the roots of #part%s going to other MPI ranks.
 *
 * @param map_data The list of outgoing local cells.
 * @param num_elements Chunk size.
 * @param extra_data Pointer to mapper data.
 */
void fof_cloud_set_outgoing_root_mapper(void *map_data, int num_elements,
                                        void *extra_data) {

#ifdef WITH_MPI

  /* Unpack the data */
  struct cell **local_cells = (struct cell **)map_data;
  const struct mapper_data_fof_cloud *data =
      (struct mapper_data_fof_cloud *)extra_data;
  const size_t *const group_index = data->group_index;
  const size_t *const group_size = data->group_size;
  const size_t nr_parts = data->nr_parts;
  const struct part *const space_parts = data->space_parts;

  /* Loop over the out-going local cells */
  for (int i = 0; i < num_elements; i++) {

    /* Get the cell and its parts */
    struct cell *local_cell = local_cells[i];
    struct part *parts = local_cell->hydro.parts;

    /* Make a list of particle offsets into the global parts array. */
    const size_t *const offset =
        group_index + (ptrdiff_t)(parts - space_parts);

    /* Set each particle's root and group properties found in the local FOF cloud */
    for (int k = 0; k < local_cell->hydro.count; k++) {

      /* TODO: Can we skip ignorable particles here?
       * Likely makes no difference */

      /* Recall we did alter the group_index with a global_offset.
       * We need to remove that here as we want the *local* root */
      const size_t root =
          fof_cloud_find_global(offset[k] - node_offset_cloud, group_index, nr_parts);

      /* TODO: Could we call fof_cloud_find() here instead?
       * Likely yes but we don't want path compression at this stage.
       * So, probably not */
      parts[k].fof_cloud_data.group_id = root;
      parts[k].fof_cloud_data.group_size = group_size[root - node_offset_cloud];
    }
  }

#endif /* WITH_MPI */
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
  struct engine *e =s->e;
  const int verbose = e->verbose;

  /* Abort if only one nore */
  if (e->nr_nodes == 1) return;

  size_t *restrict group_index = props->group_index;
  size_t *restrict group_size = props->group_size;
  const size_t nr_parts = s->nr_parts;
  const double dim[3] = {s->dim[0], s->dim[1], s->dim[2]};
  const double search_r2 = props->l_x2;

  const ticks tic_total = getticks();
  ticks tic = getticks();

  /* Make group IDs globally unique */
  for (size_t i = 0; i < nr_parts; i++) group_index[i] += node_offset_cloud;

  struct cloud_cell_pair_indices *cell_pairs = NULL;
  int cell_pair_count = 0;

  props->group_links_size = fof_cloud_props_default_group_link_size;

  int num_cells_out = 0;
  int num_cells_in = 0;

  /* Find the maximum no. of cell pairs that can communicate. */
  for (int  i = 0; i < e->nr_proxies; i++) {

    for (int j = 0; j < e->proxies[i].nr_cells_out; j++) {

      /* Only include hydro cells */
      if (e->proxies[i].cells_out_type[j] & proxy_cell_type_hydro)
        num_cells_out++;
    }

    for (int j = 0; j < e->proxies[i].nr_cells_in; j++) {

      /* Only include hydro cells */
      if (e->proxies[i].cells_in_type[j] & proxy_cell_type_hydro)
        num_cells_in++;
    }
  }

  if (verbose)
    message(
        "Finding max no. of cells + offset IDs"
        "took: %.3f %s.",
        clocks_from_ticks(getticks() - tic), clocks_getunit());

  const int cell_pair_size = num_cells_in * num_cells_out;

  /* Allocate memory for all the possible cell links */
  if (swift_memalign("fof_cloud_groups_links", (void **)&props->group_links,
                     SWIFT_STRUCT_ALIGNMENT,
                     props->group_links_size * sizeof(struct fof_cloud_mpi)) != 0)
    error("Error while allocating memory for FOF cloud links over an MPI domain");

  if (swift_memalign("fof_cloud_cell_pairs", (void **)&cell_pairs,
                     SWIFT_STRUCT_ALIGNMENT,
                     cell_pair_size * sizeof(struct cloud_cell_pair_indices)) != 0)
    error("Error while allocating memory for FOF cloud cell pair indices");

  ticks tic_pairs = getticks();

  /* Loop over cells_in and cells_out for each proxy and find which cells are in
   * range of each other to perform the FOF cloud search. Store local cells that
   * are touching foreign cells in a list. */
  for (int i = 0; i < e->nr_proxies; i++) {

    /* Only find links across an MPI rank domain on one rank */
    if (engine_rank == min(engine_rank, e->proxies[i].nodeID)) {

      for (int j = 0; j < e->proxies[i].nr_cells_out; j++) {

        /* Skip non-hydro cells. */
        if (!(e->proxies[i].cells_out_type[j] & proxy_cell_type_hydro))
          continue;

        struct cell *restrict local_cell = e->proxies[i].cells_out[j];

        /* Skip empty cells */
        if (local_cell->hydro.count == 0) continue;

        for (int k = 0; k < e->proxies[i].nr_cells_in; k++) {

          /* Skip non-hydro cells */
          if(!(e->proxies[i].cells_in_type[k] & proxy_cell_type_hydro))
            continue;

          struct cell *restrict foreign_cell = e->proxies[i].cells_in[k];

          /* Skip empty cells */
          if (foreign_cell->hydro.count == 0) continue;

          /* Add candidates in range to the list of pairs of cells to treat */
          const double r2 = cell_min_dist_fof_cloud(local_cell, foreign_cell, dim);
          if (r2 < search_r2) {
            cell_pairs[cell_pair_count].local = local_cell;
            cell_pairs[cell_pair_count].foreign = foreign_cell;

            cell_pair_count++;
          }
        }
      }
    }
  }

  if (verbose)
    message("Finding local/foreign cell pairs took: %.3f %s.",
            clocks_from_ticks(getticks() - tic_pairs), clocks_getunit());

  const ticks tic_set_roots = getticks();

  /* Set the root of outgoing particles. */

  /* Allocate array of outgoing cells and populate it */
  struct cell **local_cells =
      (struct cell **)malloc(num_cells_out * sizeof(struct cell *));
  int count = 0;
  for (int i = 0; i < e->nr_proxies; i++) {
    for (int j = 0; j < e->proxies[i].nr_cells_out; j++) {

      /* Only include hydro cells */
      if (e->proxies[i].cells_out_type[j] & proxy_cell_type_hydro) {

        local_cells[count] = e->proxies[i].cells_out[j];
        count++;
      }
    }
  }

  /* Now set the *local* roots of all the parts we are sending */
  struct mapper_data_fof_cloud data;
  data.group_index = group_index;
  data.group_size = group_size;
  data.nr_parts = nr_parts;
  data.space_parts = s->parts;
  threadpool_map(&e->threadpool, fof_cloud_set_outgoing_root_mapper,
                 local_cells, num_cells_out, sizeof(struct cell **),
                 threadpool_auto_chunk_size, &data);

  if (verbose)
    message("Initialising particle roots took: %.3f %s.",
            clocks_from_ticks(getticks() - tic_set_roots), clocks_getunit());

  free(local_cells);

  if (verbose)
    message(
        "Finding local/foreign cell pairs and initialising particle roots "
        "took: %.3f %s.",
        clocks_from_ticks(getticks() - tic), clocks_getunit());

  /* Activate the tasks exchanging all the required parts */
  engine_activate_part_comms(e);

  ticks local_fof_tic = getticks();

  /* Wait for all the communication tasks to be ready */
  MPI_Barrier(MPI_COMM_WORLD);

  if (verbose)
    message("Local FOF cloud imbalance: %.3f %s.",
            clocks_from_ticks(getticks() - local_fof_tic), clocks_getunit());

  tic = getticks();

  /* Perform send and receive tasks. */
  engine_launch(e, "fof cloud comms");

  if (verbose)
    message("MPI send/recv comms took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  /* We have now recevied the foreign particles. Each particle received
   * carries information about its own *foreign* (to us) root and the
   * size of the group fragment it belongs too its original foreign rank. */

  tic = getticks();

  props->group_link_count = 0;

  /* Perform search of group links between local and foreign cells with the
   * threadpool. */
  threadpool_map(&s->e->threadpool, fof_cloud_find_foreign_links_mapper, cell_pairs,
                 cell_pair_count, sizeof(struct cloud_cell_pair_indices), 1,
                 (struct space *)s);

  /* Clean up memory used by foreign particles. */
  swift_free("fof_cell_pairs", cell_pairs);

  tic = getticks();

  const ticks comms_tic = getticks();

  MPI_Barrier(MPI_COMM_WORLD);

  if (verbose)
    message("Imbalance took: %.3f %s.",
            clocks_from_ticks(getticks() - comms_tic), clocks_getunit());

  if (verbose)
    message("fof_cloud_search_foreign_cells() took (FOF_CLOUD SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic_total), clocks_getunit());

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

  /* Offset into parts array. */
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
  struct engine *e = s->e;
  const int verbose = e->verbose;

  /* Abort if only one node */
  if (e->nr_nodes == 1) return;

  const size_t nr_parts = s->nr_parts;
  size_t *restrict group_index = props->group_index;
  size_t *restrict group_size = props->group_size;

  const ticks tic_total = getticks();
  ticks tic = getticks();
  const ticks comms_tic = getticks();

  if (verbose)
    message(
        "Searching %zu hydro particles for cross-node links with l_x: %lf",
        nr_parts, sqrt(props->l_x2));

  /* Local copy of the variable set in the mapper */
  const int group_link_count = props->group_link_count;

  /* Sum the total number of links across MPI domains over each MPI rank. */
  int global_group_link_count = 0;
  MPI_Allreduce(&group_link_count, &global_group_link_count, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  if (global_group_link_count < 0)
    error("Overflow of the size of the global list of foreign links");

  struct fof_cloud_mpi *global_group_links = NULL;
  int *displ = NULL, *group_link_counts = NULL;

  if (swift_memalign("fof_cloud_global_group_links", (void **)&global_group_links,
                     SWIFT_STRUCT_ALIGNMENT,
                     global_group_link_count * sizeof(struct fof_cloud_mpi)) != 0)
    error("Error while allocating memory for the global list of group links");

  if (posix_memalign((void **)&group_link_counts, SWIFT_STRUCT_ALIGNMENT,
                     e->nr_nodes * sizeof(int)) != 0)
    error(
        "Error while allocating memory for the number of group links on each "
        "MPI rank");

  if (posix_memalign((void **)&displ, SWIFT_STRUCT_ALIGNMENT,
                     e->nr_nodes * sizeof(int)) != 0)
    error(
        "Error while allocating memory for the displacement in memory for the "
        "global group link list");

  /* Gather the total number of links on each rank. */
  MPI_Allgather(&group_link_count, 1, MPI_INT, group_link_counts, 1, MPI_INT,
                MPI_COMM_WORLD);

  /* Set the displacements into the global link list using the link counts from
   * each rank */
  displ[0] = 0;
  for (int i = 1; i < e->nr_nodes; i++) {
    displ[i] = displ[i - 1] + group_link_counts[i - 1];
    if (displ[i] < 0) error("Number of group links overflowing!");
  }

  /* Gather the global link list on all ranks. */
  MPI_Allgatherv(props->group_links, group_link_count, fof_cloud_mpi_type,
                 global_group_links, group_link_counts, displ, fof_cloud_mpi_type,
                 MPI_COMM_WORLD);

  /* Clean up memory. */
  free(group_link_counts);
  free(displ);
  swift_free("fof_cloud_group_links", props->group_links);
  props->group_links = NULL;

  if (verbose) {
    message("Communication took: %.3f %s.",
            clocks_from_ticks(getticks() - comms_tic), clocks_getunit());

    message("Global comms took: %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  }

  tic = getticks();

  /* Transform the group IDs to a local list going from 0-group_count so a
   * union-find can be performed.
   * Each member of a link is stored separately --> Need 2x as many entries */
  size_t *global_group_index = NULL, *global_group_id = NULL,
         *global_group_size = NULL;
  const int global_group_list_size = 2 * global_group_link_count;

  if (swift_memalign("fof_cloud_global_group_index", (void **)&global_group_index,
                     SWIFT_STRUCT_ALIGNMENT,
                     global_group_list_size * sizeof(size_t)) != 0)
    error(
        "Error while allocating memory for the displacement in memory for the "
        "global group link list");

  if (swift_memalign("fof_cloud_global_group_id", (void **)&global_group_id,
                     SWIFT_STRUCT_ALIGNMENT,
                     global_group_list_size * sizeof(size_t)) != 0)
    error(
        "Error while allocating memory for the displacement in memory for the "
        "global group link list");

  if (swift_memalign("fof_cloud_global_group_size", (void **)&global_group_size,
                     SWIFT_STRUCT_ALIGNMENT,
                     global_group_list_size * sizeof(size_t)) != 0)
    error(
        "Error while allocating memory for the displacement in memory for the "
        "global group link list");

  bzero(global_group_size, global_group_list_size * sizeof(size_t));

  /* Create hash table. */
  hashmap_t map;
  hashmap_init(&map);

  /* Store each group ID and its properties. */
  int group_count = 0;
  for (int k = 0; k < global_group_link_count; k++) {

    const size_t group_i = global_group_links[k].group_i;
    const size_t group_j = global_group_links[k].group_j;

    global_group_size[group_count] += global_group_links[k].group_i_size;
    global_group_id[group_count] = group_i;
    hashmap_add_cloud_group(group_i, group_count, &map);
    group_count++;

    global_group_size[group_count] += global_group_links[k].group_j_size;
    global_group_id[group_count] = group_j;
    hashmap_add_cloud_group(group_j, group_count, &map);
    group_count++;
  }

  if (verbose)
    message("Global list compression took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  tic = getticks();

  /* Create a global_group_index list of groups across MPI domains so that you
   * can perform a union-find locally on each node.
   * The value of which is an offset into global_group_id, which is the actual
   * root. */
  for (int i = 0; i < group_count; i++) global_group_index[i] = i;

  /* Store the original group size before incrementing in the Union-Find. */
  size_t *orig_global_group_size = NULL;

  if (swift_memalign("fof_cloud_orig_global_group_size",
                     (void **)&orig_global_group_size, SWIFT_STRUCT_ALIGNMENT,
                     group_count * sizeof(size_t)) != 0)
    error(
        "Error while allocating memory for the displacement in memory for the "
        "global group link list");

  memcpy(orig_global_group_size, global_group_size,
         group_count * sizeof(size_t));

  /* Perform a union-find on the group links. */
  for (int k = 0; k < global_group_link_count; k++) {

    /* Use the hash table to find the group offsets in the index array */
    const size_t find_i =
        hashmap_find_cloud_group_offset(global_group_links[k].group_i, &map);
    const size_t find_j =
        hashmap_find_cloud_group_offset(global_group_links[k].group_j, &map);

    /* Use the offset to find the group's root. */
    const size_t root_i = fof_cloud_find(find_i, global_group_index);
    const size_t root_j = fof_cloud_find(find_j, global_group_index);

    const size_t group_i = global_group_id[root_i];
    const size_t group_j = global_group_id[root_j];

    if (group_i == group_j) continue;

    /* Update roots accordingly */
    const size_t size_i = global_group_size[root_i];
    const size_t size_j = global_group_size[root_j];
#ifdef UNION_BY_SIZE_OVER_MPI
    if (size_i < size_j) {
      global_group_index[root_i] = root_j;
      global_group_size[root_j] += size_i;
    } else {
      global_group_index[root_j] = root_i;
      global_group_size[root_i] += size_j;
    }
#else
    if (group_j < group_i) {
      global_group_index[root_i] = root_j;
      global_group_size[root_j] += size_i;
    } else {
      global_group_index[root_j] = root_i;
      global_group_size[root_i] += size_j;
    }
#endif
  }

  hashmap_free(&map);

  if (verbose)
    message("global_group_index construction took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  tic = getticks();

  /* Update each group locally with new root information */
  for (int i = 0; i < group_count; i++) {

    const size_t group_id = global_group_id[i];
    const size_t offset = fof_cloud_find(global_group_index[i], global_group_index);
    const size_t new_root = global_group_id[offset];

    /* If the group is local update its root and size */
    if (is_local_fof_cloud(group_id, nr_parts) && new_root != group_id) {

      group_index[group_id - node_offset_cloud] = new_root;
      group_size[group_id - node_offset_cloud] -= orig_global_group_size[i];
    }

    /* If the group linked to a local root update its size */
    if (is_local_fof_cloud(new_root, nr_parts) && new_root != group_id) {

      /* Use group sizes before Union-Find */
      group_size[new_root - node_offset_cloud] += orig_global_group_size[i];
    }
  }

  if (verbose)
    message("Updating groups locally took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  /* Clean up memory. */
  swift_free("fof_cloud_global_group_links", global_group_links);
  swift_free("fof_cloud_global_group_index", global_group_index);
  swift_free("fof_cloud_global_group_size", global_group_size);
  swift_free("fof_cloud_global_group_id", global_group_id);
  swift_free("fof_cloud_orig_global_group_size", orig_global_group_size);

  if (verbose) {
    message("link_foreign_fragmens() took (FOF_CLOUD SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic_total), clocks_getunit());
  }

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

  const int verbose = s->e->verbose;
#ifdef WITH_MPI
  const int nr_nodes = s->e->nr_nodes;
#endif
  const ticks tic_total = getticks();

  struct part *parts = s->parts;
  const size_t nr_parts = s->nr_parts;

  const size_t min_group_size = props->min_group_size;
  const size_t group_id_offset = props->group_id_offset;
  const size_t group_id_default = props->group_id_default;

  size_t num_groups_local = 0;
  size_t num_parts_in_groups_local = 0;
  size_t max_group_size_local = 0;

  /* Local copy of the arrays */
  size_t *restrict group_index = props->group_index;
  size_t *restrict group_size = props->group_size;

  const ticks tic_num_groups_calc = getticks();

  for (size_t i = 0; i < nr_parts; i++) {

#ifdef WITH_MPI
    /* Find the total number of groups */
    if (group_index[i] == i + node_offset_cloud && group_size[i] >= min_group_size)
      num_groups_local++;
#else
    /* Find the total number of groups */
    if (group_index[i] == i && group_size[i] >= min_group_size)
      num_groups_local++;
#endif

    /* Find the total number of particles in groups */
    if (group_size[i] >= min_group_size)
      num_parts_in_groups_local += group_size[i];

    /* Find the largest group */
    if  (group_size[i] > max_group_size_local)
      max_group_size_local = group_size[i];
  }

  if (verbose)
    message(
        "Calculating the total no. of local groups took: (FOF_CLOUD SCALING): %.3f "
        "%s.",
        clocks_from_ticks(getticks() - tic_num_groups_calc), clocks_getunit());

  /* Sort the groups in descending order based upon size and re-label their
   * IDs 0-num_groups. */
  struct cloud_group_length *high_group_sizes = NULL;
  int group_count = 0;
  if (swift_memalign("fof_cloud_high_group_sizes", (void **)&high_group_sizes, 32,
                     num_groups_local * sizeof(struct cloud_group_length)) != 0)
    error("Failed to allocate list of large groups for cloud.");

  /* Store the group_sizes and their offset. */
  for (size_t i = 0; i < nr_parts; i++) {

#ifdef WITH_MPI
    if (group_index[i] == i + node_offset_cloud && group_size[i] >= min_group_size) {
      high_group_sizes[group_count].index = node_offset_cloud + i;
      high_group_sizes[group_count++].size = group_size[i];
    }
#else
    if (group_index[i] == i && group_size[i] >= min_group_size) {
      high_group_sizes[group_count].index = i;
      high_group_sizes[group_count++].size = group_size[i];
    }
#endif
  }

  ticks tic = getticks();

  /* Find global properties. */
  long long num_groups = 0, num_parts_in_groups = 0, max_group_size = 0;
#ifdef WITH_MPI
  MPI_Allreduce(&num_groups_local, &num_groups, 1, MPI_LONG_LONG_INT, MPI_SUM,
                MPI_COMM_WORLD);

  if (verbose)
    message("Finding the total no. of groups took: (FOF_CLOUD SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic_num_groups_calc),
            clocks_getunit());

  MPI_Reduce(&num_parts_in_groups_local, &num_parts_in_groups, 1,
             MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_group_size_local, &max_group_size, 1, MPI_LONG_LONG_INT,
             MPI_MAX, 0, MPI_COMM_WORLD);
#else
  num_groups = num_groups_local;

  num_parts_in_groups = num_parts_in_groups_local;
  max_group_size = max_group_size_local;
#endif /* WITH_MPI */
  props->num_groups = num_groups;

  /* Find number of groups on lower numbered MPI ranks */
#ifdef WITH_MPI
  long long nglocal = num_groups_local;
  long long ngsum;
  MPI_Scan(&nglocal, &ngsum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  const size_t num_groups_prev = (size_t)(ngsum - nglocal);
#endif /* WITH_MPI */

  if (verbose)
    message("Finding the total no. of groups took: (FOF_CLOUD SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic_num_groups_calc),
            clocks_getunit());

  /* Sort local groups into descending order of size */
  qsort(high_group_sizes, num_groups_local, sizeof(struct cloud_group_length),
        cmp_func_cloud_group_size);

  tic = getticks();

  /* Set default group ID for all particles */
  threadpool_map(&s->e->threadpool, fof_cloud_set_initial_group_id_mapper, s->parts,
                 s->nr_parts, sizeof(struct part), threadpool_auto_chunk_size,
                 (void *)&group_id_default);

  if (verbose)
    message("Setting default group ID took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

  /* Assign final group IDs to local root particles where the global root is
   * on this node and the group is large enough. Within a node IDs are
   * assigned in descending order of particle number. */
  for (size_t i = 0; i < num_groups_local; i++) {
#ifdef WITH_MPI
    parts[high_group_sizes[i].index - node_offset_cloud].fof_cloud_data.group_id =
        group_id_offset + i + num_groups_prev;
#else
    parts[high_group_sizes[i].index].fof_cloud_data.group_id = group_id_offset + i;
#endif
  }

#ifdef WITH_MPI

  /* Now, for each local root where the global root is on some other node
   * AND the total size of the group is >= min_group_size we need to
   * retrieve the parts.group_id we just assigned to the global root.
   *
   * Will do that by sending the group_index of these lcoal roots to the
   * node where their global root is stored and receiving back the new
   * group_id associated with that particle.
   *
   * Identify local roots with global root on another node and large enough
   * group_size. Store index of the local and global roots in these cases.
   *
   * NOTE: if group_size only contains the total FoF mass for global roots,
   * then we have to communicate ALL fragments where the global root is not
   * on this node. Hence the commented out extra conditions below.*/
  size_t nsend = 0;
  for (size_t i = 0; i < nr_parts; i++) {
    if ((!is_local_fof_cloud(group_index[i],
                             nr_parts))) { /* && (group_size[i] >= min_group_size)) { */
      nsend++;
    }
  }

  struct fof_cloud_final_index *fof_cloud_index_send =
      (struct fof_cloud_final_index *)swift_malloc(
          "fof_cloud_index_send", sizeof(struct fof_cloud_final_index) * nsend);
  nsend = 0;
  for (size_t i = 0; i < nr_parts; i++) {
    if ((!is_local_fof_cloud(group_index[i],
                             nr_parts))) { /* && (group_size[i] >= min_group_size)) { */
      fof_cloud_index_send[nsend].local_root = node_offset_cloud + i;
      fof_cloud_index_send[nsend].global_root = group_index[i];
      nsend++;
    }
  }

  /* Sort by global root - this puts the groups in order of which node they're
   * stored on */
  qsort(fof_cloud_index_send, nsend, sizeof(struct fof_cloud_final_index),
        compare_fof_cloud_final_index_global_root);

  /* Determine range of global indexes (i.e. particles) on each node */
  size_t *num_on_node = (size_t *)malloc(nr_nodes * sizeof(size_t));
  MPI_Allgather(&nr_parts, sizeof(size_t), MPI_BYTE, num_on_node,
                sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
  size_t *first_on_node = (size_t *)malloc(nr_nodes * sizeof(size_t));
  first_on_node[0] = 0;
  for (int i = 1; i < nr_nodes; i++)
    first_on_node[i] = first_on_node[i - 1] + num_on_node[i - 1];

  /* Determine how many entries go to each node */
  int *sendcount = (int *)malloc(nr_nodes * sizeof(int));
  for (int i = 0; i < nr_nodes; i++) sendcount[i] = 0;
  int dest = 0;
  for (size_t i = 0; i < nsend; i++) {
    while ((fof_cloud_index_send[i].global_root >=
            first_on_node[dest] + num_on_node[dest]) ||
           (num_on_node[dest] == 0)) {
      dest++;
    }
    if (dest >= nr_nodes) error("Node index out of range!");
    sendcount[dest]++;
  }

  int *recvcount = NULL, *sendoffset = NULL, *recvoffset = NULL;
  size_t nrecv = 0;

  fof_cloud_compute_send_recv_offsets(nr_nodes, sendcount, &recvcount, &sendoffset,
                                      &recvoffset, &nrecv);

  struct fof_cloud_final_index * fof_cloud_index_recv =
      (struct fof_cloud_final_index *)swift_malloc(
          "fof_cloud_index_recv", nrecv * sizeof(struct fof_cloud_final_index));

  /* Exchange group indexes */
  MPI_Alltoallv(fof_cloud_index_send, sendcount, sendoffset, fof_cloud_final_index_type,
                fof_cloud_index_recv, recvcount, recvoffset, fof_cloud_final_index_type,
                MPI_COMM_WORLD);

  /* For each received global root, look up the group ID we assigned and store
   * it in the struct */
  for (size_t i = 0; i < nrecv; i++) {
    if ((fof_cloud_index_recv[i].global_root < node_offset_cloud) ||
        (fof_cloud_index_recv[i].global_root >= node_offset_cloud + nr_parts)) {
      error("Recieved global root index out of range!");
    }
    fof_cloud_index_recv[i].global_root =
        parts[fof_cloud_index_recv[i].global_root - node_offset_cloud].fof_cloud_data.group_id;
  }

  /* Send the result back */
  MPI_Alltoallv(fof_cloud_index_recv, recvcount, recvoffset, fof_cloud_final_index_type,
                fof_cloud_index_send, sendcount, sendoffset, fof_cloud_final_index_type,
                MPI_COMM_WORLD);

  /* Update local parts.group_id */
  for (size_t i = 0; i < nsend; i++) {
    if ((fof_cloud_index_send[i].local_root < node_offset_cloud) ||
        (fof_cloud_index_send[i].local_root >= node_offset_cloud + nr_parts)) {
      error("Sent local root index out of range!");
    }
    parts[fof_cloud_index_send[i].local_root - node_offset_cloud].fof_cloud_data.group_id =
        fof_cloud_index_send[i].global_root;
  }

  free(sendcount);
  free(recvcount);
  free(sendoffset);
  free(recvoffset);
  swift_free("fof_cloud_index_send", fof_cloud_index_send);
  swift_free("fof_cloud_index_recv", fof_cloud_index_recv);

#endif /* WITH_MPI */

  /* Assign every particle the group_id of its local root. */
  for (size_t i = 0; i < nr_parts; i++) {
    const size_t root = fof_cloud_find_local(i, nr_parts, group_index);
    parts[i].fof_cloud_data.group_id = parts[root].fof_cloud_data.group_id;
  }

  if (verbose)
    message("Group sorting took: %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());

  /* Allocate and initialise a group mass and centre of mass array. */
  if (swift_memalign("fof_cloud_group_mass", (void **)&props->group_mass, 32,
                     num_groups_local * sizeof(double)) != 0)
    error("Failed to allocate list of group masses for FOF cloud search.");

  if (swift_memalign("fof_cloud_group_size", (void **)&props->final_group_size, 32,
                     num_groups_local * sizeof(long long)) != 0)
    error("Failed to allocate list of group masses for FOF cloud search.");

  if (swift_memalign("fof_cloud_group_centre_of_mass",
                     (void **)&props->group_centre_of_mass, 32,
                     num_groups_local * 3 * sizeof(double)) != 0)
    error("Failed to allocate list of group CoM for FOF cloud search.");

  if (swift_memalign("fof_cloud_group_first_position",
                     (void **)&props->group_first_position, 32,
                     num_groups_local * 3 * sizeof(double)) != 0)
    error("Failed to allocate list of group first positions for FOF cloud search.");

  bzero(props->group_mass, num_groups_local * sizeof(double));
  bzero(props->final_group_size, num_groups_local * sizeof(long long));
  bzero(props->group_centre_of_mass, num_groups_local * 3 * sizeof(double));

  for (size_t i = 0; i < 3 * num_groups_local; i++) {
    props->group_first_position[i] = -FLT_MAX;
  }

  /* Allocate and initialise arrays to identify the densest gas particle. */
  if (swift_memalign("fof_cloud_max_part_density_index",
                     (void **)&props->max_part_density_index, 32,
                     num_groups_local * sizeof(long long)) != 0)
    error(
        "Failed to allocate list of max group density indices for FOF cloud "
        "search.");

  if (swift_memalign("fof_max_part_density", (void **)&props->max_part_density,
                     32, num_groups_local * sizeof(float)) != 0)
    error("Failed to allocate list of max group densities for FOF cloud search.");

  /* No densest particle found so far */
  bzero(props->max_part_density, num_groups_local * sizeof(float));

  for (size_t i = 0; i < num_groups_local; i++) {
    props->max_part_density_index[i] = -1LL;
  }

  const ticks tic_calc_props = getticks();

#ifdef WITH_MPI
  fof_cloud_calc_group_mass(props, s, num_groups_local, num_groups_prev,
                            num_on_node, first_on_node, props->group_mass);
  free(num_on_node);
  free(first_on_node);
#else
  fof_cloud_calc_group_mass(props, s, num_groups_local, /*num_groups_prev=*/0,
                            /*num_on_node=*/NULL, /*first_on_node=*/NULL, props->group_mass);
#endif

  /* Finalise the group data before dump */
  fof_cloud_finalise_group_data(props, high_group_sizes, s->parts, s->periodic,
                                s->dim, num_groups_local);

  if (verbose)
    message("Computing group properties took: %.3f %s.",
            clocks_from_ticks(getticks() - tic_calc_props), clocks_getunit());

  /* Free the left-overs */
  swift_free("fof_cloud_high_group_sizes", high_group_sizes);
  swift_free("fof_cloud_group_mass", props->group_mass);
  swift_free("fof_cloud_group_size", props->final_group_size);
  swift_free("fof_cloud_group_centre_of_mass", props->group_centre_of_mass);
  swift_free("fof_cloud_group_first_position", props->group_first_position);
  swift_free("fof_cloud_max_part_density_index", props->max_part_density_index);
  swift_free("fof_cloud_max_part_density", props->max_part_density);
  props->group_mass = NULL;
  props->final_group_size = NULL;
  props->group_centre_of_mass = NULL;
  props->max_part_density_index = NULL;
  props->max_part_density = NULL;

  swift_free("fof_cloud_group_index", props->group_index);
  swift_free("fof_cloud_distance", props->distance_to_link);
  swift_free("fof_cloud_group_size", props->group_size);
  props->group_index = NULL;
  props->group_size = NULL;

  if (engine_rank == 0) {
    message(
        "No. of groups: %lld. No. of particles in groups: %lld. No. of "
        "particles not in groups: %lld.",
        num_groups, num_parts_in_groups,
        s->e->total_nr_parts - num_parts_in_groups);

    message("Largest group by size: %lld", max_group_size);
  }
  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic_total),
            clocks_getunit());

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

}

#endif /* WITH_FOF_CLOUD */
