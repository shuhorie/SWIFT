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
