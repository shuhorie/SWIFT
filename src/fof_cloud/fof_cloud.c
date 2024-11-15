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





#endif /* WITH_FOF_CLOUD */
