/*******************************************************************************
 * This file is part of SWIFT.
 * This file is developed for the on-the-fly cloud finding with FoF
 * based on Horie+2024.
 * Copyright (c) 2024 Shu Horie (shorie@ccs.tsukuba.ac.jp).
 ******************************************************************************/
#ifndef SWIFT_FOF_CLOUD_STRUCT_H
#define SWIFT_FOF_CLOUD_STRUCT_H

/* Config parameters. */
#include <config.h>

#ifdef WITH_FOF_CLOUD

/**
 * @brief Particle-carried fields for the FoF cloud cheme.
 */
struct fof_cloud_part_data {

  /*! Particle group ID */
  size_t group_id;

  /*! Size of the FOF group of this particle */
  size_t group_size;
};

#else

/**
 * @brief Particle-carried fields for the FoF cloud scheme.
 */
struct fof_cloud_part_data {};

#endif

#endif /* SWIFT_FOF_CLOUD_STRUCT_H */ 