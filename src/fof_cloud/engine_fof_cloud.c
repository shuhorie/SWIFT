/*******************************************************************************
 * This file is part of SWIFT.
 * This file is developed for the on-the-fly cloud finding with FoF
 * based on Horie+2024.
 * Copyright (c) 2024 Shu Horie (shorie@ccs.tsukuba.ac.jp).
 ******************************************************************************/

/* Config parameters. */
#include <config.h>

/* This object's header. */
#include "engine.h"

/* Local headers. */
#include "fof_cloud.h"


/**
 * @brief Activate all the FoF linking tasks for cloud finding.
 *
 * Marks all the other task types to be skipped.
 *
 * @param e The #engine to act on.
 */
void engine_activate_fof_cloud_tasks(struct engine *e) {

  const ticks tic = getticks();

  struct scheduler *s = &e->sched;
  const int nr_tasks = s->nr_tasks;
  struct task *tasks = s->tasks;

  for (int k = 0; k < nr_tasks; k++) {
    struct task *t =&tasks[k];

    if (t->type == task_type_fof_cloud_self ||
        t->type == task_type_fof_cloud_pair)
      scheduler_activate(s, t);
    else
      t->skip = 1;
  }

  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/**
 * @brief Run a FOF search to identify clouds on-the-fly.
 *
 * @param e the engine
 * @param foreign_buffers_allocated Are the foreign buffers currently
 * allocated?
 */
void engine_fof_cloud(struct engine *e,
                      const int foreign_buffers_allocated) {

  printf("This is from engine_fof_cloud!!!\n");

#ifdef WITH_FOF_CLOUD

  const ticks tic = getticks();

  /* Start by cleaning up the foreign buffers */
  if (foreign_buffers_allocated) {
#ifdef WITH_MPI
    space_free_foreign_parts(e->s, /*clear pointers=*/1);
#endif
  }

  /* Initialise FoF parameters and allocate FoF arrays */
  /* Not need to do this? */
//   fof_allocate(e->s, e->fof_cloud_properties);

  /* Make FoF cloud tasks */
  engine_make_fof_cloud_tasks(e);

  /* and activate them */
  engine_activate_fof_cloud_tasks(e);

  /* Print the number of active tasks? */
  if (e->verbose) engine_print_task_counts(e);

  /* Perfome local FOF_CLOUD tasks */
  engine_launch(e, "fof_cloud");

  /* Compute group size (only of local fragments with MPI) */
  // develop a function here

#ifdef WITH_MPI

  // MPI task here
#endif


#ifdef WITH_MPI

  /* Link the foreign fragments and finalise global group list (nothing to do
   * without MPI) */
//   fof_link_foreign_fragments(e->fof_properties, e->s);
#endif

  /* Compute group properties and act on the results */
//   fof_cloud_compute_group_props()

  /* Restore the foreign buffers as they were*/
  if (foreign_buffers_allocated) {
#ifdef WITH_MPI
    engine_allocate_foreign_particles(e, /*fof=*/0);
#endif
  }

  if (engine_rank == 0)
    message("Complete FoF search for cloud finding took: %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());

#else
  error("SWIFT was not compiled with FOF_CLOUD enabled!");
#endif
}
